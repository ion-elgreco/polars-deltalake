//! `DeltaSource` — pyclass driving `delta_kernel::Scan` against our engine.
//! Each `next()` yields the next polars `DataFrame` as `PyDataFrame` for
//! the Python `register_io_source` plugin.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::{ExpressionRef, Predicate, PredicateRef};
use delta_kernel::scan::Scan;
use delta_kernel::scan::state::{ScanFile, transform_to_logical};
use delta_kernel::schema::{MetadataValue, SchemaRef, StructType};
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::table_properties::TableProperties;
use delta_kernel::{Engine, Snapshot, SnapshotRef};
use polars::io::cloud::CloudOptions;
use polars::lazy::frame::LazyFrame;
use polars::prelude::{DataFrame, Expr, Schema as PlSchema};
use polars_plan::dsl::{DslBuilder, Operator, ScanSources};
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySchema};
use tokio::runtime::Runtime;
use url::Url;

use crate::engine::{PolarsEngine, PolarsEngineData, path_for_polars_io};
use crate::errors::py_err;
use crate::translation::schema::KernelSchemaExt;
use crate::translation::to_kernel::polars_expr_to_kernel_predicate;

type BatchIter = Box<dyn Iterator<Item = Result<DataFrame, delta_kernel::Error>> + Send>;

// `unsendable` because the scan iterator is `Send` but not `Sync`; Python
// only ever drives this from a single thread anyway (the one that holds the
// GIL when entering the io-source generator).
#[pyclass(module = "polars_deltalake._internal", unsendable)]
pub struct DeltaSource {
    engine: Arc<PolarsEngine>,
    snapshot: SnapshotRef,
    schema: Arc<PlSchema>,
    projection: Option<Vec<String>>,
    n_rows: Option<usize>,
    /// File-level stats skipping in kernel.
    kernel_predicate: Option<PredicateRef>,
    /// User predicate, pre-split at top-level `AND`. Empty when unset.
    original_predicate: Vec<Expr>,
    iter: Option<BatchIter>,
    rows_emitted: usize,
}

#[pymethods]
impl DeltaSource {
    #[new]
    #[pyo3(signature = (uri, version=None, storage_options=None))]
    fn new(
        uri: &str,
        version: Option<u64>,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        Self::open(uri, version, storage_options.unwrap_or_default()).map_err(py_err)
    }

    /// Polars `Schema` (DataType-keyed) of the logical table — what
    /// `register_io_source` expects.
    fn schema(&self) -> PySchema {
        PySchema(self.schema.clone())
    }

    #[pyo3(signature = (with_columns=None, n_rows=None, predicate=None))]
    fn configure(
        &mut self,
        with_columns: Option<Vec<String>>,
        n_rows: Option<usize>,
        predicate: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.projection = with_columns;
        self.n_rows = n_rows;
        match predicate {
            None => {
                self.kernel_predicate = None;
                self.original_predicate.clear();
            }
            Some(p) => {
                let expr = extract_expr_via_json(&p)?;
                let conjuncts: Vec<Expr> =
                    flatten_and_conjuncts(&expr).into_iter().cloned().collect();
                // Per-conjunct so one untranslatable term doesn't disable
                // file-skipping for its siblings.
                let translated: Vec<Predicate> = conjuncts
                    .iter()
                    .filter_map(|c| match polars_expr_to_kernel_predicate(c) {
                        Some(kp) => Some(kp),
                        None => {
                            tracing::debug!(
                                target: "polars_deltalake::pushdown",
                                conjunct = ?c,
                                "conjunct not translatable to kernel; \
                                 relying on polars-io / Python-side filter",
                            );
                            None
                        }
                    })
                    .collect();
                self.kernel_predicate =
                    (!translated.is_empty()).then(|| Arc::new(Predicate::and_from(translated)));
                self.original_predicate = conjuncts;
            }
        }
        self.iter = None;
        self.rows_emitted = 0;
        Ok(())
    }

    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        self.next_batch().map_err(py_err)
    }
}

impl DeltaSource {
    fn open(
        uri: &str,
        version: Option<u64>,
        storage_options: HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        let url = parse_uri(uri)?;
        let engine = Arc::new(
            PolarsEngine::open(&url, storage_options.into_iter())
                .map_err(|e| anyhow::anyhow!("failed to build PolarsEngine: {e:#}"))?,
        );

        let mut sb = Snapshot::builder_for(url);
        if let Some(v) = version {
            sb = sb.at_version(v);
        }
        let snapshot = sb
            .build(engine.as_ref() as &dyn Engine)
            .map_err(|e| anyhow::anyhow!("failed to build delta snapshot: {e:#}"))?;

        let schema = snapshot.schema().to_polars()?;

        Ok(Self {
            engine,
            snapshot,
            schema,
            projection: None,
            n_rows: None,
            kernel_predicate: None,
            original_predicate: Vec::new(),
            iter: None,
            rows_emitted: 0,
        })
    }

    fn build_scan(&self) -> anyhow::Result<Scan> {
        let mut sb = self.snapshot.clone().scan_builder();
        if let Some(cols) = &self.projection {
            let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
            let projected = self
                .snapshot
                .schema()
                .project_as_struct(&refs)
                .map_err(|e| anyhow::anyhow!("projection error: {e:#}"))?;
            sb = sb.with_schema(Arc::new(projected));
        }
        if let Some(pred) = &self.kernel_predicate {
            sb = sb.with_predicate(pred.clone());
        }
        sb.build()
            .map_err(|e| anyhow::anyhow!("failed to build scan: {e:#}"))
    }

    fn build_iter(&mut self) -> anyhow::Result<()> {
        let scan = self.build_scan()?;
        let engine: Arc<dyn Engine> = self.engine.clone();

        let resolved = resolve_scan(&scan, engine.as_ref())?;
        if resolved.paths.is_empty() {
            self.iter = Some(Box::new(std::iter::empty()));
            self.rows_emitted = 0;
            return Ok(());
        }
        let ResolvedScan {
            paths,
            rewrites,
            path_index,
        } = resolved;

        let physical_schema = scan.physical_schema().clone();
        let logical_schema = scan.logical_schema().clone();
        let select_exprs = select_exprs_for_schema(&physical_schema);

        // Drop partition-touching conjuncts — they'd fail parquet column
        // resolution. Python-side filter is the correctness backstop.
        let polars_predicate: Option<Expr> = {
            let column_mapped = has_column_mapping(self.snapshot.table_properties());
            let logical_schema = self.snapshot.schema();
            let survivors: Vec<Expr> = self
                .original_predicate
                .iter()
                .filter_map(|c| {
                    if column_mapped {
                        rewrite_predicate_to_physical(c, &logical_schema, &physical_schema)
                    } else {
                        predicate_only_touches_data_columns(c, &physical_schema).then(|| c.clone())
                    }
                })
                .collect();
            conjunction(survivors)
        };

        let lazy = build_lazy_scan(
            paths,
            self.engine.cloud_options(),
            &select_exprs,
            polars_predicate.as_ref(),
            &physical_schema,
        )?;

        let rt: &'static Runtime = crate::engine::rt();
        let _enter = rt.enter();
        let mut df = lazy
            .collect()
            .map_err(|e| anyhow::anyhow!("bulk scan collect failed: {e:#}"))?;
        df.rechunk_mut();

        let source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send> =
            Box::new(std::iter::once(Ok(df)));
        let logical_iter = LogicalScanIter::new(
            source,
            path_index,
            rewrites,
            engine,
            physical_schema,
            logical_schema,
        );
        self.iter = Some(Box::new(logical_iter));
        self.rows_emitted = 0;
        Ok(())
    }

    fn next_batch(&mut self) -> anyhow::Result<Option<PyDataFrame>> {
        if self.iter.is_none() {
            self.build_iter()?;
        }

        loop {
            if let Some(cap) = self.n_rows
                && self.rows_emitted >= cap
            {
                return Ok(None);
            }

            let it = self.iter.as_mut().expect("iter set above");
            let Some(next) = it.next() else {
                return Ok(None);
            };
            let mut df = next.map_err(|e| anyhow::anyhow!("scan iteration failed: {e:#}"))?;

            if df.height() == 0 {
                continue;
            }

            if let Some(cap) = self.n_rows {
                let remaining = cap.saturating_sub(self.rows_emitted);
                if df.height() > remaining {
                    df = df.head(Some(remaining));
                }
            }

            self.rows_emitted += df.height();
            return Ok(Some(PyDataFrame(df)));
        }
    }
}

pub(crate) fn parse_uri(uri: &str) -> anyhow::Result<Url> {
    if let Ok(url) = Url::parse(uri) {
        return Ok(url);
    }
    let abs = std::path::absolute(uri).map_err(|e| anyhow::anyhow!("resolve {uri} failed: {e}"))?;
    Url::from_directory_path(&abs)
        .or_else(|_| Url::from_file_path(&abs))
        .map_err(|_| anyhow::anyhow!("failed to build file:// URL from {}", abs.display()))
}

/// Per-file work to apply post-read: kernel `Transform` + DV keep-mask.
pub(crate) struct LogicalRewrite {
    /// Physical → logical (column-mapping + partition values).
    pub(crate) transform: Option<ExpressionRef>,
    /// DV keep-mask (true = keep), or `None` if the file has no DV.
    pub(crate) selection_vector: Option<Vec<bool>>,
}

/// `Scan::scan_metadata` drained into bulk-read inputs.
pub(crate) struct ResolvedScan {
    /// Files in scan order.
    pub(crate) paths: Vec<PlRefPath>,
    /// Parallel to `paths`.
    pub(crate) rewrites: Vec<LogicalRewrite>,
    /// `FILE_ID_COL` value (= `PlRefPath::as_str()`) → index in `paths` /
    /// `rewrites`.
    pub(crate) path_index: HashMap<String, usize>,
}

/// Per-row file-identity column injected via `include_file_paths`. The
/// split-and-rewrite pass reads it back to look up each row's
/// `LogicalRewrite`.
pub(crate) const FILE_ID_COL: &str = "__pldl_file__";

/// Gnarly workaround:
/// JSON instead of bincode: bincode encodes enum variants positionally, and
/// our feature subset shifts `FunctionExpr` discriminants relative to the
/// Python wheel's full-feature build — variant-name keying survives that.
fn extract_expr_via_json(predicate: &Bound<'_, PyAny>) -> PyResult<Expr> {
    let py = predicate.py();
    let pyexpr = predicate.getattr("_pyexpr")?;
    let buf = py.import("io")?.getattr("BytesIO")?.call0()?;
    pyexpr.call_method1("serialize_json", (&buf,))?;
    let bytes: Vec<u8> = buf.call_method0("getvalue")?.extract()?;
    serde_json::from_slice::<Expr>(&bytes).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("failed to deserialize polars Expr: {e}"))
    })
}

/// Split a top-level `AND` chain so each conjunct can be routed
/// independently (kernel file-skipping vs polars-io row-group pushdown).
fn flatten_and_conjuncts(expr: &Expr) -> Vec<&Expr> {
    fn walk<'a>(expr: &'a Expr, acc: &mut Vec<&'a Expr>) {
        match expr {
            Expr::BinaryExpr {
                left,
                op: Operator::And,
                right,
            } => {
                walk(left, acc);
                walk(right, acc);
            }
            other => acc.push(other),
        }
    }
    let mut out = Vec::new();
    walk(expr, &mut out);
    out
}

/// Inverse of [`flatten_and_conjuncts`]. `None` for an empty input.
fn conjunction(mut conjuncts: Vec<Expr>) -> Option<Expr> {
    let first = conjuncts.pop()?;
    Some(conjuncts.into_iter().fold(first, |acc, e| acc.and(e)))
}

/// Kernel's `StructField::physical_name(mode)` is `pub(crate)`, so read the
/// underlying metadata key ourselves.
const PHYSICAL_NAME_KEY: &str = "delta.columnMapping.physicalName";

fn physical_name(field: &delta_kernel::schema::StructField) -> &str {
    match field.metadata.get(PHYSICAL_NAME_KEY) {
        Some(MetadataValue::String(s)) => s.as_str(),
        _ => field.name.as_str(),
    }
}

fn has_column_mapping(props: &TableProperties) -> bool {
    matches!(
        props.column_mapping_mode,
        Some(ColumnMappingMode::Id | ColumnMappingMode::Name)
    )
}

/// Used to drop predicates touching partition columns — kernel adds those
/// post-read via `Transform`, so the parquet reader can't see them.
fn predicate_only_touches_data_columns(expr: &Expr, physical_schema: &StructType) -> bool {
    let phys_names: std::collections::HashSet<&str> =
        physical_schema.fields().map(|f| f.name.as_str()).collect();
    polars_plan::utils::expr_to_leaf_column_names(expr)
        .iter()
        .all(|n| phys_names.contains(n.as_str()))
}

/// Only meaningful when column mapping is active — see [`has_column_mapping`].
/// `None` if the predicate references a partition column or an unknown name.
fn rewrite_predicate_to_physical(
    expr: &Expr,
    logical_schema: &StructType,
    physical_schema: &StructType,
) -> Option<Expr> {
    let phys_names: std::collections::HashSet<&str> =
        physical_schema.fields().map(|f| f.name.as_str()).collect();
    let mut logical_to_phys: HashMap<String, PlSmallStr> = HashMap::new();
    for field in logical_schema.fields() {
        let phys = physical_name(field);
        if phys_names.contains(phys) {
            logical_to_phys.insert(field.name.to_string(), PlSmallStr::from_str(phys));
        }
    }
    let referenced = polars_plan::utils::expr_to_leaf_column_names(expr);
    if !referenced
        .iter()
        .all(|n| logical_to_phys.contains_key(n.as_str()))
    {
        return None;
    }
    let rewritten = expr.clone().map_expr(|node| match node {
        Expr::Column(name) => match logical_to_phys.get(name.as_str()) {
            Some(phys) => Expr::Column(phys.clone()),
            None => Expr::Column(name),
        },
        other => other,
    });
    Some(rewritten)
}

/// Single `scan_parquet` plan over all `paths`, with the file-id column
/// appended to `select_exprs` so it survives projection.
pub(crate) fn build_lazy_scan(
    paths: Vec<PlRefPath>,
    cloud_opts: Option<&CloudOptions>,
    select_exprs: &[Expr],
    predicate: Option<&Expr>,
    physical_schema: &StructType,
) -> anyhow::Result<LazyFrame> {
    let parquet_options = crate::engine::parquet_options(physical_schema)
        .map_err(|e| anyhow::anyhow!("kernel→polars schema conversion failed: {e:#}"))?;
    let unified_scan_args =
        crate::engine::unified_scan_args(cloud_opts, Some(PlSmallStr::from_static(FILE_ID_COL)));

    let sources = ScanSources::Paths(paths.into());
    let lazy: LazyFrame = DslBuilder::scan_parquet(sources, parquet_options, unified_scan_args)
        .map_err(|e| anyhow::anyhow!("scan_parquet plan failed: {e:#}"))?
        .build()
        .into();

    let mut select_with_file_id: Vec<Expr> = Vec::with_capacity(select_exprs.len() + 1);
    select_with_file_id.extend(select_exprs.iter().cloned());
    select_with_file_id.push(polars::prelude::col(PlSmallStr::from_static(FILE_ID_COL)));

    // Filter sits directly above the scan node so polars-io's parquet
    // predicate pushdown can pick it up.
    let mut plan = lazy;
    if let Some(pred) = predicate {
        plan = plan.filter(pred.clone());
    }
    Ok(plan.select(select_with_file_id))
}

/// `col(...)` per kernel physical-schema field. Caller appends file-id /
/// other metadata columns before handing to `build_lazy_scan`.
pub(crate) fn select_exprs_for_schema(physical_schema: &StructType) -> Vec<Expr> {
    physical_schema
        .fields()
        .map(|f| polars::prelude::col(PlSmallStr::from_str(f.name.as_str())))
        .collect()
}

/// Drain `scan.scan_metadata` into a `ResolvedScan`. Materializes each DV
/// eagerly so the read path doesn't need the engine handle.
///
/// Kernel's `ScanCallback` is `fn(&mut T, ScanFile)` (no `Result`), so
/// callback errors are stashed on the context and surfaced after each
/// `visit_scan_files` call.
pub(crate) fn resolve_scan(scan: &Scan, engine: &dyn Engine) -> anyhow::Result<ResolvedScan> {
    struct Ctx<'a> {
        engine: &'a dyn Engine,
        table_root: &'a Url,
        paths: Vec<PlRefPath>,
        rewrites: Vec<LogicalRewrite>,
        path_index: HashMap<String, usize>,
        err: Option<delta_kernel::Error>,
    }

    fn visit_one(ctx: &mut Ctx<'_>, scan_file: ScanFile) -> Result<(), delta_kernel::Error> {
        let abs = ctx.table_root.join(&scan_file.path).map_err(|e| {
            delta_kernel::Error::Generic(format!(
                "failed to resolve scan file path {}: {e}",
                scan_file.path
            ))
        })?;
        let pl_path = path_for_polars_io(&abs)?;
        let sv = if scan_file.dv_info.has_vector() {
            scan_file
                .dv_info
                .get_selection_vector(ctx.engine, ctx.table_root)?
        } else {
            None
        };

        let idx = ctx.paths.len();
        ctx.path_index.insert(pl_path.as_str().to_string(), idx);
        ctx.paths.push(pl_path);
        ctx.rewrites.push(LogicalRewrite {
            transform: scan_file.transform,
            selection_vector: sv,
        });
        Ok(())
    }

    // `ScanCallback` is a bare `fn` pointer with no `Result` return, so
    // bubble visitor errors through the context.
    fn callback(ctx: &mut Ctx<'_>, scan_file: ScanFile) {
        if ctx.err.is_some() {
            return;
        }
        if let Err(e) = visit_one(ctx, scan_file) {
            ctx.err = Some(e);
        }
    }

    let table_root = scan.table_root().clone();
    let mut ctx = Ctx {
        engine,
        table_root: &table_root,
        paths: Vec::new(),
        rewrites: Vec::new(),
        path_index: HashMap::new(),
        err: None,
    };

    for res in scan
        .scan_metadata(engine)
        .map_err(|e| anyhow::anyhow!("scan_metadata failed: {e:#}"))?
    {
        let metadata = res.map_err(|e| anyhow::anyhow!("scan_metadata item failed: {e:#}"))?;
        ctx = metadata
            .visit_scan_files(ctx, callback)
            .map_err(|e| anyhow::anyhow!("visit_scan_files failed: {e:#}"))?;
        if let Some(e) = ctx.err.take() {
            return Err(anyhow::anyhow!("scan_metadata callback failed: {e:#}"));
        }
    }

    Ok(ResolvedScan {
        paths: ctx.paths,
        rewrites: ctx.rewrites,
        path_index: ctx.path_index,
    })
}

/// Splits each bulk-read frame on `FILE_ID_COL` runs and applies the
/// per-file `LogicalRewrite`, yielding logical-schema frames in scan order.
pub(crate) struct LogicalScanIter {
    /// Raw bulk-read frames carrying `FILE_ID_COL`. One frame on the eager
    /// path; many on a streaming follow-up.
    source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
    /// `FILE_ID_COL` value → index in `rewrites`.
    path_index: HashMap<String, usize>,
    rewrites: Vec<LogicalRewrite>,
    engine: Arc<dyn Engine>,
    physical_schema: SchemaRef,
    logical_schema: SchemaRef,
    /// One inner frame may span multiple files; we slice into per-file
    /// frames here and drain before pulling the next inner frame.
    pending: VecDeque<Result<DataFrame, delta_kernel::Error>>,
}

impl LogicalScanIter {
    pub(crate) fn new(
        source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
        path_index: HashMap<String, usize>,
        rewrites: Vec<LogicalRewrite>,
        engine: Arc<dyn Engine>,
        physical_schema: SchemaRef,
        logical_schema: SchemaRef,
    ) -> Self {
        Self {
            source,
            path_index,
            rewrites,
            engine,
            physical_schema,
            logical_schema,
            pending: VecDeque::new(),
        }
    }

    /// Slice on file-id boundaries, push each per-file logical frame onto
    /// `pending`. Boundary detection runs through polars' vectorized
    /// `rle`, which returns one row per run with `{len, value}`.
    fn split_and_buffer(&mut self, df: DataFrame) -> Result<(), delta_kernel::Error> {
        if df.height() == 0 {
            return Ok(());
        }
        let file_col = df
            .column(FILE_ID_COL)
            .map_err(|e| delta_kernel::Error::Generic(format!("file-id column missing: {e}")))?;
        let runs = polars::prelude::rle(file_col).map_err(|e| {
            delta_kernel::Error::Generic(format!("rle on file-id column failed: {e}"))
        })?;
        // `rle` returns `Struct{ len: u32, value: <input dtype> }` — fixed
        // by polars contract, so post-rle field/type lookups are infallible.
        let runs = runs.struct_().expect("rle returns Struct");
        let lens = runs
            .field_by_name(polars::prelude::RLE_LENGTH_COLUMN_NAME)
            .expect("rle Struct has length field");
        let vals = runs
            .field_by_name(polars::prelude::RLE_VALUE_COLUMN_NAME)
            .expect("rle Struct has value field");
        let lens = lens.u32().expect("rle length is u32");
        let vals = vals.str().expect("rle value is String");

        let mut offset: usize = 0;
        for (len_opt, val_opt) in lens.iter().zip(vals.iter()) {
            let len = len_opt.unwrap_or(0) as usize;
            if let Some(file_id) = val_opt {
                let sub = df.slice(offset as i64, len);
                let out = self.apply_rewrite(file_id, sub);
                self.pending.push_back(out);
            }
            offset += len;
        }
        Ok(())
    }

    /// Drop the file-id column, apply DV + `Transform`, return logical frame.
    fn apply_rewrite(
        &mut self,
        file_id: &str,
        mut df: DataFrame,
    ) -> Result<DataFrame, delta_kernel::Error> {
        df.drop_in_place(FILE_ID_COL)
            .map_err(|e| delta_kernel::Error::Generic(format!("drop {FILE_ID_COL}: {e}")))?;

        let idx = *self.path_index.get(file_id).ok_or_else(|| {
            delta_kernel::Error::Generic(format!("unknown file_id from polars-io scan: {file_id}"))
        })?;
        // SV applied exactly once per file — take it instead of cloning. For
        // a million-row DV file this saves a per-batch `Vec<bool>` allocation.
        let rewrite = &mut self.rewrites[idx];
        let sv = rewrite.selection_vector.take();
        let transform = rewrite.transform.clone();

        let mut physical: Box<dyn EngineData> = Box::new(PolarsEngineData::new(df));
        if let Some(sv) = sv {
            physical = physical.apply_selection_vector(sv)?;
        }

        let logical = transform_to_logical(
            self.engine.as_ref(),
            physical,
            &self.physical_schema,
            &self.logical_schema,
            transform,
        )?;

        let out = logical
            .into_any()
            .downcast::<PolarsEngineData>()
            .map_err(|_| {
                delta_kernel::Error::Generic(
                    "transform_to_logical returned non-PolarsEngineData".into(),
                )
            })?;
        Ok(out.into_inner())
    }
}

impl Iterator for LogicalScanIter {
    type Item = Result<DataFrame, delta_kernel::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.pending.pop_front() {
                return Some(item);
            }
            let raw = self.source.next()?;
            match raw {
                Err(e) => return Some(Err(delta_kernel::Error::Generic(format!("{e:#}")))),
                Ok(df) => {
                    if let Err(e) = self.split_and_buffer(df) {
                        return Some(Err(e));
                    }
                }
            }
        }
    }
}
