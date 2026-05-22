//! [`TableState`] = the opened table (snapshot + engine + schema).
//! [`TableScan`] = a scan handle for table snapshot

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use delta_kernel::expressions::{Predicate, PredicateRef};
use delta_kernel::scan::Scan;
use delta_kernel::{Engine, Snapshot, SnapshotRef};
use polars::prelude::{DataFrame, Expr, Schema as PlSchema};
use polars_plan::dsl::Engine as PolarsEngineMode;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySchema};
use tokio::runtime::Runtime;
use url::Url;

use crate::engine::PolarsEngine;
use crate::errors::py_err;
use crate::translation::schema::KernelSchemaExt;
use crate::translation::to_kernel::polars_expr_to_kernel_predicate;

mod logical;
mod plan;
mod predicate;
mod read;

pub(crate) use read::{build_lazy_scan, select_exprs_for_schema};

use logical::LogicalScanIter;
use plan::{ResolvedScan, resolve_scan};
use predicate::{
    Conjunct, classify_conjuncts, conjunction, extract_expr_via_json, file_skip_via_partition_eval,
    flatten_and_conjuncts, has_column_mapping,
};

type BatchIter = Box<dyn Iterator<Item = Result<DataFrame, delta_kernel::Error>> + Send>;

/// Rows per morsel handed to the Python plugin. polars-stream's default is
/// ~12.5k — too small; each batch pays the FFI crossing cost. 100k gives
/// polars-stream more parallelism + keeps `split_and_buffer`'s single-file
/// big wins on many-file scans.
const COLLECT_CHUNK_ROWS: usize = 100_000;

#[pyclass(frozen, module = "polars_deltalake._internal")]
pub struct TableState {
    engine: Arc<PolarsEngine>,
    snapshot: SnapshotRef,
    schema: Arc<PlSchema>,
}

#[pymethods]
impl TableState {
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

    /// Test / debug helper: classify a predicate by where each conjunct
    /// would be routed, without running a scan.
    fn _classify_predicate(&self, predicate: Bound<'_, PyAny>) -> PyResult<HashMap<String, usize>> {
        let expr = extract_expr_via_json(&predicate)?;
        let conjuncts: Vec<Conjunct> = flatten_and_conjuncts(&expr)
            .into_iter()
            .map(|c| Conjunct {
                expr: c.clone(),
                kernel_translatable: polars_expr_to_kernel_predicate(c).is_some(),
            })
            .collect();
        let scan = self
            .snapshot
            .clone()
            .scan_builder()
            .build()
            .map_err(|e| py_err(anyhow::anyhow!("failed to build scan: {e:#}")))?;
        let logical_schema = self.snapshot.schema();
        let routing = classify_conjuncts(
            &conjuncts,
            has_column_mapping(self.snapshot.table_properties()),
            &logical_schema,
            scan.physical_schema(),
        );
        Ok(HashMap::from([
            ("kernel".to_string(), routing.kernel.len()),
            ("parquet_filter".to_string(), routing.parquet_filter.len()),
            ("partition_prune".to_string(), routing.partition_prune.len()),
            ("post_transform".to_string(), routing.post_transform.len()),
        ]))
    }
}

impl TableState {
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
        })
    }
}

#[pyclass(frozen, module = "polars_deltalake._internal")]
pub struct TableScan {
    engine: Arc<PolarsEngine>,
    snapshot: SnapshotRef,
    state: Mutex<ScanState>,
}

#[derive(Default)]
struct ScanState {
    projection: Option<Vec<String>>,
    n_rows: Option<usize>,
    /// File-level stats skipping in kernel.
    kernel_predicate: Option<PredicateRef>,
    /// User predicate, pre-split at top-level `AND`, each tagged with whether
    /// it was kernel-translatable. Empty when unset.
    original_predicate: Vec<Conjunct>,
    iter: Option<BatchIter>,
    rows_emitted: usize,
}

#[pymethods]
impl TableScan {
    #[new]
    fn new(state: &TableState) -> Self {
        Self {
            engine: state.engine.clone(),
            snapshot: state.snapshot.clone(),
            state: Mutex::new(ScanState::default()),
        }
    }

    #[pyo3(signature = (with_columns=None, n_rows=None, predicate=None))]
    fn configure(
        &self,
        with_columns: Option<Vec<String>>,
        n_rows: Option<usize>,
        predicate: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut state = self.state.lock().expect("Mutex poisoned");
        state.projection = with_columns;
        state.n_rows = n_rows;
        match predicate {
            None => {
                state.kernel_predicate = None;
                state.original_predicate.clear();
            }
            Some(p) => {
                let expr = extract_expr_via_json(&p)?;
                // Per-conjunct so one untranslatable term doesn't disable
                // file-skipping for its siblings.
                let mut translated: Vec<Predicate> = Vec::new();
                let mut conjuncts: Vec<Conjunct> = Vec::new();
                for c in flatten_and_conjuncts(&expr) {
                    let kernel_translatable = match polars_expr_to_kernel_predicate(c) {
                        Some(kp) => {
                            translated.push(kp);
                            true
                        }
                        None => {
                            tracing::debug!(
                                target: "polars_deltalake::pushdown",
                                conjunct = ?c,
                                "conjunct not translatable to kernel; \
                                 relying on polars-io / Python-side filter",
                            );
                            false
                        }
                    };
                    conjuncts.push(Conjunct {
                        expr: c.clone(),
                        kernel_translatable,
                    });
                }
                state.kernel_predicate =
                    (!translated.is_empty()).then(|| Arc::new(Predicate::and_from(translated)));
                state.original_predicate = conjuncts;
            }
        }
        state.iter = None;
        state.rows_emitted = 0;
        Ok(())
    }

    fn next(&self, py: Python<'_>) -> PyResult<Option<PyDataFrame>> {
        py.detach(|| self.next_batch()).map_err(py_err)
    }
}

impl TableScan {
    fn build_scan(&self, state: &ScanState) -> anyhow::Result<Scan> {
        let mut sb = self.snapshot.clone().scan_builder();
        if let Some(cols) = &state.projection {
            let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
            let projected = self
                .snapshot
                .schema()
                .project_as_struct(&refs)
                .map_err(|e| anyhow::anyhow!("projection error: {e:#}"))?;
            sb = sb.with_schema(Arc::new(projected));
        }
        if let Some(pred) = &state.kernel_predicate {
            sb = sb.with_predicate(pred.clone());
        }
        sb.build()
            .map_err(|e| anyhow::anyhow!("failed to build scan: {e:#}"))
    }

    fn build_iter(&self, state: &mut ScanState) -> anyhow::Result<()> {
        let scan = self.build_scan(state)?;
        let engine: Arc<dyn Engine> = self.engine.clone();

        let resolved = resolve_scan(&scan, engine.as_ref())?;
        if resolved.files.is_empty() {
            state.iter = Some(Box::new(std::iter::empty()));
            state.rows_emitted = 0;
            return Ok(());
        }
        let ResolvedScan {
            mut files,
            mut path_index,
        } = resolved;

        let physical_schema = scan.physical_schema().clone();
        let logical_schema = scan.logical_schema().clone();
        let select_exprs = select_exprs_for_schema(&physical_schema);

        let column_mapped = has_column_mapping(self.snapshot.table_properties());
        let table_logical_schema = self.snapshot.schema();

        let routing = classify_conjuncts(
            &state.original_predicate,
            column_mapped,
            &table_logical_schema,
            &physical_schema,
        );

        if !routing.partition_prune.is_empty() {
            let surviving = file_skip_via_partition_eval(
                &routing.partition_prune,
                &files,
                &table_logical_schema,
            )?;
            if surviving.is_empty() {
                state.iter = Some(Box::new(std::iter::empty()));
                state.rows_emitted = 0;
                return Ok(());
            }
            if surviving.len() < files.len() {
                files = files
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, f)| surviving.contains(&i).then_some(f))
                    .collect();
                path_index = files
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (f.path.as_str().to_string(), i))
                    .collect();
            }
        }

        let polars_predicate: Option<Expr> = conjunction(routing.parquet_filter);

        let (paths, rewrites): (Vec<_>, Vec<_>) =
            files.into_iter().map(|f| (f.path, f.rewrite)).unzip();

        // Skip the file-id column + `LogicalScanIter` when no file needs
        // DV/Transform — the common case (non-partitioned, non-DV, non-CM).
        let needs_rewrite = rewrites
            .iter()
            .any(|r| r.transform.is_some() || r.dv.is_some());

        let lazy = build_lazy_scan(
            paths,
            self.engine.cloud_options(),
            &select_exprs,
            polars_predicate.as_ref(),
            &physical_schema,
            needs_rewrite,
        )?;

        let rt: &'static Runtime = crate::engine::rt();
        let _enter = rt.enter();
        let chunk_size = std::num::NonZeroUsize::new(COLLECT_CHUNK_ROWS);
        let batches = lazy
            .collect_batches(PolarsEngineMode::Streaming, true, chunk_size, false)
            .map_err(|e| anyhow::anyhow!("collect_batches failed: {e:#}"))?;
        let source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send> =
            Box::new(batches.map(|r| r.map_err(|e| anyhow::anyhow!("scan batch failed: {e:#}"))));

        let new_iter: BatchIter = if needs_rewrite {
            Box::new(LogicalScanIter::new(
                source,
                path_index,
                rewrites,
                engine,
                physical_schema,
                logical_schema,
                conjunction(routing.post_transform),
            ))
        } else {
            debug_assert!(
                routing.post_transform.is_empty(),
                "post_transform requires Transform to materialize partition cols",
            );
            Box::new(source.map(|r| r.map_err(|e| delta_kernel::Error::Generic(format!("{e:#}")))))
        };
        state.iter = Some(new_iter);
        state.rows_emitted = 0;
        Ok(())
    }

    fn next_batch(&self) -> anyhow::Result<Option<PyDataFrame>> {
        let mut state = self.state.lock().expect("Mutex poisoned");
        if state.iter.is_none() {
            self.build_iter(&mut state)?;
        }

        loop {
            if let Some(cap) = state.n_rows
                && state.rows_emitted >= cap
            {
                return Ok(None);
            }

            let it = state.iter.as_mut().expect("iter set above");
            let Some(next) = it.next() else {
                return Ok(None);
            };
            let mut df = next.map_err(|e| anyhow::anyhow!("scan iteration failed: {e:#}"))?;

            if df.height() == 0 {
                continue;
            }

            if let Some(cap) = state.n_rows {
                let remaining = cap.saturating_sub(state.rows_emitted);
                if df.height() > remaining {
                    df = df.head(Some(remaining));
                }
            }

            state.rows_emitted += df.height();
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
