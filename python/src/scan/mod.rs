//! `DeltaSource` — pyclass driving `delta_kernel::Scan` against our engine.
//! Each `next()` yields the next polars `DataFrame` as `PyDataFrame` for
//! the Python `register_io_source` plugin.

use std::collections::HashMap;
use std::sync::Arc;

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
    conjunction, extract_expr_via_json, flatten_and_conjuncts, has_column_mapping,
    predicate_only_touches_data_columns, rewrite_predicate_to_physical,
};

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
        // `maintain_order=true` keeps file-id runs contiguous, which the
        // `rle` split + DV-prefix consumption in `LogicalScanIter` requires.
        let batches = lazy
            .collect_batches(PolarsEngineMode::Streaming, true, None, false)
            .map_err(|e| anyhow::anyhow!("collect_batches failed: {e:#}"))?;
        let source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send> =
            Box::new(batches.map(|r| r.map_err(|e| anyhow::anyhow!("scan batch failed: {e:#}"))));
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
