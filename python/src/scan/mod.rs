//! [`TableState`] = the opened table (snapshot + engine + schema).
//! [`TableScan`] = a scan handle for table snapshot

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use delta_kernel::expressions::{Predicate, PredicateRef};
use delta_kernel::scan::{PartitionValuesOptions, Scan};
use delta_kernel::{Engine, Snapshot, SnapshotRef};
use polars::prelude::{DataFrame, Expr, Schema as PlSchema};
use polars_plan::dsl::Engine as PolarsEngineMode;
use pyo3::prelude::*;
use pyo3_polars::PySchema;
use url::Url;

use crate::engine::{COLLECT_CHUNK_ROWS, PolarsEngine};
use crate::errors::py_err;
use crate::translation::schema::KernelSchemaExt;
use crate::translation::to_kernel::polars_expr_to_kernel_predicate;

mod cdf;
mod ffi;
mod logical;
mod plan;
mod predicate;
mod read;

pub(crate) use cdf::{CdfTableScan, CdfTableState};
pub(crate) use read::{build_lazy_scan, select_exprs_for_schema};

use ffi::{MorselState, SendExport, morsel_to_py, next_morsel};
use logical::LogicalScanIter;
use plan::{ResolvedScan, resolve_scan};
use predicate::{
    Conjunct, classify_conjuncts, conjunction, extract_expr_via_json, file_skip_via_partition_eval,
    flatten_and_conjuncts,
};

type BatchIter = Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>;

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
        let schema = self.snapshot.schema();
        let conjuncts: Vec<Conjunct> = flatten_and_conjuncts(&expr)
            .into_iter()
            .map(|c| Conjunct {
                expr: c.clone(),
                kernel_translatable: polars_expr_to_kernel_predicate(c, &schema).is_some(),
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
            self.snapshot.table_configuration().column_mapping_mode(),
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
    morsel: MorselState,
    /// File-level stats skipping in kernel.
    kernel_predicate: Option<PredicateRef>,
    /// User predicate, pre-split at top-level `AND`, each tagged with whether
    /// it was kernel-translatable. Empty when unset.
    original_predicate: Vec<Conjunct>,
    iter: Option<BatchIter>,
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
        state.morsel.n_rows = n_rows;
        match predicate {
            None => {
                state.kernel_predicate = None;
                state.original_predicate.clear();
            }
            Some(p) => {
                let expr = extract_expr_via_json(&p)?;
                let schema = self.snapshot.schema();
                // Per-conjunct so one untranslatable term doesn't disable
                // file-skipping for its siblings.
                let mut translated: Vec<Predicate> = Vec::new();
                let mut conjuncts: Vec<Conjunct> = Vec::new();
                for c in flatten_and_conjuncts(&expr) {
                    let kernel_translatable = match polars_expr_to_kernel_predicate(c, &schema) {
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
        state.morsel.reset();
        Ok(())
    }

    fn next<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let exports = py.detach(|| self.next_exports()).map_err(py_err)?;
        match exports {
            Some(exports) => morsel_to_py(py, exports).map(Some),
            None => Ok(None),
        }
    }
}

impl TableScan {
    fn build_scan(&self, state: &ScanState) -> anyhow::Result<Scan> {
        let mut sb = self
            .snapshot
            .clone()
            .scan_builder()
            // The metadata plan parses partition values into a typed struct
            // (`add.partitionValues_parsed`) the resolver turns into per-file
            // literals.
            .with_partition_values(PartitionValuesOptions::with_struct());
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

        let resolved = resolve_scan(&scan, self.engine.as_ref())?;
        if resolved.files.is_empty() {
            state.iter = Some(Box::new(std::iter::empty()));
            state.morsel.reset();
            return Ok(());
        }
        let ResolvedScan {
            mut files,
            mut path_index,
        } = resolved;

        let physical_schema = scan.physical_schema().clone();
        let select_exprs = select_exprs_for_schema(&physical_schema);

        let mode = self.snapshot.table_configuration().column_mapping_mode();
        let table_logical_schema = self.snapshot.schema();

        let routing = classify_conjuncts(
            &state.original_predicate,
            mode,
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
                state.morsel.reset();
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
        // DV/select — the common case (non-partitioned, non-DV, non-CM).
        let needs_rewrite = rewrites
            .iter()
            .any(|r| r.select.is_some() || r.dv.is_some());

        let lazy = build_lazy_scan(
            paths,
            self.engine.cloud_options(),
            &select_exprs,
            polars_predicate.as_ref(),
            &physical_schema,
            needs_rewrite,
        )?;

        let chunk_size = std::num::NonZeroUsize::new(COLLECT_CHUNK_ROWS);
        let batches = lazy
            .collect_batches(PolarsEngineMode::Streaming, true, chunk_size, false)
            .map_err(|e| anyhow::anyhow!("collect_batches failed: {e:#}"))?;
        let source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send> =
            Box::new(batches.map(|r| r.map_err(|e| anyhow::anyhow!("scan batch failed: {e:#}"))));

        let new_iter: BatchIter = if needs_rewrite {
            Box::new(
                LogicalScanIter::new(
                    source,
                    path_index,
                    rewrites,
                    conjunction(routing.post_transform),
                )
                .map(|r| r.map_err(|e| anyhow::anyhow!("scan iteration failed: {e:#}"))),
            )
        } else {
            debug_assert!(
                routing.post_transform.is_empty(),
                "post_transform requires the select list to materialize partition cols",
            );
            source
        };
        state.iter = Some(new_iter);
        state.morsel.reset();
        Ok(())
    }

    fn next_exports(&self) -> anyhow::Result<Option<Vec<SendExport>>> {
        let mut state = self.state.lock().expect("Mutex poisoned");
        if state.iter.is_none() {
            self.build_iter(&mut state)?;
        }
        let ScanState { morsel, iter, .. } = &mut *state;
        next_morsel(morsel, iter.as_mut().expect("iter set above").as_mut())
    }
}

pub(crate) fn parse_uri(uri: &str) -> anyhow::Result<Url> {
    if let Ok(url) = Url::parse(uri) {
        if url.scheme().len() != 1 {
            return Ok(url);
        }
    }
    let abs = std::path::absolute(uri).map_err(|e| anyhow::anyhow!("resolve {uri} failed: {e}"))?;
    Url::from_directory_path(&abs)
        .or_else(|_| Url::from_file_path(&abs))
        .map_err(|_| anyhow::anyhow!("failed to build file:// URL from {}", abs.display()))
}

#[cfg(test)]
mod parse_uri_tests {
    use super::parse_uri;

    #[test]
    fn remote_urls_pass_through() {
        assert_eq!(parse_uri("s3://bucket/key").unwrap().scheme(), "s3");
        assert_eq!(parse_uri("memory:///x").unwrap().scheme(), "memory");
        assert_eq!(parse_uri("file:///tmp/x").unwrap().scheme(), "file");
    }

    #[test]
    #[cfg(not(windows))]
    fn unix_local_path_becomes_file_url() {
        assert_eq!(parse_uri("/tmp/some/table").unwrap().scheme(), "file");
    }

    #[test]
    #[cfg(windows)]
    fn windows_drive_path_becomes_file_url() {
        // `C:\...` parses as a URL whose scheme is the single-letter drive ("c");
        // it must be converted to a file:// URL, not passed through as scheme "c".
        let url = parse_uri("C:\\data\\tbl").unwrap();
        assert_eq!(url.scheme(), "file");
        assert!(url.as_str().starts_with("file:///C:/"), "{url}");
    }
}
