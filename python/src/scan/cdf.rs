//! Change Data Feed scan over `delta_kernel::table_changes::TableChanges`.
//! Predicates: kernel-translatable, data-only conjuncts push for file skip;
//! the full polars predicate is re-applied per batch (kernel's filter is
//! best-effort) and CDF-metadata-column conjuncts never push — kernel#525.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use delta_kernel::Engine;
use delta_kernel::expressions::{Predicate, PredicateRef};
use delta_kernel::table_changes::TableChanges;
use polars::prelude::{DataFrame, Expr, IntoLazy, Schema as PlSchema};
use pyo3::prelude::*;
use pyo3_polars::PySchema;


use crate::engine::{PolarsEngine, PolarsEngineData};
use crate::errors::py_err;
use crate::translation::schema::KernelSchemaExt;
use crate::translation::to_kernel::polars_expr_to_kernel_predicate;

use super::ffi::{MorselState, SendExport, morsel_to_py, next_morsel};
use super::parse_uri;
use super::predicate::{extract_expr_via_json, flatten_and_conjuncts};

type BatchIter = Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>;

// Kernel's `CHANGE_TYPE_COL_NAME` / `COMMIT_VERSION_COL_NAME` /
// `COMMIT_TIMESTAMP_COL_NAME`
const CDF_META_COLS: [&str; 3] = ["_change_type", "_commit_version", "_commit_timestamp"];

#[pyclass(frozen, module = "polars_deltalake._internal")]
pub struct CdfTableState {
    engine: Arc<PolarsEngine>,
    table_changes: Arc<TableChanges>,
    schema: Arc<PlSchema>,
}

#[pymethods]
impl CdfTableState {
    #[new]
    #[pyo3(signature = (uri, start_version, end_version=None, storage_options=None))]
    fn new(
        uri: &str,
        start_version: u64,
        end_version: Option<u64>,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        Self::open(
            uri,
            start_version,
            end_version,
            storage_options.unwrap_or_default(),
        )
        .map_err(py_err)
    }

    fn schema(&self) -> PySchema {
        PySchema(self.schema.clone())
    }

    fn start_version(&self) -> u64 {
        self.table_changes.start_version()
    }

    fn end_version(&self) -> u64 {
        self.table_changes.end_version()
    }
}

impl CdfTableState {
    fn open(
        uri: &str,
        start_version: u64,
        end_version: Option<u64>,
        storage_options: HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        let url = parse_uri(uri)?;
        let engine = Arc::new(
            PolarsEngine::open(&url, storage_options.into_iter())
                .map_err(|e| anyhow::anyhow!("failed to build PolarsEngine: {e:#}"))?,
        );
        let table_changes = TableChanges::try_new(
            url,
            engine.as_ref() as &dyn Engine,
            start_version,
            end_version,
        )
        .map_err(|e| anyhow::anyhow!("failed to build TableChanges: {e:#}"))?;
        let schema = table_changes.schema().to_polars()?;
        Ok(Self {
            engine,
            table_changes: Arc::new(table_changes),
            schema,
        })
    }
}

#[pyclass(frozen, module = "polars_deltalake._internal")]
pub struct CdfTableScan {
    engine: Arc<PolarsEngine>,
    table_changes: Arc<TableChanges>,
    state: Mutex<ScanState>,
}

#[derive(Default)]
struct ScanState {
    projection: Option<Vec<String>>,
    morsel: MorselState,
    kernel_predicate: Option<PredicateRef>,
    polars_predicate: Option<Expr>,
    iter: Option<BatchIter>,
}

#[pymethods]
impl CdfTableScan {
    #[new]
    fn new(state: &CdfTableState) -> Self {
        Self {
            engine: state.engine.clone(),
            table_changes: state.table_changes.clone(),
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
        state.kernel_predicate = None;
        state.polars_predicate = None;

        if let Some(p) = predicate {
            let expr = extract_expr_via_json(&p)?;
            let schema = self.table_changes.schema();
            let mut translated: Vec<Predicate> = Vec::new();
            for c in flatten_and_conjuncts(&expr) {
                if touches_cdf_metadata(c) {
                    continue;
                }
                if let Some(kp) = polars_expr_to_kernel_predicate(c, schema) {
                    translated.push(kp);
                }
            }
            state.kernel_predicate =
                (!translated.is_empty()).then(|| Arc::new(Predicate::and_from(translated)));
            state.polars_predicate = Some(expr);
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

impl CdfTableScan {
    fn build_iter(&self, state: &mut ScanState) -> anyhow::Result<()> {
        let mut builder = self.table_changes.clone().scan_builder();
        if let Some(cols) = &state.projection {
            let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
            let projected = self
                .table_changes
                .schema()
                .project(&refs)
                .map_err(|e| anyhow::anyhow!("CDF projection error: {e:#}"))?;
            builder = builder.with_schema(projected);
        }
        if let Some(pred) = &state.kernel_predicate {
            builder = builder.with_predicate(pred.clone());
        }
        let scan = builder
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build TableChangesScan: {e:#}"))?;

        let engine: Arc<dyn Engine> = self.engine.clone();
        let raw = scan
            .execute(engine)
            .map_err(|e| anyhow::anyhow!("TableChangesScan::execute failed: {e:#}"))?;

        let polars_predicate = state.polars_predicate.clone();
        let iter: BatchIter = Box::new(raw.map(move |batch| -> anyhow::Result<DataFrame> {
            let batch = batch.map_err(|e| anyhow::anyhow!("CDF batch iteration failed: {e:#}"))?;
            let pl_data: Box<PolarsEngineData> = batch.into_any().downcast().map_err(|_| {
                anyhow::anyhow!("CDF batch is not PolarsEngineData — wrong engine wired up?")
            })?;
            let mut df = pl_data.into_inner();
            if let Some(pred) = &polars_predicate {
                df = crate::engine::collect_streaming_single(df.lazy().filter(pred.clone()))
                    .map_err(|e| anyhow::anyhow!("CDF post-filter failed: {e:#}"))?;
            }
            Ok(df)
        }));

        state.iter = Some(iter);
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

fn touches_cdf_metadata(expr: &Expr) -> bool {
    polars_plan::utils::expr_to_leaf_column_names(expr)
        .iter()
        .any(|n| CDF_META_COLS.contains(&n.as_str()))
}
