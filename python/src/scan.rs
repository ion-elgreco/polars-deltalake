//! `DeltaSource` — pyclass driving `delta_kernel::Scan` against our engine.
//! Each `next()` yields the next polars `DataFrame` as `PyDataFrame` for
//! the Python `register_io_source` plugin.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::scan::Scan;
use delta_kernel::{Engine, Snapshot, SnapshotRef};
use polars::prelude::{DataFrame, Schema as PlSchema};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr, PySchema};
use url::Url;

use crate::engine::{PolarsEngine, PolarsEngineData};
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
    predicate: Option<PredicateRef>,
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
    ) {
        self.projection = with_columns;
        self.n_rows = n_rows;
        self.predicate = predicate
            .and_then(|p| p.extract::<PyExpr>().ok())
            .and_then(|pe| polars_expr_to_kernel_predicate(&pe.0))
            .map(Arc::new);
        self.iter = None;
        self.rows_emitted = 0;
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
            predicate: None,
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
        if let Some(pred) = &self.predicate {
            sb = sb.with_predicate(pred.clone());
        }
        sb.build()
            .map_err(|e| anyhow::anyhow!("failed to build scan: {e:#}"))
    }

    fn build_iter(&mut self) -> anyhow::Result<()> {
        let scan = self.build_scan()?;
        let engine: Arc<dyn Engine> = self.engine.clone();
        let raw = scan
            .execute(engine)
            .map_err(|e| anyhow::anyhow!("scan.execute failed: {e:#}"))?;

        let it = raw.map(|res| -> Result<DataFrame, delta_kernel::Error> {
            let data: Box<dyn EngineData> = res?;
            let any = data.into_any();
            let polars_data: Box<PolarsEngineData> =
                any.downcast::<PolarsEngineData>().map_err(|_| {
                    delta_kernel::Error::Generic(
                        "scan emitted EngineData that is not PolarsEngineData".into(),
                    )
                })?;
            Ok(polars_data.into_inner())
        });

        self.iter = Some(Box::new(it));
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
