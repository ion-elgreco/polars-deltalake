//! `PolarsEngine` — the four polars-backed handlers tied together as a single
//! `delta_kernel::Engine`. One per opened table; all engines share a single
//! process-wide tokio runtime so opening many tables doesn't churn worker
//! threads.

use std::sync::{Arc, OnceLock};

use delta_kernel::plans::PlanExecutor;
use delta_kernel::{
    DeltaResult, Engine, EvaluationHandler, JsonHandler, ParquetHandler, StorageHandler,
};
use tokio::runtime::Runtime;
use url::Url;

mod data;
mod executor;
mod handlers;

pub(crate) use data::PolarsEngineData;
pub(crate) use data::resolve_path as resolve_series_path;
pub(crate) use executor::PolarsPlanExecutor;

use handlers::{ObjectStoreStorageHandler, PolarsJsonHandler, PolarsParquetHandler};
pub(crate) use handlers::{parquet_options, path_for_polars_io, unified_scan_args};

use crate::translation::PolarsEvaluationHandler;

/// Rows per morsel handed downstream. polars-stream's default is ~12.5k —
/// too small; each batch pays the FFI / kernel-handoff crossing cost. 100k
/// gives polars-stream more parallelism + keeps `split_and_buffer`'s
/// single-file big wins on many-file scans.
pub(crate) const COLLECT_CHUNK_ROWS: usize = 100_000;

/// Process-wide tokio runtime shared across every `PolarsEngine` instance.
pub(crate) fn rt() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .max(8);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("polars-deltalake")
            .worker_threads(workers)
            .build()
            .expect("Failed to create a tokio runtime.");
        tracing::info!(
            target: "polars_deltalake::runtime",
            runtime = "polars-deltalake",
            workers = rt.metrics().num_workers(),
            "tokio runtime initialised"
        );
        rt
    })
}

pub(crate) struct PolarsEngine {
    storage: Arc<ObjectStoreStorageHandler>,
    json: Arc<PolarsJsonHandler>,
    parquet: Arc<PolarsParquetHandler>,
    evaluation: Arc<PolarsEvaluationHandler>,
    executor: Arc<PolarsPlanExecutor>,
}

impl PolarsEngine {
    /// `storage_options` forwarded to `object_store::parse_url_opts` so cloud
    /// credentials can be passed inline instead of via env vars.
    pub(crate) fn open(
        table_url: &Url,
        storage_options: impl IntoIterator<Item = (String, String)>,
    ) -> DeltaResult<Self> {
        let rt = rt();
        let opts: std::collections::HashMap<String, String> = storage_options.into_iter().collect();

        let storage = Arc::new(ObjectStoreStorageHandler::new(table_url, opts.clone(), rt)?);

        let json = Arc::new(PolarsJsonHandler::new(storage.clone()));
        let parquet = Arc::new(PolarsParquetHandler::new(storage.clone(), opts, rt)?);
        let evaluation = Arc::new(PolarsEvaluationHandler::new());
        let executor = Arc::new(PolarsPlanExecutor::new(
            storage.clone(),
            parquet.cloud_options().cloned(),
            rt,
        ));

        Ok(Self {
            storage,
            json,
            parquet,
            evaluation,
            executor,
        })
    }

    /// Pre-built cloud options the parquet handler holds, exposed so the
    /// scan-driver's bulk-read path can reuse them without rebuilding from
    /// `storage_options`.
    pub(crate) fn cloud_options(&self) -> Option<&polars::io::cloud::CloudOptions> {
        self.parquet.cloud_options()
    }
}

impl Engine for PolarsEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.evaluation.clone()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.storage.clone()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.json.clone()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet.clone()
    }

    /// Opts kernel into declarative-plan execution: snapshot P&M replay and
    /// (via `declarative_metadata_scan_plan`) scan-file log replay run as
    /// polars queries instead of per-file handler calls.
    fn plan_executor(&self) -> Option<Arc<dyn PlanExecutor>> {
        Some(self.executor.clone())
    }
}
