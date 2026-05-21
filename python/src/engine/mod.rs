//! `PolarsEngine` — the four polars-backed handlers tied together as a single
//! `delta_kernel::Engine`. One per opened table; all engines share a single
//! process-wide tokio runtime so opening many tables doesn't churn worker
//! threads.

use std::sync::{Arc, OnceLock};

use delta_kernel::{
    DeltaResult, Engine, EvaluationHandler, JsonHandler, ParquetHandler, StorageHandler,
};
use tokio::runtime::Runtime;
use url::Url;

mod data;
mod handlers;

pub(crate) use data::PolarsEngineData;

use handlers::{ObjectStoreStorageHandler, PolarsJsonHandler, PolarsParquetHandler};

use crate::translation::PolarsEvaluationHandler;

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

        Ok(Self {
            storage,
            json,
            parquet,
            evaluation,
        })
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
}
