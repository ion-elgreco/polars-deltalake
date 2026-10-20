//! `PolarsEngine` — a `delta_kernel::Engine` whose storage, JSON and parquet
//! handlers all route through `PolarsPlanExecutor` (kernel's
//! `PlanBasedEngine` layout, without its arrow dependency). One per opened
//! table; all engines share a single process-wide tokio runtime so opening
//! many tables doesn't churn worker threads.

use std::sync::{Arc, OnceLock};

use delta_kernel::plans::PlanExecutor;
use delta_kernel::{
    Engine, EvaluationHandler, JsonHandler, KernelResult, ParquetHandler, StorageHandler,
};
use tokio::runtime::Runtime;
use url::Url;

mod data;
mod executor;
mod handlers;
mod log_cache;
mod plan_based;

pub(crate) use data::PolarsEngineData;
pub(crate) use data::resolve_path as resolve_series_path;
pub(crate) use data::{
    collect_streaming_batches, collect_streaming_batches_sized, collect_streaming_single,
    select_anchored,
};
pub(crate) use executor::PolarsPlanExecutor;

#[cfg(test)]
pub(crate) use handlers::parse_ndjson_inferred;
use handlers::{ObjectStoreStorageHandler, cloud_options_for};
pub(crate) use handlers::{
    dsl_parquet_scan, fetch_parquet_metadata, path_for_polars_io, unified_scan_args,
};
use plan_based::{PlanBasedJsonHandler, PlanBasedParquetHandler, PlanBasedStorageHandler};

use crate::translation::PolarsEvaluationHandler;

/// Rows per morsel handed downstream. polars-stream's default is ~12.5k —
/// too small; each batch pays the FFI / kernel-handoff crossing cost. 100k
/// gives polars-stream more parallelism + keeps `split_and_buffer`'s
/// single-file big wins on many-file scans.
pub(crate) const COLLECT_CHUNK_ROWS: usize = 100_000;

/// Morsel size for the data scan: every morsel pays a per-column FFI
/// crossing, so narrow frames get more rows per morsel. Never below
/// `COLLECT_CHUNK_ROWS`; at most ten times it.
pub(crate) fn scan_chunk_rows(n_cols: usize) -> usize {
    (10_000_000 / n_cols.max(1)).clamp(COLLECT_CHUNK_ROWS, 10 * COLLECT_CHUNK_ROWS)
}

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
    executor: Arc<PolarsPlanExecutor>,
    evaluation: Arc<PolarsEvaluationHandler>,
    storage: Arc<PlanBasedStorageHandler>,
    json: Arc<PlanBasedJsonHandler>,
    parquet: Arc<PlanBasedParquetHandler>,
}

impl PolarsEngine {
    /// `storage_options` forwarded to `object_store::parse_url_opts` so cloud
    /// credentials can be passed inline instead of via env vars.
    pub(crate) fn open(
        table_url: &Url,
        storage_options: impl IntoIterator<Item = (String, String)>,
    ) -> KernelResult<Self> {
        let opts: std::collections::HashMap<String, String> = storage_options.into_iter().collect();
        let storage = Arc::new(ObjectStoreStorageHandler::new(
            table_url,
            opts.clone(),
            rt(),
        )?);
        let cloud_opts = cloud_options_for(table_url, opts)?;
        let executor = Arc::new(PolarsPlanExecutor::new(storage, cloud_opts));
        let plans: Arc<dyn PlanExecutor> = executor.clone();

        Ok(Self {
            evaluation: Arc::new(PolarsEvaluationHandler::new()),
            storage: Arc::new(PlanBasedStorageHandler::new(plans.clone())),
            json: Arc::new(PlanBasedJsonHandler::new(plans.clone())),
            parquet: Arc::new(PlanBasedParquetHandler::new(plans)),
            executor,
        })
    }

    /// Pre-built cloud options, exposed so the scan-driver's bulk-read path
    /// can reuse them without rebuilding from `storage_options`.
    pub(crate) fn cloud_options(&self) -> Option<&polars::io::cloud::CloudOptions> {
        self.executor.cloud_options()
    }

    /// Concrete accessor — the `Engine::plan_executor` trait method returns
    /// an `Option` only for kernel's boundary; here it always exists.
    pub(crate) fn executor(&self) -> &Arc<PolarsPlanExecutor> {
        &self.executor
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
