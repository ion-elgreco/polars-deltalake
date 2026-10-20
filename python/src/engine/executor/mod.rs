//! `PolarsPlanExecutor` — delta-kernel `PlanExecutor` backed by polars.
//!
//! I/O operations delegate to `ObjectStoreStorageHandler`, the one real I/O
//! implementation; the engine's kernel-facing handlers are thin wrappers
//! over this executor. Query plans compile to a polars `LazyFrame` pipeline
//! (see [`query`]) and stream back as `PolarsEngineData` batches.

use std::sync::Arc;

use delta_kernel::plans::{IoOperation, Operation, PlanExecutor, PlanResult};
use delta_kernel::{FileMeta, KernelResult, StorageHandler};
use polars::io::cloud::CloudOptions;

use super::handlers::{ObjectStoreStorageHandler, kernel_parquet_footer};

mod query;

pub(crate) struct PolarsPlanExecutor {
    storage: Arc<ObjectStoreStorageHandler>,
    /// Shared with the parquet handler; `None` for `file://`.
    cloud_opts: Option<CloudOptions>,
}

impl PolarsPlanExecutor {
    pub(crate) fn new(
        storage: Arc<ObjectStoreStorageHandler>,
        cloud_opts: Option<CloudOptions>,
    ) -> Self {
        Self {
            storage,
            cloud_opts,
        }
    }

    pub(crate) fn cloud_options(&self) -> Option<&CloudOptions> {
        self.cloud_opts.as_ref()
    }

    fn execute_io(&self, op: IoOperation) -> KernelResult<PlanResult> {
        match op {
            IoOperation::FileListing { url } => {
                // `list_from` materializes before returning, so collecting to
                // re-box as `Send + 'static` costs nothing extra.
                let metas: Vec<KernelResult<FileMeta>> = self.storage.list_from(&url)?.collect();
                Ok(PlanResult::FileMeta(Box::new(metas.into_iter())))
            }
            IoOperation::ReadBytes { files } => {
                let payloads: Vec<KernelResult<bytes::Bytes>> =
                    self.storage.read_files(files)?.collect();
                Ok(PlanResult::Bytes(Box::new(payloads.into_iter())))
            }
            IoOperation::WriteBytes {
                url,
                data,
                overwrite,
            } => {
                self.storage.put(&url, data, overwrite)?;
                Ok(PlanResult::Unit)
            }
            IoOperation::HeadFile { url } => {
                let meta = self.storage.head(&url)?;
                Ok(PlanResult::FileMeta(Box::new(std::iter::once(Ok(meta)))))
            }
            IoOperation::AtomicCopy {
                source,
                destination,
            } => {
                self.storage.copy_atomic(&source, &destination)?;
                Ok(PlanResult::Unit)
            }
            IoOperation::ParquetFooter { file } => {
                let footer = kernel_parquet_footer(self.storage.as_ref(), &file)?;
                Ok(PlanResult::ParquetFooter(footer))
            }
        }
    }
}

impl PlanExecutor for PolarsPlanExecutor {
    fn execute_op(&self, op: Operation) -> KernelResult<PlanResult> {
        match op {
            Operation::IoOperation(io) => self.execute_io(io),
            Operation::QueryPlan(plan) => self.execute_query(plan),
        }
    }
}
