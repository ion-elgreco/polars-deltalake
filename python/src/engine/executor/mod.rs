//! `PolarsPlanExecutor` — delta-kernel `PlanExecutor` backed by polars.
//!
//! I/O operations delegate to the same `ObjectStoreStorageHandler` the
//! classic handlers use, so there is one real I/O implementation. Query
//! plans compile to a polars `LazyFrame` pipeline (see [`query`]) and
//! stream back as `PolarsEngineData` batches.

use std::sync::Arc;

use delta_kernel::plans::{IoOperation, Operation, PlanExecutor, PlanResult};
use delta_kernel::{DeltaResult, FileMeta, StorageHandler};
use polars::io::cloud::CloudOptions;
use tokio::runtime::Runtime;

use super::handlers::{ObjectStoreStorageHandler, kernel_parquet_footer};

mod query;

pub(crate) struct PolarsPlanExecutor {
    storage: Arc<ObjectStoreStorageHandler>,
    /// Shared with the parquet handler; `None` for `file://`.
    cloud_opts: Option<CloudOptions>,
    rt: &'static Runtime,
}

impl PolarsPlanExecutor {
    pub(crate) fn new(
        storage: Arc<ObjectStoreStorageHandler>,
        cloud_opts: Option<CloudOptions>,
        rt: &'static Runtime,
    ) -> Self {
        Self {
            storage,
            cloud_opts,
            rt,
        }
    }

    fn execute_io(&self, op: IoOperation) -> DeltaResult<PlanResult> {
        match op {
            IoOperation::FileListing { url } => {
                // `list_from` materializes before returning, so collecting to
                // re-box as `Send + 'static` costs nothing extra.
                let metas: Vec<DeltaResult<FileMeta>> = self.storage.list_from(&url)?.collect();
                Ok(PlanResult::FileMeta(Box::new(metas.into_iter())))
            }
            IoOperation::ReadBytes { files } => {
                let payloads: Vec<DeltaResult<bytes::Bytes>> =
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
    fn execute_op(&self, op: Operation) -> DeltaResult<PlanResult> {
        match op {
            Operation::IoOperation(io) => self.execute_io(io),
            Operation::QueryPlan(plan) => self.execute_query(plan),
        }
    }
}
