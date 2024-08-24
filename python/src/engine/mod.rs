pub(crate) mod executor;
pub(crate) mod fs_client;
pub(crate) mod json;
pub(crate) mod parquet;

use delta_kernel::{
    DeltaResult, Engine, ExpressionEvaluator, ExpressionHandler, FileSystemClient, JsonHandler,
    ParquetHandler,
};
use executor::TaskExecutor;
use fs_client::ObjectStoreFileSystemClient;
use json::PolarsJsonHandler;
use object_store::{parse_url_opts, path::Path, DynObjectStore};
use parquet::ParquetHandler;
use std::sync::Arc;
use url::Url;

pub struct PolarsEngine<E: TaskExecutor> {
    store: Arc<DynObjectStore>,
    file_system: Arc<ObjectStoreFileSystemClient<E>>,
    json: Arc<PolarsJsonHandler>,
    parquet: Arc<parquet::ParquetHandler>,
    expression: Arc<PolarsExpressionHandler>,
}

impl<E: TaskExecutor> PolarsEngine<E> {
    /// Create a new [`DefaultEngine`] instance
    ///
    /// The `path` parameter is used to determine the type of storage used.
    ///
    /// The `task_executor` is used to spawn async IO tasks. See [executor::TaskExecutor].
    pub fn try_new<I, K, V>(path: &Url, options: I, task_executor: Arc<E>) -> DeltaResult<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let (store, prefix) = parse_url_opts(path, options)?;
        let store = Arc::new(store);
        Ok(Self {
            file_system: Arc::new(ObjectStoreFileSystemClient::new(
                store.clone(),
                prefix,
                task_executor.clone(),
            )),
            json: Arc::new(PolarsJsonHandler::new(store.clone())),
            parquet: Arc::new(ParquetHandler::new(store.clone())),
            store,
            expression: Arc::new(PolarsExpressionHandler {}),
        })
    }

    pub fn new(store: Arc<DynObjectStore>, prefix: Path, task_executor: Arc<E>) -> Self {
        Self {
            file_system: Arc::new(ObjectStoreFileSystemClient::new(
                store.clone(),
                prefix,
                task_executor.clone(),
            )),
            json: Arc::new(JsonHandler::new(store.clone(), task_executor.clone())),
            parquet: Arc::new(ParquetHandler::new(store.clone(), task_executor)),
            store,
            expression: Arc::new(PolarsExpressionHandler {}),
        }
    }

    pub fn get_object_store_for_url(&self, _url: &Url) -> Option<Arc<DynObjectStore>> {
        Some(self.store.clone())
    }
}

impl<E: TaskExecutor> Engine for PolarsEngine<E> {
    fn get_expression_handler(&self) -> Arc<dyn ExpressionHandler> {
        self.expression.clone()
    }

    fn get_file_system_client(&self) -> Arc<dyn FileSystemClient> {
        self.file_system.clone()
    }

    /// Get the connector provided [`ParquetHandler`].
    fn get_parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet.clone()
    }

    fn get_json_handler(&self) -> Arc<dyn JsonHandler> {
        self.json.clone()
    }
}
