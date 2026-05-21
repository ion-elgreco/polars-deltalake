//! Polars-backed implementations of three of kernel's engine handler traits
//! (`StorageHandler`, `JsonHandler`, `ParquetHandler`). The fourth handler,
//! `EvaluationHandler`, lives in [`crate::translation`] because its body
//! is mostly kernel↔polars expression translation, shared with the rest of
//! the translation layer.

mod json;
mod parquet;
mod storage;

pub(super) use json::PolarsJsonHandler;
pub(super) use parquet::PolarsParquetHandler;
pub(super) use storage::ObjectStoreStorageHandler;
