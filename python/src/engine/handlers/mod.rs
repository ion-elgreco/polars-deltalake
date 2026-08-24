//! Polars-backed implementations of three of kernel's engine handler traits
//! (`StorageHandler`, `JsonHandler`, `ParquetHandler`). The fourth handler,
//! `EvaluationHandler`, lives in [`crate::translation`] because its body
//! is mostly kernel↔polars expression translation, shared with the rest of
//! the translation layer.

mod json;
mod parquet;
mod storage;

pub(super) use json::PolarsJsonHandler;
pub(crate) use json::{align_lazy, parse_ndjson_inferred};
pub(super) use parquet::PolarsParquetHandler;
pub(crate) use parquet::{
    ensure_no_field_id_matching, kernel_parquet_footer, parquet_options, path_for_polars_io,
    unified_scan_args,
};
pub(crate) use storage::ObjectStoreStorageHandler;
