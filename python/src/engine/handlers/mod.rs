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
    MetadataColumns, dsl_parquet_scan, ensure_no_field_id_matching, fetch_parquet_metadata,
    kernel_parquet_footer, path_for_polars_io, row_index_as_long, split_metadata_columns,
    unified_scan_args,
};
pub(crate) use storage::ObjectStoreStorageHandler;
