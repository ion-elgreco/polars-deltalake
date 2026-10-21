//! The polars-io code behind `PolarsPlanExecutor`: the object_store
//! `StorageHandler` its I/O operations run on, the NDJSON parse / align
//! helpers behind `ScanJson` and `parse_json`, and the parquet scan /
//! footer helpers behind `ScanParquet`. Kernel never sees these directly —
//! the engine's handlers (`engine::plan_based`) are thin wrappers over the
//! executor. The fourth handler, `EvaluationHandler`, lives in
//! [`crate::translation`] because its body is mostly kernel↔polars
//! expression translation, shared with the rest of the translation layer.

mod json;
mod parquet;
mod storage;

#[cfg(test)]
pub(crate) use json::parse_ndjson_inferred;
pub(crate) use json::{align_lazy, parse_commit_files, parse_json_strings};
pub(crate) use parquet::{
    MetadataColumns, cloud_options_for, dsl_parquet_scan, ensure_no_field_id_matching,
    fetch_parquet_metadata, kernel_parquet_footer, metadata_only_frame, path_for_polars_io,
    row_index_as_long, split_metadata_columns, unified_scan_args,
};
pub(crate) use storage::ObjectStoreStorageHandler;
