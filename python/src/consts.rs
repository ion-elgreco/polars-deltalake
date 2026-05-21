//! Shared string constants used at the boundary with delta-kernel.

/// Our polars encoding for kernel `Map<K, V>` is `List<Struct<{key, value}>>`.
pub(crate) const MAP_KEY_FIELD: &str = "key";
pub(crate) const MAP_VALUE_FIELD: &str = "value";

/// Per `EvaluationHandler` contract: non-struct and predicate outputs land
/// in a single column with this name.
pub(crate) const KERNEL_OUTPUT_COL: &str = "output";
