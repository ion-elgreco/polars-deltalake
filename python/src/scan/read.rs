//! Bulk-read plan construction for the scan path.

use delta_kernel::schema::StructType;
use polars::io::cloud::CloudOptions;
use polars::lazy::frame::LazyFrame;
use polars::prelude::{Expr, IdxSize};
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;

/// Per-row file-identity column injected via `include_file_paths`. The
/// split-and-rewrite pass reads it back to look up each row's
/// `LogicalRewrite`.
pub(crate) const FILE_ID_COL: &str = "__pldl_file__";

/// Per-row physical position across the whole scan, from polars' scan-level
/// row index. A deletion vector addresses a file's physical rows, and
/// polars numbers rows before the pushed predicate runs and keeps counting
/// through row groups its statistics skip. So the predicate can go into the
/// parquet reader (row-group skipping, pre-filtered decode) and the DV
/// keep-mask matches on this index afterwards instead of on batch position.
/// The index is scan-wide: each file starts at the physical row count of
/// every file before it.
pub(crate) const ROW_INDEX_COL: &str = "__pldl_row__";

/// `scan_parquet` plan over `paths`. `include_file_id` injects FILE_ID_COL
/// so the read path can slice rows back to source files for DV / select;
/// `include_row_index` injects ROW_INDEX_COL for the DV keep-mask.
/// `physical_limit` caps the rows read after the predicate: polars pushes
/// it into the reader when there is no predicate and stops the scan early
/// otherwise.
pub(crate) fn build_lazy_scan(
    paths: Vec<PlRefPath>,
    cloud_opts: Option<&CloudOptions>,
    select_exprs: &[Expr],
    predicate: Option<&Expr>,
    physical_schema: &StructType,
    include_file_id: bool,
    include_row_index: bool,
    physical_limit: Option<IdxSize>,
) -> anyhow::Result<LazyFrame> {
    let file_id = include_file_id.then(|| PlSmallStr::from_static(FILE_ID_COL));
    let row_index = include_row_index.then(|| PlSmallStr::from_static(ROW_INDEX_COL));
    let unified_scan_args = crate::engine::unified_scan_args(cloud_opts, file_id.clone());
    let lazy = crate::engine::dsl_parquet_scan(
        paths,
        physical_schema,
        unified_scan_args,
        row_index.as_ref(),
    )
    .map_err(|e| anyhow::anyhow!("scan_parquet plan failed: {e:#}"))?;

    let mut final_select: Vec<Expr> = Vec::with_capacity(select_exprs.len() + 2);
    final_select.extend(select_exprs.iter().cloned());
    final_select.extend(
        file_id
            .into_iter()
            .chain(row_index)
            .map(polars::prelude::col),
    );

    // Filter sits directly above the scan node so polars-io's parquet
    // predicate pushdown can pick it up.
    let mut plan = lazy;
    if let Some(pred) = predicate {
        plan = plan.filter(pred.clone());
    }
    if let Some(len) = physical_limit {
        plan = plan.slice(0, len);
    }
    Ok(plan.select(final_select))
}
