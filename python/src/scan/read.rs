//! Bulk-read plan construction for the scan path.

use delta_kernel::schema::StructType;
use polars::io::cloud::CloudOptions;
use polars::lazy::frame::LazyFrame;
use polars::prelude::{Expr, RowIndex};
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;

/// Per-row file-identity column injected via `include_file_paths`. The
/// split-and-rewrite pass reads it back to look up each row's
/// `LogicalRewrite`.
pub(crate) const FILE_ID_COL: &str = "__pldl_file__";

/// Per-row physical position across the whole scan, from polars' scan-level
/// row index. polars numbers rows before the pushed predicate runs and keeps
/// counting through row groups its statistics skip, so a DV keep-mask can
/// address rows by index after the reader has filtered.
pub(crate) const ROW_INDEX_COL: &str = "__pldl_row__";

/// `scan_parquet` plan over `paths`. `include_file_id` injects FILE_ID_COL
/// so the read path can slice rows back to source files for DV / select;
/// `include_row_index` injects ROW_INDEX_COL for the DV keep-mask.
pub(crate) fn build_lazy_scan(
    paths: Vec<PlRefPath>,
    cloud_opts: Option<&CloudOptions>,
    select_exprs: &[Expr],
    predicate: Option<&Expr>,
    physical_schema: &StructType,
    include_file_id: bool,
    include_row_index: bool,
) -> anyhow::Result<LazyFrame> {
    let mut unified_scan_args = crate::engine::unified_scan_args(
        cloud_opts,
        include_file_id.then(|| PlSmallStr::from_static(FILE_ID_COL)),
    );
    if include_row_index {
        unified_scan_args.row_index = Some(RowIndex {
            name: PlSmallStr::from_static(ROW_INDEX_COL),
            offset: 0,
        });
    }
    let lazy = crate::engine::dsl_parquet_scan(paths, physical_schema, unified_scan_args, None)
        .map_err(|e| anyhow::anyhow!("scan_parquet plan failed: {e:#}"))?;

    let mut final_select: Vec<Expr> = Vec::with_capacity(select_exprs.len() + 2);
    final_select.extend(select_exprs.iter().cloned());
    if include_file_id {
        final_select.push(polars::prelude::col(PlSmallStr::from_static(FILE_ID_COL)));
    }
    if include_row_index {
        final_select.push(polars::prelude::col(PlSmallStr::from_static(ROW_INDEX_COL)));
    }

    // Filter sits directly above the scan node so polars-io's parquet
    // predicate pushdown can pick it up.
    let mut plan = lazy;
    if let Some(pred) = predicate {
        plan = plan.filter(pred.clone());
    }
    Ok(plan.select(final_select))
}
