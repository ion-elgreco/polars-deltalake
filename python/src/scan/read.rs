//! polars-io `scan_parquet` plan construction shared with the kernel
//! `ParquetHandler`.

use delta_kernel::schema::StructType;
use polars::io::cloud::CloudOptions;
use polars::lazy::frame::LazyFrame;
use polars::prelude::Expr;
use polars_plan::dsl::{DslBuilder, ScanSources};
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;

/// Per-row file-identity column injected via `include_file_paths`. The
/// split-and-rewrite pass reads it back to look up each row's
/// `LogicalRewrite`.
pub(crate) const FILE_ID_COL: &str = "__pldl_file__";

/// Single `scan_parquet` plan over all `paths`, with the file-id column
/// appended to `select_exprs` so it survives projection.
pub(crate) fn build_lazy_scan(
    paths: Vec<PlRefPath>,
    cloud_opts: Option<&CloudOptions>,
    select_exprs: &[Expr],
    predicate: Option<&Expr>,
    physical_schema: &StructType,
) -> anyhow::Result<LazyFrame> {
    let parquet_options = crate::engine::parquet_options(physical_schema)
        .map_err(|e| anyhow::anyhow!("kernel→polars schema conversion failed: {e:#}"))?;
    let unified_scan_args =
        crate::engine::unified_scan_args(cloud_opts, Some(PlSmallStr::from_static(FILE_ID_COL)));

    let sources = ScanSources::Paths(paths.into());
    let lazy: LazyFrame = DslBuilder::scan_parquet(sources, parquet_options, unified_scan_args)
        .map_err(|e| anyhow::anyhow!("scan_parquet plan failed: {e:#}"))?
        .build()
        .into();

    let mut select_with_file_id: Vec<Expr> = Vec::with_capacity(select_exprs.len() + 1);
    select_with_file_id.extend(select_exprs.iter().cloned());
    select_with_file_id.push(polars::prelude::col(PlSmallStr::from_static(FILE_ID_COL)));

    // Filter sits directly above the scan node so polars-io's parquet
    // predicate pushdown can pick it up.
    let mut plan = lazy;
    if let Some(pred) = predicate {
        plan = plan.filter(pred.clone());
    }
    Ok(plan.select(select_with_file_id))
}

/// `col(...)` per kernel physical-schema field. Caller appends file-id /
/// other metadata columns before handing to `build_lazy_scan`.
pub(crate) fn select_exprs_for_schema(physical_schema: &StructType) -> Vec<Expr> {
    physical_schema
        .fields()
        .map(|f| polars::prelude::col(PlSmallStr::from_str(f.name.as_str())))
        .collect()
}
