//! `Scan::scan_metadata` → resolved file list + per-file rewrites.

use std::collections::HashMap;

use delta_kernel::Engine;
use delta_kernel::expressions::ExpressionRef;
use delta_kernel::scan::Scan;
use delta_kernel::scan::state::ScanFile;
use polars_utils::pl_path::PlRefPath;
use url::Url;

use crate::engine::path_for_polars_io;

/// Per-file work to apply post-read: kernel `Transform` + DV keep-mask.
pub(crate) struct LogicalRewrite {
    /// Physical → logical (column-mapping + partition values).
    pub(crate) transform: Option<ExpressionRef>,
    /// Per-file DV state — sorted deleted row indices + cursor of how many
    /// rows of the file have been consumed by prior batches. `None` if the
    /// file has no DV.
    pub(crate) dv: Option<DvState>,
}

pub(crate) struct DvState {
    /// Sorted ascending row indices to drop. Consumed entries are drained
    /// off the front as batches are processed.
    pub(crate) deleted: Vec<u64>,
    /// Absolute row offset within the file already covered by past batches.
    pub(crate) cursor: u64,
}

/// `Scan::scan_metadata` drained into bulk-read inputs.
pub(crate) struct ResolvedScan {
    /// Files in scan order.
    pub(crate) paths: Vec<PlRefPath>,
    /// Parallel to `paths`.
    pub(crate) rewrites: Vec<LogicalRewrite>,
    /// Parallel to `paths`. Used for predicate-driven file skipping via
    /// polars (untranslatable partition conjuncts) before the bulk read.
    pub(crate) partition_values: Vec<HashMap<String, String>>,
    /// `FILE_ID_COL` value (= `PlRefPath::as_str()`) → index in `paths` /
    /// `rewrites`.
    pub(crate) path_index: HashMap<String, usize>,
}

/// Drain `scan.scan_metadata` into a `ResolvedScan`. Materializes each DV
/// eagerly so the read path doesn't need the engine handle.
///
/// Kernel's `ScanCallback` is `fn(&mut T, ScanFile)` (no `Result`), so
/// callback errors are stashed on the context and surfaced after each
/// `visit_scan_files` call.
pub(crate) fn resolve_scan(scan: &Scan, engine: &dyn Engine) -> anyhow::Result<ResolvedScan> {
    struct Ctx<'a> {
        engine: &'a dyn Engine,
        table_root: &'a Url,
        paths: Vec<PlRefPath>,
        rewrites: Vec<LogicalRewrite>,
        partition_values: Vec<HashMap<String, String>>,
        path_index: HashMap<String, usize>,
        err: Option<delta_kernel::Error>,
    }

    fn visit_one(ctx: &mut Ctx<'_>, scan_file: ScanFile) -> Result<(), delta_kernel::Error> {
        let abs = ctx.table_root.join(&scan_file.path).map_err(|e| {
            delta_kernel::Error::Generic(format!(
                "failed to resolve scan file path {}: {e}",
                scan_file.path
            ))
        })?;
        let pl_path = path_for_polars_io(&abs)?;
        // `Vec<u64>` of deleted row indices is far smaller than a `Vec<bool>`
        // keep-mask for sparse deletes — only ~8 bytes per delete vs. one byte
        // per row in the file.
        let dv = if scan_file.dv_info.has_vector() {
            scan_file
                .dv_info
                .get_row_indexes(ctx.engine, ctx.table_root)?
                .map(|deleted| DvState { deleted, cursor: 0 })
        } else {
            None
        };

        let idx = ctx.paths.len();
        ctx.path_index.insert(pl_path.as_str().to_string(), idx);
        ctx.paths.push(pl_path);
        ctx.rewrites.push(LogicalRewrite {
            transform: scan_file.transform,
            dv,
        });
        ctx.partition_values.push(scan_file.partition_values);
        Ok(())
    }

    // `ScanCallback` is a bare `fn` pointer with no `Result` return, so
    // bubble visitor errors through the context.
    fn callback(ctx: &mut Ctx<'_>, scan_file: ScanFile) {
        if ctx.err.is_some() {
            return;
        }
        if let Err(e) = visit_one(ctx, scan_file) {
            ctx.err = Some(e);
        }
    }

    let table_root = scan.table_root().clone();
    let mut ctx = Ctx {
        engine,
        table_root: &table_root,
        paths: Vec::new(),
        rewrites: Vec::new(),
        partition_values: Vec::new(),
        path_index: HashMap::new(),
        err: None,
    };

    for res in scan
        .scan_metadata(engine)
        .map_err(|e| anyhow::anyhow!("scan_metadata failed: {e:#}"))?
    {
        let metadata = res.map_err(|e| anyhow::anyhow!("scan_metadata item failed: {e:#}"))?;
        ctx = metadata
            .visit_scan_files(ctx, callback)
            .map_err(|e| anyhow::anyhow!("visit_scan_files failed: {e:#}"))?;
        if let Some(e) = ctx.err.take() {
            return Err(anyhow::anyhow!("scan_metadata callback failed: {e:#}"));
        }
    }

    Ok(ResolvedScan {
        paths: ctx.paths,
        rewrites: ctx.rewrites,
        partition_values: ctx.partition_values,
        path_index: ctx.path_index,
    })
}
