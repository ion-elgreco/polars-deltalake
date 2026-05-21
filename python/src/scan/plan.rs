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
    /// DV keep-mask (true = keep), or `None` if the file has no DV.
    pub(crate) selection_vector: Option<Vec<bool>>,
}

/// `Scan::scan_metadata` drained into bulk-read inputs.
pub(crate) struct ResolvedScan {
    /// Files in scan order.
    pub(crate) paths: Vec<PlRefPath>,
    /// Parallel to `paths`.
    pub(crate) rewrites: Vec<LogicalRewrite>,
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
        let sv = if scan_file.dv_info.has_vector() {
            scan_file
                .dv_info
                .get_selection_vector(ctx.engine, ctx.table_root)?
        } else {
            None
        };

        let idx = ctx.paths.len();
        ctx.path_index.insert(pl_path.as_str().to_string(), idx);
        ctx.paths.push(pl_path);
        ctx.rewrites.push(LogicalRewrite {
            transform: scan_file.transform,
            selection_vector: sv,
        });
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
        path_index: ctx.path_index,
    })
}
