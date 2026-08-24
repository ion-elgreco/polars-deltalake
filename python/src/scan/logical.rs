//! `LogicalScanIter` — splits bulk-read frames on file-id boundaries and
//! applies the per-file DV keep-mask + physical→logical select list.

use std::collections::{HashMap, VecDeque};

use polars::prelude::{BooleanChunked, DataFrame, Expr, IntoLazy, NamedFrom, StringChunked};
use polars_plan::dsl::Engine as PolarsEngineMode;

use crate::scan::plan::{DvState, LogicalRewrite};
use crate::scan::read::FILE_ID_COL;

/// Splits each bulk-read frame on `FILE_ID_COL` runs and applies the
/// per-file `LogicalRewrite`, yielding logical-schema frames in scan order.
pub(crate) struct LogicalScanIter {
    /// Raw bulk-read frames carrying `FILE_ID_COL`. One frame on the eager
    /// path; many on a streaming follow-up.
    source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
    /// `FILE_ID_COL` value → index in `rewrites`.
    path_index: HashMap<String, usize>,
    rewrites: Vec<LogicalRewrite>,
    /// Mixed atomic conjuncts (touching both partition and data cols) —
    /// applied after the select list materializes partition values.
    orphan_predicate: Option<Expr>,
    /// One inner frame may span multiple files; we slice into per-file
    /// frames here and drain before pulling the next inner frame.
    pending: VecDeque<Result<DataFrame, delta_kernel::Error>>,
}

impl LogicalScanIter {
    pub(crate) fn new(
        source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
        path_index: HashMap<String, usize>,
        rewrites: Vec<LogicalRewrite>,
        orphan_predicate: Option<Expr>,
    ) -> Self {
        Self {
            source,
            path_index,
            rewrites,
            orphan_predicate,
            pending: VecDeque::new(),
        }
    }

    /// Slice on file-id boundaries, push each per-file logical frame onto
    /// `pending`. With `maintain_order=true` the file-id column is composed
    /// of contiguous runs, so we binary-search the end of each run rather
    /// than scanning row-by-row
    fn split_and_buffer(&mut self, df: DataFrame) -> Result<(), delta_kernel::Error> {
        let n = df.height();
        if n == 0 {
            return Ok(());
        }
        let file_col = df
            .column(FILE_ID_COL)
            .map_err(|e| delta_kernel::Error::Generic(format!("file-id column missing: {e}")))?;
        let file_str = file_col
            .str()
            .map_err(|e| delta_kernel::Error::Generic(format!("file-id column not Utf8: {e}")))?;

        if let Some(first) = file_str.get(0)
            && Some(first) == file_str.get(n - 1)
        {
            let file_id = first.to_owned();
            let out = self.apply_rewrite(&file_id, df);
            self.pending.push_back(out);
            return Ok(());
        }

        let mut start = 0usize;
        while start < n {
            let cur = file_str.get(start);
            let end = find_run_end(file_str, start, n, cur);
            if let Some(file_id) = cur {
                let file_id = file_id.to_owned();
                let sub = df.slice(start as i64, end - start);
                let out = self.apply_rewrite(&file_id, sub);
                self.pending.push_back(out);
            }
            start = end;
        }
        Ok(())
    }

    /// Drop the file-id column, apply the DV keep-mask and the
    /// physical→logical select, return the logical frame.
    fn apply_rewrite(
        &mut self,
        file_id: &str,
        mut df: DataFrame,
    ) -> Result<DataFrame, delta_kernel::Error> {
        df.drop_in_place(FILE_ID_COL)
            .map_err(|e| delta_kernel::Error::Generic(format!("drop {FILE_ID_COL}: {e}")))?;

        let idx = *self.path_index.get(file_id).ok_or_else(|| {
            delta_kernel::Error::Generic(format!("unknown file_id from polars-io scan: {file_id}"))
        })?;
        let rewrite = &mut self.rewrites[idx];

        if let Some(state) = rewrite.dv.as_mut() {
            let mask = build_keep_mask(state, df.height());
            let mask = BooleanChunked::new("__pldl_dv__".into(), mask.as_slice());
            df = df
                .filter(&mask)
                .map_err(|e| delta_kernel::Error::Generic(format!("DV filter: {e}")))?;
        }

        if rewrite.select.is_some() || self.orphan_predicate.is_some() {
            let mut lazy = df.lazy();
            if let Some(select) = &rewrite.select {
                lazy = lazy.select(select.clone());
            }
            if let Some(pred) = &self.orphan_predicate {
                lazy = lazy.filter(pred.clone());
            }
            df = lazy
                .collect_with_engine(PolarsEngineMode::Streaming)
                .map_err(|e| delta_kernel::Error::Generic(format!("logical rewrite eval: {e}")))?
                .unwrap_single();
        }
        Ok(df)
    }
}

/// First index in `start..end` where `file_str.get(i) != value`, or `end`
/// if the value extends all the way. Requires the column to be a single
/// run within `[start, returned_end)`.
fn find_run_end<'a>(
    file_str: &'a StringChunked,
    start: usize,
    end: usize,
    value: Option<&'a str>,
) -> usize {
    let mut lo = start + 1;
    let mut hi = end;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if file_str.get(mid) == value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Drains consumed entries off the front so per-file state shrinks as the
/// file is read.
fn build_keep_mask(state: &mut DvState, batch_num_rows: usize) -> Vec<bool> {
    let range_end = state.cursor + batch_num_rows as u64;
    let split = state.deleted.partition_point(|&i| i < range_end);
    let mut mask = vec![true; batch_num_rows];
    for i in state.deleted.drain(..split) {
        // Every i here is in [cursor, range_end) — earlier batches already
        // drained anything below cursor — so the subtraction can't underflow.
        mask[(i - state.cursor) as usize] = false;
    }
    state.cursor = range_end;
    mask
}

impl Iterator for LogicalScanIter {
    type Item = Result<DataFrame, delta_kernel::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.pending.pop_front() {
                return Some(item);
            }
            let raw = self.source.next()?;
            match raw {
                Err(e) => return Some(Err(delta_kernel::Error::Generic(format!("{e:#}")))),
                Ok(df) => {
                    if let Err(e) = self.split_and_buffer(df) {
                        return Some(Err(e));
                    }
                }
            }
        }
    }
}
