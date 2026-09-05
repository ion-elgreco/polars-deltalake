//! `LogicalScanIter` — splits bulk-read frames on file-id boundaries and
//! applies the per-file DV keep-mask + physical→logical select list.

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use delta_kernel::StorageHandler;
use polars::prelude::{
    BooleanChunked, Column, DataFrame, Expr, IdxCa, IdxSize, IntoLazy, LiteralValue, PolarsResult,
    Scalar, StringChunked, col,
};
use polars::series::IsSorted;
use polars_arrow::bitmap::MutableBitmap;
use polars_utils::pl_str::PlSmallStr;
use url::Url;

use crate::engine::select_anchored;
use crate::scan::plan::{LazyDv, LogicalRewrite, ScanFileMeta};
use crate::scan::read::{FILE_ID_COL, ROW_INDEX_COL};

/// One column of a pre-parsed simple select: a rename of a read column or a
/// broadcast literal (partition value).
enum SimpleOp {
    Rename { from: PlSmallStr, to: PlSmallStr },
    Broadcast { name: PlSmallStr, value: Scalar },
}

/// Real-world `LogicalRewrite` selects are renames + partition literals, so
/// they apply as direct column ops — no per-batch plan build / streaming
/// collect. Anything else (nested column-mapping rebuilds) falls back to
/// the lazy path.
pub(crate) struct SimpleSelect {
    ops: Vec<SimpleOp>,
}

impl SimpleSelect {
    /// `None` when any expr is not a plain rename or a column-free literal.
    pub(crate) fn parse(exprs: &[Expr]) -> Option<Self> {
        let mut ops = Vec::with_capacity(exprs.len());
        for expr in exprs {
            match expr {
                Expr::Column(name) => ops.push(SimpleOp::Rename {
                    from: name.clone(),
                    to: name.clone(),
                }),
                Expr::Alias(inner, to) => match inner.as_ref() {
                    Expr::Column(from) => ops.push(SimpleOp::Rename {
                        from: from.clone(),
                        to: to.clone(),
                    }),
                    // Partition literals already carry their output dtype, so
                    // read the scalar straight off instead of planning a query
                    // per file.
                    Expr::Literal(LiteralValue::Scalar(value)) => ops.push(SimpleOp::Broadcast {
                        name: to.clone(),
                        value: value.clone(),
                    }),
                    // Column-free but not already a scalar (a cast, a Series
                    // literal): evaluate once. A multi-value literal has no
                    // broadcast form; let the lazy path evaluate it rather
                    // than silently keeping element 0.
                    _ if polars_plan::utils::expr_to_leaf_column_names(inner).is_empty() => {
                        let evaluated = DataFrame::empty()
                            .lazy()
                            .select([expr.clone()])
                            .collect()
                            .ok()?;
                        let col = evaluated.columns().first()?;
                        if col.len() != 1 {
                            return None;
                        }
                        let av = col.get(0).ok()?.into_static();
                        ops.push(SimpleOp::Broadcast {
                            name: to.clone(),
                            value: Scalar::new(col.dtype().clone(), av),
                        });
                    }
                    _ => return None,
                },
                _ => return None,
            }
        }
        Some(Self { ops })
    }

    pub(crate) fn apply(&self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let height = df.height();
        let cols: Vec<Column> = self
            .ops
            .iter()
            .map(|op| match op {
                SimpleOp::Rename { from, to } => df.column(from).map(|c| {
                    let mut c = c.clone();
                    c.rename(to.clone());
                    c
                }),
                SimpleOp::Broadcast { name, value } => {
                    Ok(Column::new_scalar(name.clone(), value.clone(), height))
                }
            })
            .collect::<PolarsResult<_>>()?;
        DataFrame::new(height, cols)
    }
}

/// A file's deletion vector placed in the scan-wide row index.
struct PlacedDv {
    dv: LazyDv,
    /// `ROW_INDEX_COL` values the file's physical rows occupy.
    rows: Range<u64>,
}

/// One file's select list, its pre-parsed fast path, and its placed DV.
struct FileRewrite {
    select: Option<Vec<Expr>>,
    simple: Option<SimpleSelect>,
    dv: Option<PlacedDv>,
}

/// Splits each bulk-read frame on `FILE_ID_COL` runs and applies the
/// per-file `LogicalRewrite`, yielding logical-schema frames in scan order.
pub(crate) struct LogicalScanIter {
    /// Raw bulk-read frames carrying `FILE_ID_COL`. One frame on the eager
    /// path; many on a streaming follow-up.
    source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
    /// `FILE_ID_COL` value → index in `files`.
    path_index: HashMap<String, usize>,
    files: Vec<FileRewrite>,
    /// The read also carries `ROW_INDEX_COL`: on whenever a file has a DV.
    row_index: bool,
    /// For reading a file's DV the first time one of its batches arrives.
    storage: Arc<dyn StorageHandler>,
    table_root: Url,
    /// Mixed atomic conjuncts (touching both partition and data cols) —
    /// applied after the select list materializes partition values.
    orphan_predicate: Option<Expr>,
    /// Set when the scan was widened past the caller's projection to give
    /// `orphan_predicate` its columns. Drops the extras again once the
    /// filter has run.
    output_projection: Option<Vec<Expr>>,
    /// One inner frame may span multiple files; we slice into per-file
    /// frames here and drain before pulling the next inner frame.
    pending: VecDeque<Result<DataFrame, delta_kernel::Error>>,
}

impl LogicalScanIter {
    /// `files` in scan order, the order polars numbers `ROW_INDEX_COL` in;
    /// `spans` are their `place_dvs` placements.
    pub(crate) fn new(
        source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
        path_index: HashMap<String, usize>,
        files: Vec<ScanFileMeta>,
        spans: Vec<Option<Range<u64>>>,
        storage: Arc<dyn StorageHandler>,
        table_root: Url,
        orphan_predicate: Option<Expr>,
        output_projection: Option<Vec<String>>,
    ) -> Self {
        let row_index = spans.iter().any(Option::is_some);
        let files = files
            .into_iter()
            .zip(spans)
            .map(|(file, rows)| {
                let LogicalRewrite { select, dv } = file.rewrite;
                FileRewrite {
                    simple: select.as_deref().and_then(SimpleSelect::parse),
                    select,
                    dv: dv.zip(rows).map(|(dv, rows)| PlacedDv { dv, rows }),
                }
            })
            .collect();
        Self {
            source,
            path_index,
            files,
            row_index,
            storage,
            table_root,
            orphan_predicate,
            output_projection: output_projection
                .map(|cols| cols.iter().map(|c| col(c.as_str())).collect()),
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
            let Some(file_id) = cur else {
                // polars-io always populates the column; skipping the run
                // would drop those rows from the scan without a trace.
                return Err(delta_kernel::Error::Generic(format!(
                    "polars-io scan produced {} row(s) with a null {FILE_ID_COL}",
                    end - start
                )));
            };
            let file_id = file_id.to_owned();
            let sub = df.slice(start as i64, end - start);
            let out = self.apply_rewrite(&file_id, sub);
            self.pending.push_back(out);
            start = end;
        }
        Ok(())
    }

    /// Drop the file-id and row-index columns, apply the DV keep-mask and
    /// the physical→logical select, return the logical frame.
    fn apply_rewrite(
        &self,
        file_id: &str,
        mut df: DataFrame,
    ) -> Result<DataFrame, delta_kernel::Error> {
        df.drop_in_place(FILE_ID_COL)
            .map_err(|e| delta_kernel::Error::Generic(format!("drop {FILE_ID_COL}: {e}")))?;
        let row_index = self
            .row_index
            .then(|| df.drop_in_place(ROW_INDEX_COL))
            .transpose()
            .map_err(|e| delta_kernel::Error::Generic(format!("drop {ROW_INDEX_COL}: {e}")))?;

        let idx = *self.path_index.get(file_id).ok_or_else(|| {
            delta_kernel::Error::Generic(format!("unknown file_id from polars-io scan: {file_id}"))
        })?;
        let entry = &self.files[idx];

        if let Some(dv) = &entry.dv {
            let rows = row_index
                .as_ref()
                .expect("row index requested whenever a file has a DV")
                .idx()
                .map_err(|e| {
                    delta_kernel::Error::Generic(format!(
                        "{ROW_INDEX_COL} is not an index column: {e}"
                    ))
                })?;
            let deleted = dv
                .dv
                .rows(&self.storage, &self.table_root)
                .map_err(|e| delta_kernel::Error::Generic(format!("{e:#}")))?;
            if let Some(mask) = keep_mask(deleted, &dv.rows, rows)? {
                df = df
                    .filter(&mask)
                    .map_err(|e| delta_kernel::Error::Generic(format!("DV filter: {e}")))?;
            }
        }

        // With no lazy stage the fast path stays collect-free; once one is
        // needed, every active stage chains into that one plan: the select
        // first, the orphan filter after it (logical/partition names), and
        // the projection last so the filter still sees the columns the read
        // was widened for.
        let has_lazy_stage = self.orphan_predicate.is_some()
            || self.output_projection.is_some()
            || (entry.simple.is_none() && entry.select.is_some());
        if !has_lazy_stage {
            if let Some(fast) = &entry.simple {
                df = fast.apply(&df).map_err(|e| {
                    delta_kernel::Error::Generic(format!("logical rewrite eval: {e}"))
                })?;
            }
            return Ok(df);
        }

        let mut lazy = df.lazy();
        if let Some(select) = &entry.select {
            lazy = select_anchored(lazy, select);
        }
        if let Some(pred) = &self.orphan_predicate {
            lazy = lazy.filter(pred.clone());
        }
        if let Some(cols) = &self.output_projection {
            lazy = lazy.select(cols.clone());
        }
        crate::engine::collect_streaming_single(lazy)
            .map_err(|e| delta_kernel::Error::Generic(format!("logical rewrite eval: {e}")))
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

/// Keep-mask for one file's batch, `None` when no row is deleted. An
/// unfiltered batch is one contiguous physical slice (polars flags its row
/// index ascending), so the deleted positions inside it go straight into
/// the bitmap. A filtered batch merges its ascending index against the
/// deleted list; a descending step re-seeks with a binary search rather
/// than trusting the order. Every index must fall inside the file, or the
/// placement is wrong and the mask would be too.
fn keep_mask(
    deleted: &[u64],
    span: &Range<u64>,
    rows: &IdxCa,
) -> Result<Option<BooleanChunked>, delta_kernel::Error> {
    const NAME: &str = "__pldl_dv__";
    if rows.null_count() > 0 {
        return Err(delta_kernel::Error::Generic(format!(
            "{ROW_INDEX_COL} has nulls"
        )));
    }
    let file_local = |global: IdxSize| {
        let global = global as u64;
        if span.contains(&global) {
            Ok(global - span.start)
        } else {
            Err(delta_kernel::Error::Generic(format!(
                "row index {global} falls outside its file's rows {span:?}"
            )))
        }
    };
    let n = rows.len();
    let mut keep = MutableBitmap::from_len_set(n);

    if rows.is_sorted_flag() == IsSorted::Ascending
        && let (Some(first), Some(last)) = (rows.first(), rows.last())
        && (last as u64).checked_sub(first as u64) == Some(n as u64 - 1)
    {
        let lo = file_local(first)?;
        file_local(last)?;
        let start = deleted.partition_point(|&d| d < lo);
        let end = deleted.partition_point(|&d| d < lo + n as u64);
        if start == end {
            return Ok(None);
        }
        for &d in &deleted[start..end] {
            keep.set((d - lo) as usize, false);
        }
        return Ok(Some(BooleanChunked::from_bitmap(NAME.into(), keep.into())));
    }

    let mut dropped = 0usize;
    let mut i = 0usize;
    let mut p = 0usize;
    let mut prev = u64::MAX;
    for arr in rows.downcast_iter() {
        for &idx in arr.values().iter() {
            let local = file_local(idx)?;
            if local < prev {
                p = deleted.partition_point(|&d| d < local);
            } else {
                while p < deleted.len() && deleted[p] < local {
                    p += 1;
                }
            }
            if p < deleted.len() && deleted[p] == local {
                keep.set(i, false);
                dropped += 1;
            }
            prev = local;
            i += 1;
        }
    }
    Ok((dropped > 0).then(|| BooleanChunked::from_bitmap(NAME.into(), keep.into())))
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

#[cfg(test)]
mod simple_select_tests {
    use polars::prelude::{IntoLazy, col, df, lit};

    use super::*;

    fn exprs() -> Vec<Expr> {
        vec![col("phys").alias("logical"), lit(7i64).alias("part")]
    }

    #[test]
    fn parses_renames_and_literals() {
        assert!(SimpleSelect::parse(&exprs()).is_some());
        let rebuild = col("a").struct_().field_by_name("x").alias("y");
        assert!(SimpleSelect::parse(&[rebuild]).is_none());
    }

    /// The fast path must be indistinguishable from the lazy select.
    #[test]
    fn fast_apply_matches_lazy_select() {
        let frame = df!("phys" => [1i64, 2], "extra" => ["a", "b"]).unwrap();
        let fast = SimpleSelect::parse(&exprs())
            .unwrap()
            .apply(&frame)
            .unwrap();
        let lazy = frame.lazy().select(exprs()).collect().unwrap();
        assert!(fast.equals_missing(&lazy), "{fast:?} vs {lazy:?}");
    }

    /// A Series literal has no single broadcast value, so the fast path must
    /// decline instead of replicating element 0 over every row.
    #[test]
    fn multi_value_literal_declines_fast_path() {
        use polars::prelude::{NamedFrom, Series};

        let multi = lit(Series::new("s".into(), [1i64, 2])).alias("part");
        assert!(SimpleSelect::parse(&[multi]).is_none());
    }

    /// An all-partition projection selects nothing but literals; polars sizes
    /// that select from the expressions and would return one row per file.
    #[test]
    fn column_free_select_keeps_the_input_height() {
        let frame = df!("phys" => [1i64, 2, 3])
            .unwrap()
            .select(["phys"])
            .unwrap();
        let frame = frame.drop("phys").unwrap();
        assert_eq!(frame.height(), 3);
        assert_eq!(frame.width(), 0);

        let select = vec![lit(7i64).alias("part")];
        let out = select_anchored(frame.lazy(), &select).collect().unwrap();
        assert_eq!(out.height(), 3);
        assert_eq!(out.get_column_names(), ["part"]);
    }
}

#[cfg(test)]
mod keep_mask_tests {
    use polars::prelude::NamedFrom;

    use super::*;

    struct Dv {
        deleted: Vec<u64>,
        rows: Range<u64>,
    }

    fn dv(mut deleted: Vec<u64>, rows: Range<u64>) -> Dv {
        deleted.sort_unstable();
        Dv { deleted, rows }
    }

    fn keep_mask(dv: &Dv, rows: &IdxCa) -> Result<Option<BooleanChunked>, delta_kernel::Error> {
        super::keep_mask(&dv.deleted, &dv.rows, rows)
    }

    /// `None` means every row survives.
    fn bools(mask: Option<BooleanChunked>, n: usize) -> Vec<bool> {
        match mask {
            Some(m) => (0..n).map(|i| m.get(i).unwrap()).collect(),
            None => vec![true; n],
        }
    }

    /// Unflagged, so the merge path.
    fn mask(dv: &Dv, rows: &[IdxSize]) -> Result<Vec<bool>, delta_kernel::Error> {
        keep_mask(dv, &IdxCa::new("r".into(), rows)).map(|m| bools(m, rows.len()))
    }

    /// Flagged ascending, as polars hands over a scan row index.
    fn flagged(rows: &[IdxSize]) -> IdxCa {
        let mut ca = IdxCa::new("r".into(), rows);
        ca.set_sorted_flag(IsSorted::Ascending);
        ca
    }

    /// kernel's `row_indexes` inherits the DV bitmap's iteration order; the
    /// merge requires ascending. Unsorted input must still mask exactly the
    /// deleted rows.
    #[test]
    fn keep_mask_is_exact_for_unsorted_dv_indices() {
        let dv = dv(vec![5, 100, 3], 0..200);
        let rows: Vec<IdxSize> = (0..110).collect();
        let expected: Vec<bool> = (0..110u64).map(|i| i != 3 && i != 5 && i != 100).collect();
        assert_eq!(mask(&dv, &rows).unwrap(), expected);
    }

    /// The DV indexes the file; the scan indexes every file before it too.
    #[test]
    fn file_span_maps_scan_index_to_file_index() {
        let dv = dv(vec![2], 1000..1010);
        assert_eq!(
            mask(&dv, &[1000, 1001, 1002, 1003]).unwrap(),
            [true, true, false, true]
        );
    }

    /// Row groups the reader skipped never reach the mask; the rows that do
    /// must still land on their own physical positions.
    #[test]
    fn skipped_rows_do_not_shift_the_mask() {
        let dv = dv(vec![1, 7, 8], 0..10);
        assert_eq!(
            mask(&dv, &[6, 7, 8, 9]).unwrap(),
            [true, false, false, true]
        );
    }

    #[test]
    fn non_ascending_rows_still_mask_exactly() {
        let dv = dv(vec![1, 3], 0..10);
        assert_eq!(
            mask(&dv, &[3, 1, 2, 3, 0]).unwrap(),
            [false, false, true, false, true]
        );
    }

    /// An index outside the file means the placement is wrong; masking on
    /// would silently keep deleted rows or drop live ones.
    #[test]
    fn index_outside_the_file_errors() {
        let dv = dv(vec![0], 10..15);
        assert!(mask(&dv, &[9]).is_err());
        assert!(mask(&dv, &[15]).is_err());
        assert_eq!(mask(&dv, &[10, 14]).unwrap(), [false, true]);
    }

    /// An unfiltered batch is one contiguous slice and takes the positional
    /// path; it must still respect the file's bounds, and a slice with no
    /// deleted row in it needs no mask at all.
    #[test]
    fn contiguous_flagged_batch_masks_positionally() {
        let dv = dv(vec![0, 3, 4, 9], 100..110);
        let rows: Vec<IdxSize> = (100..110).collect();
        assert_eq!(
            bools(keep_mask(&dv, &flagged(&rows)).unwrap(), 10),
            [
                false, true, true, false, false, true, true, true, true, false
            ]
        );
        let rows: Vec<IdxSize> = (105..115).collect();
        assert!(keep_mask(&dv, &flagged(&rows)).is_err());
        let rows: Vec<IdxSize> = (105..109).collect();
        assert!(keep_mask(&dv, &flagged(&rows)).unwrap().is_none());
    }

    /// A filtered batch keeps the flag but has gaps, so it merges instead.
    #[test]
    fn flagged_batch_with_gaps_merges() {
        let dv = dv(vec![1, 7, 8], 0..10);
        assert_eq!(
            bools(keep_mask(&dv, &flagged(&[0, 1, 7, 9])).unwrap(), 4),
            [true, false, false, true]
        );
        assert!(keep_mask(&dv, &flagged(&[0, 2, 9])).unwrap().is_none());
    }
}
