//! `LogicalScanIter` — splits bulk-read frames on file-id boundaries and
//! applies the per-file DV keep-mask + physical→logical select list.

use std::collections::{HashMap, VecDeque};

use polars::prelude::{
    BooleanChunked, Column, DataFrame, Expr, IntoLazy, LiteralValue, NamedFrom, PolarsResult,
    Scalar, StringChunked, col,
};
use polars_utils::pl_str::PlSmallStr;

use crate::engine::select_anchored;
use crate::scan::plan::{DvState, LogicalRewrite};
use crate::scan::read::FILE_ID_COL;

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

/// One file's rewrite state plus its pre-parsed select fast path.
struct FileRewrite {
    rewrite: LogicalRewrite,
    simple: Option<SimpleSelect>,
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
    /// Physical-name data conjuncts held back from the parquet scan because
    /// a DV is in play: the keep-mask addresses file row positions, so the
    /// predicate has to run after it.
    physical_predicate: Option<Expr>,
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
    pub(crate) fn new(
        source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
        path_index: HashMap<String, usize>,
        rewrites: Vec<LogicalRewrite>,
        physical_predicate: Option<Expr>,
        orphan_predicate: Option<Expr>,
        output_projection: Option<Vec<String>>,
    ) -> Self {
        let files = rewrites
            .into_iter()
            .map(|rewrite| {
                let simple = rewrite.select.as_deref().and_then(SimpleSelect::parse);
                FileRewrite { rewrite, simple }
            })
            .collect();
        Self {
            source,
            path_index,
            files,
            physical_predicate,
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
        let entry = &mut self.files[idx];

        if let Some(state) = entry.rewrite.dv.as_mut() {
            let mask = build_keep_mask(state, df.height());
            let mask = BooleanChunked::new("__pldl_dv__".into(), mask.as_slice());
            df = df
                .filter(&mask)
                .map_err(|e| delta_kernel::Error::Generic(format!("DV filter: {e}")))?;
        }

        // With no predicate in play the fast path stays collect-free; once a
        // predicate forces a streaming collect anyway, every active stage
        // chains into that one plan: the physical filter must precede the
        // select (physical names), the orphan filter must follow it
        // (logical/partition names), and the projection comes last so the
        // filter still sees the columns the read was widened for.
        let has_lazy_stage = self.physical_predicate.is_some()
            || self.orphan_predicate.is_some()
            || self.output_projection.is_some()
            || (entry.simple.is_none() && entry.rewrite.select.is_some());
        if !has_lazy_stage {
            if let Some(fast) = &entry.simple {
                df = fast.apply(&df).map_err(|e| {
                    delta_kernel::Error::Generic(format!("logical rewrite eval: {e}"))
                })?;
            }
            return Ok(df);
        }

        let mut lazy = df.lazy();
        if let Some(pred) = &self.physical_predicate {
            lazy = lazy.filter(pred.clone());
        }
        if let Some(select) = &entry.rewrite.select {
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
    use super::*;

    /// kernel's `row_indexes` inherits the DV bitmap's iteration order; the
    /// mask math (`partition_point` + cursor subtraction) requires ascending.
    /// Unsorted input must still mask exactly the deleted rows.
    #[test]
    fn keep_mask_is_exact_for_unsorted_dv_indices() {
        let mut state = DvState::new(vec![5, 100, 3]);
        let first = build_keep_mask(&mut state, 10);
        let expected: Vec<bool> = (0..10u64).map(|i| i != 3 && i != 5).collect();
        assert_eq!(first, expected, "batch 1 must drop rows 3 and 5");
        let second = build_keep_mask(&mut state, 100);
        let expected: Vec<bool> = (10..110u64).map(|i| i != 100).collect();
        assert_eq!(second, expected, "batch 2 must drop file row 100");
    }
}
