//! `LogicalScanIter` — splits bulk-read frames on file-id boundaries and
//! applies the per-file DV keep-mask + physical→logical select list.

use std::collections::{HashMap, VecDeque};

use polars::prelude::{
    BooleanChunked, Column, DataFrame, Expr, IntoLazy, LiteralValue, NamedFrom, PolarsResult,
    Scalar, StringChunked,
};
use polars_plan::dsl::Engine as PolarsEngineMode;
use polars_utils::pl_str::PlSmallStr;

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
        let mut literal_exprs: Vec<Expr> = Vec::new();
        // Positions in `ops` waiting on the one-shot eval below.
        let mut deferred: Vec<usize> = Vec::new();
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
                    _ if polars_plan::utils::expr_to_leaf_column_names(inner).is_empty() => {
                        deferred.push(ops.len());
                        literal_exprs.push(expr.clone());
                        ops.push(SimpleOp::Broadcast {
                            name: to.clone(),
                            // Placeholder; filled from the one-shot eval below.
                            value: Scalar::null(polars::prelude::DataType::Null),
                        });
                    }
                    _ => return None,
                },
                _ => return None,
            }
        }
        if !literal_exprs.is_empty() {
            // Column-free but not already a scalar (a cast, a Series
            // literal): one evaluation for all of them, then pure broadcast.
            let evaluated = DataFrame::empty()
                .lazy()
                .select(literal_exprs)
                .collect()
                .ok()?;
            for (idx, col) in deferred.into_iter().zip(evaluated.columns()) {
                // A multi-value literal has no broadcast form; let the lazy
                // path evaluate it rather than silently keeping element 0.
                if col.len() != 1 {
                    return None;
                }
                let av = col.get(0).ok()?.into_static();
                if let SimpleOp::Broadcast { value, .. } = &mut ops[idx] {
                    *value = Scalar::new(col.dtype().clone(), av);
                }
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

/// Splits each bulk-read frame on `FILE_ID_COL` runs and applies the
/// per-file `LogicalRewrite`, yielding logical-schema frames in scan order.
pub(crate) struct LogicalScanIter {
    /// Raw bulk-read frames carrying `FILE_ID_COL`. One frame on the eager
    /// path; many on a streaming follow-up.
    source: Box<dyn Iterator<Item = anyhow::Result<DataFrame>> + Send>,
    /// `FILE_ID_COL` value → index in `rewrites`.
    path_index: HashMap<String, usize>,
    rewrites: Vec<LogicalRewrite>,
    /// Physical-name data conjuncts held back from the parquet scan because
    /// a DV is in play: the keep-mask addresses file row positions, so the
    /// predicate has to run after it.
    physical_predicate: Option<Expr>,
    /// Mixed atomic conjuncts (touching both partition and data cols) —
    /// applied after the select list materializes partition values.
    orphan_predicate: Option<Expr>,
    /// Index-aligned with `rewrites`: the pre-parsed fast path for each
    /// select list, when it qualifies and no predicate needs the lazy path.
    simple: Vec<Option<SimpleSelect>>,
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
    ) -> Self {
        let any_predicate = physical_predicate.is_some() || orphan_predicate.is_some();
        let simple = rewrites
            .iter()
            .map(|r| match (&r.select, any_predicate) {
                (Some(select), false) => SimpleSelect::parse(select),
                _ => None,
            })
            .collect();
        Self {
            source,
            path_index,
            rewrites,
            physical_predicate,
            orphan_predicate,
            simple,
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

        if let Some(fast) = &self.simple[idx] {
            df = fast
                .apply(&df)
                .map_err(|e| delta_kernel::Error::Generic(format!("logical rewrite eval: {e}")))?;
        } else if rewrite.select.is_some()
            || self.physical_predicate.is_some()
            || self.orphan_predicate.is_some()
        {
            let mut lazy = df.lazy();
            // Physical names are still in place here, before the select.
            if let Some(pred) = &self.physical_predicate {
                lazy = lazy.filter(pred.clone());
            }
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
}
