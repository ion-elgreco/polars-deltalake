//! `LogicalScanIter` — splits bulk-read frames on file-id boundaries and
//! applies the per-file DV + kernel `Transform`.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use delta_kernel::Engine;
use delta_kernel::ExpressionEvaluator;
use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::ExpressionRef;
use delta_kernel::schema::SchemaRef;
use polars::prelude::{DataFrame, Expr, IntoLazy, StringChunked};
use polars_plan::dsl::Engine as PolarsEngineMode;

use crate::engine::PolarsEngineData;
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
    engine: Arc<dyn Engine>,
    physical_schema: SchemaRef,
    logical_schema: SchemaRef,
    /// Parallel to `rewrites`. Reused across batches of the same file.
    evaluator_cache: Vec<Option<Arc<dyn ExpressionEvaluator>>>,
    /// Mixed atomic conjuncts (touching both partition and data cols) —
    /// applied after `transform_to_logical` materializes partition values.
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
        engine: Arc<dyn Engine>,
        physical_schema: SchemaRef,
        logical_schema: SchemaRef,
        orphan_predicate: Option<Expr>,
    ) -> Self {
        let n_files = rewrites.len();
        Self {
            source,
            path_index,
            rewrites,
            engine,
            physical_schema,
            logical_schema,
            evaluator_cache: vec![None; n_files],
            orphan_predicate,
            pending: VecDeque::new(),
        }
    }

    /// Cached per `idx`; building one parses kernel `Transform` into polars `Expr`s.
    fn evaluator_for(
        &mut self,
        idx: usize,
        transform: ExpressionRef,
    ) -> Result<Arc<dyn ExpressionEvaluator>, delta_kernel::Error> {
        if let Some(e) = &self.evaluator_cache[idx] {
            return Ok(e.clone());
        }
        let evaluator = self.engine.evaluation_handler().new_expression_evaluator(
            self.physical_schema.clone(),
            transform,
            self.logical_schema.as_ref().clone().into(),
        )?;
        self.evaluator_cache[idx] = Some(evaluator.clone());
        Ok(evaluator)
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

    /// Drop the file-id column, apply DV + `Transform`, return logical frame.
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
        let sv_chunk = rewrite
            .dv
            .as_mut()
            .map(|state| build_keep_mask(state, df.height()));
        let transform = rewrite.transform.clone();

        let mut physical: Box<dyn EngineData> = Box::new(PolarsEngineData::new(df));
        if let Some(sv) = sv_chunk {
            physical = physical.apply_selection_vector(sv)?;
        }

        let logical: Box<dyn EngineData> = match transform {
            Some(t) => self.evaluator_for(idx, t)?.evaluate(physical.as_ref())?,
            None => physical,
        };

        let out = logical
            .into_any()
            .downcast::<PolarsEngineData>()
            .map_err(|_| {
                delta_kernel::Error::Generic(
                    "transform_to_logical returned non-PolarsEngineData".into(),
                )
            })?;
        let mut df = out.into_inner();
        if let Some(pred) = &self.orphan_predicate {
            df = df
                .lazy()
                .filter(pred.clone())
                .collect_with_engine(PolarsEngineMode::Streaming)
                .map_err(|e| delta_kernel::Error::Generic(format!("orphan predicate eval: {e}")))?;
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
