//! `LogicalScanIter` — splits bulk-read frames on file-id boundaries and
//! applies the per-file DV + kernel `Transform`.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use delta_kernel::Engine;
use delta_kernel::engine_data::EngineData;
use delta_kernel::scan::state::transform_to_logical;
use delta_kernel::schema::SchemaRef;
use polars::prelude::DataFrame;

use crate::engine::PolarsEngineData;
use crate::scan::plan::LogicalRewrite;
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
    ) -> Self {
        Self {
            source,
            path_index,
            rewrites,
            engine,
            physical_schema,
            logical_schema,
            pending: VecDeque::new(),
        }
    }

    /// Slice on file-id boundaries, push each per-file logical frame onto
    /// `pending`. Boundary detection runs through polars' vectorized
    /// `rle`, which returns one row per run with `{len, value}`.
    fn split_and_buffer(&mut self, df: DataFrame) -> Result<(), delta_kernel::Error> {
        if df.height() == 0 {
            return Ok(());
        }
        let file_col = df
            .column(FILE_ID_COL)
            .map_err(|e| delta_kernel::Error::Generic(format!("file-id column missing: {e}")))?;
        let runs = polars::prelude::rle(file_col).map_err(|e| {
            delta_kernel::Error::Generic(format!("rle on file-id column failed: {e}"))
        })?;
        // `rle` returns `Struct{ len: u32, value: <input dtype> }` — fixed
        // by polars contract, so post-rle field/type lookups are infallible.
        let runs = runs.struct_().expect("rle returns Struct");
        let lens = runs
            .field_by_name(polars::prelude::RLE_LENGTH_COLUMN_NAME)
            .expect("rle Struct has length field");
        let vals = runs
            .field_by_name(polars::prelude::RLE_VALUE_COLUMN_NAME)
            .expect("rle Struct has value field");
        let lens = lens.u32().expect("rle length is u32");
        let vals = vals.str().expect("rle value is String");

        let mut offset: usize = 0;
        for (len_opt, val_opt) in lens.iter().zip(vals.iter()) {
            let len = len_opt.unwrap_or(0) as usize;
            if let Some(file_id) = val_opt {
                let sub = df.slice(offset as i64, len);
                let out = self.apply_rewrite(file_id, sub);
                self.pending.push_back(out);
            }
            offset += len;
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
        // SV applied exactly once per file — take it instead of cloning. For
        // a million-row DV file this saves a per-batch `Vec<bool>` allocation.
        let rewrite = &mut self.rewrites[idx];
        let sv = rewrite.selection_vector.take();
        let transform = rewrite.transform.clone();

        let mut physical: Box<dyn EngineData> = Box::new(PolarsEngineData::new(df));
        if let Some(sv) = sv {
            physical = physical.apply_selection_vector(sv)?;
        }

        let logical = transform_to_logical(
            self.engine.as_ref(),
            physical,
            &self.physical_schema,
            &self.logical_schema,
            transform,
        )?;

        let out = logical
            .into_any()
            .downcast::<PolarsEngineData>()
            .map_err(|_| {
                delta_kernel::Error::Generic(
                    "transform_to_logical returned non-PolarsEngineData".into(),
                )
            })?;
        Ok(out.into_inner())
    }
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
