//! polars-FFI export driver shared by `TableScan` and `CdfTableScan`.

use std::mem::ManuallyDrop;

use polars::prelude::DataFrame;
use polars_ffi::version_0::{SeriesExport, export_column};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use rayon::prelude::*;

/// Wide frames amortize the rayon spin-up cost
const PAR_EXPORT_THRESHOLD: usize = 32;

#[repr(transparent)]
pub(crate) struct SendExport(pub(crate) ManuallyDrop<SeriesExport>);
unsafe impl Send for SendExport {}

static IMPORT_COLUMNS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn import_columns_fn(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    IMPORT_COLUMNS
        .get_or_try_init(py, || {
            py.import("polars")?
                .getattr("DataFrame")?
                .getattr("_import_columns")
                .map(|m| m.unbind())
        })
        .map(|p| p.bind(py))
}

#[derive(Default)]
pub(crate) struct MorselState {
    pub(crate) n_rows: Option<usize>,
    pub(crate) rows_emitted: usize,
}

impl MorselState {
    pub(crate) fn reset(&mut self) {
        self.rows_emitted = 0;
    }
}

/// Pull one morsel from `iter`: honour `n_rows`, skip empty frames, head-
/// trim the final partial morsel, rechunk, and FFI-export columns.
pub(crate) fn next_morsel<I>(
    state: &mut MorselState,
    iter: &mut I,
) -> anyhow::Result<Option<Vec<SendExport>>>
where
    I: Iterator<Item = anyhow::Result<DataFrame>> + ?Sized,
{
    loop {
        if let Some(cap) = state.n_rows
            && state.rows_emitted >= cap
        {
            return Ok(None);
        }
        let Some(next) = iter.next() else {
            return Ok(None);
        };
        let mut df = next?;
        if df.height() == 0 {
            continue;
        }
        if let Some(cap) = state.n_rows {
            let remaining = cap.saturating_sub(state.rows_emitted);
            if df.height() > remaining {
                df = df.head(Some(remaining));
            }
        }
        // Rechunk once here saves every downstream consumer (hash joins
        // especially) from rechunking per batch.
        df.rechunk_mut_par();
        state.rows_emitted += df.height();

        let columns = df.columns();
        let export = |c| SendExport(ManuallyDrop::new(export_column(c)));
        let exports: Vec<SendExport> = if columns.len() >= PAR_EXPORT_THRESHOLD {
            columns.par_iter().map(export).collect()
        } else {
            columns.iter().map(export).collect()
        };
        return Ok(Some(exports));
    }
}

/// Hand `exports` to polars' `DataFrame._import_columns` (rebuilds a
/// `pl.DataFrame` from the raw FFI pointer).
pub(crate) fn morsel_to_py<'py>(
    py: Python<'py>,
    exports: Vec<SendExport>,
) -> PyResult<Bound<'py, PyAny>> {
    let addr = exports.as_ptr() as usize;
    import_columns_fn(py)?.call1((addr, exports.len()))
}
