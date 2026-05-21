//! Native polars I/O plugin for Delta Lake — a delta-kernel `Engine`
//! backed entirely by polars-io + object_store.

mod consts;
mod engine;
mod errors;
mod scan;
mod translation;

use pyo3::prelude::*;

use crate::scan::DeltaSource;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DeltaSource>()?;
    Ok(())
}
