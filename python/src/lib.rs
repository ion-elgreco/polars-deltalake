//! Native polars I/O plugin for Delta Lake — a delta-kernel `Engine`
//! backed entirely by polars-io + object_store.

mod consts;
mod engine;
mod errors;
mod scan;
mod translation;

use pyo3::prelude::*;
#[cfg(not(test))]
use pyo3_polars::PolarsAllocator;

use crate::scan::{CdfTableScan, CdfTableState, TableScan, TableState};

#[cfg(not(test))]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TableState>()?;
    m.add_class::<TableScan>()?;
    m.add_class::<CdfTableState>()?;
    m.add_class::<CdfTableScan>()?;
    Ok(())
}
