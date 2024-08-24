#![feature(iterator_try_collect)]

mod data;
mod engine;
use std::collections::HashMap;

use delta_kernel::Table;
#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;
use polars_lazy::frame::LazyFrame;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::{exceptions::PyRuntimeError, pyfunction, pymodule, PyResult};
use pyo3_polars::PyLazyFrame;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[inline]
fn rt() -> pyo3::PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new().map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn scan_delta(
    uri: String,
    version: Option<u64>,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<PyLazyFrame> {
    let dt = Table::try_from_uri(uri).map_err(|err| PyIOError::new_err(err.to_string()))?;
    let table_snapshot = dt
        .snapshot(engine, version)
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
    Ok(PyLazyFrame(LazyFrame()))
}

#[pymodule]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(pyo3::wrap_pyfunction!(scan_delta, m)?)?;
    Ok(())
}
