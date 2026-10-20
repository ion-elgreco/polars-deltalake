use delta_kernel::KernelError;
use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;

pub(crate) fn to_kernel_err<E: std::fmt::Display>(e: E) -> KernelError {
    KernelError::Generic(e.to_string())
}

pub(crate) fn py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}
