fn main() {
    // uv-managed CPython keeps libpython off the loader path, and Cargo
    // applies pyo3-ffi's own rpath only to pyo3-ffi's binaries.
    #[cfg(feature = "libpython-rpath")]
    pyo3_build_config::add_libpython_rpath_link_args();
}
