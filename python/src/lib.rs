mod error;

use arrow_schema::Schema as ArrowSchema;
// use deltalake::kernel::{Schema, SchemaRef};
// use deltalake::{DeltaOps, DeltaResult};
use polars::prelude::{Schema, SchemaRef};
use deltalake::{DeltaTable, DeltaTableError};
use error::PythonError;
use polars_lazy::prelude::LazyFrame;
use polars_arrow::datatypes::ArrowSchema as PolarsArrowSchema;
use polars_arrow::datatypes::Field;
use polars_plan::logical_plan::FileScan;
use polars_plan::logical_plan::{FileInfo, LogicalPlan};
use polars_plan::prelude::{FileScanOptions, ParquetOptions};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;
use pyo3::{pyfunction, pymodule, PyResult, Python};
use pyo3_polars::PyLazyFrame;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[inline]
fn rt() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new().map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn scan_delta(
    uri: String,
    version: Option<i64>,
    storage_options: Option<HashMap<String, String>>,
    parallel: bool,
    low_memory: bool,
    use_statistics: bool,
) -> PyResult<()> {
    let mut builder = deltalake::DeltaTableBuilder::from_uri(uri);
    if let Some(storage_options) = storage_options {
        builder = builder.with_storage_options(storage_options)
    }
    if let Some(version) = version {
        builder = builder.with_version(version)
    }
    let table = rt()?.block_on(builder.load()).map_err(PythonError::from)?;

    let file_paths = table.get_file_uris().map_err(PythonError::from)?.map(|path| PathBuf::from(path)).collect::<Vec<PathBuf>>();

    let schema: ArrowSchema = table
        .snapshot()
        .map_err(PythonError::from)?
        .schema()
        .try_into()
        .map_err(|_| DeltaTableError::Generic("can't convert schema".to_string()))
        .map_err(PythonError::from)?;

    let polars_arrow_schema: PolarsArrowSchema = schema
        .all_fields()
        .iter()
        .map(|f| (*f).clone().into())
        .collect::<Vec<Field>>()
        .into();

    let polars_schema: Schema = polars_arrow_schema.into();
    // let schema_ref: SchemaRef = polars_schema.into();

    let mut file_info = FileInfo::new(
        polars_schema.into(),
        Some(polars_arrow_schema.into()),
        (None, 10),
    );


    let options = FileScanOptions {
        with_columns: None,
        cache: true,
        n_rows: None,
        rechunk: false,
        row_index: None,
        file_counter: Default::default(),
        hive_partitioning: false,
    };

    let scan_type = FileScan::Parquet {
            options: ParquetOptions {
                parallel,
                low_memory,
                use_statistics,
            },
            cloud_options: None,
            metadata: None,
        };

    let mut lf: LazyFrame = LogicalPlan::Scan {
        paths:file_paths.into(),
        file_info,
        file_options: options,
        predicate: None,
        scan_type: scan_type
    }
    .into()
    .build()
    .into();


    dbg!(table.version());
    Ok(PyLazyFrame(lf))
}

#[pymodule]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(pyo3::wrap_pyfunction!(scan_delta, m)?)?;
    deltalake::azure::register_handlers(None);
    Ok(())
}
