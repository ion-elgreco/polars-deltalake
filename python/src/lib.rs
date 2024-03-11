mod error;

use arrow_schema::Schema as ArrowSchema;
use deltalake::storage::StorageOptions;
use polars::io::cloud::CloudOptions;
use polars::io::parquet::ParallelStrategy;
use polars::io::predicates::{BatchStats, ColumnStats};
// use deltalake::kernel::{Schema, SchemaRef};
// use deltalake::{DeltaOps, DeltaResult};
// use polars_plan::logical_plan::hive;
use deltalake::DeltaTableError;
use error::PythonError;
use polars::prelude::Schema;
use polars_arrow::datatypes::ArrowSchema as PolarsArrowSchema;
use polars_arrow::datatypes::Field;
use polars_lazy::prelude::LazyFrame;
use polars_plan::logical_plan::{FileInfo, FileScan, LogicalPlan};
use polars_plan::prelude::{FileScanOptions, ParquetOptions};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;
use pyo3::{pyfunction, pymodule, PyResult, Python};
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PyLazyFrame;
use std::collections::HashMap;
use std::path::PathBuf;

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
fn custom_scan_delta(
    uri: String,
    low_memory: bool,
    use_statistics: bool,
    version: Option<i64>,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<PyLazyFrame> {
    let mut builder = deltalake::DeltaTableBuilder::from_uri(&uri);
    if let Some(storage_options) = storage_options.clone() {
        builder = builder.with_storage_options(storage_options)
    }
    if let Some(version) = version {
        builder = builder.with_version(version)
    }

    let table = rt()?.block_on(builder.load()).map_err(PythonError::from)?;

    let file_paths = table
        .get_file_uris()
        .map_err(PythonError::from)?
        .map(|path| PathBuf::from(path))
        .collect::<Vec<PathBuf>>();

    let partition_cols = table
        .metadata()
        .map_err(PythonError::from)?
        .partition_columns
        .clone();

    let hive_partitioning = !partition_cols.is_empty();

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
        .filter(|f| !partition_cols.contains(f.name()))
        .map(|f| (*f).clone().into())
        .collect::<Vec<Field>>()
        .into();

    let polars_schema: Schema = polars_arrow_schema.clone().into();

    dbg!(polars_schema.to_arrow(true));

    let mut file_info = FileInfo::new(
        polars_schema.clone().into(),
        Some(polars_schema.to_arrow(true).into()),
        (None, 10),
    );

    if hive_partitioning {
        file_info
            .init_hive_partitions(file_paths[0].as_path())
            .map_err(PyPolarsErr::from)?;
    }
    let options = FileScanOptions {
        with_columns: None,
        cache: true,
        n_rows: None,
        rechunk: false,
        row_index: None,
        file_counter: Default::default(),
        hive_partitioning: hive_partitioning,
    };

    let cloud_options = storage_options
        .map(|opts| CloudOptions::from_untyped_config(&uri, &opts))
        .transpose()
        .map_err(PyPolarsErr::from)?;

    let scan_type = FileScan::Parquet {
        options: ParquetOptions {
            parallel: ParallelStrategy::Auto,
            low_memory,
            use_statistics,
        },
        cloud_options,
        metadata: None,
    };

    let plan = LogicalPlan::Scan {
        paths: file_paths.into(),
        file_info,
        file_options: options,
        predicate: None,
        scan_type: scan_type,
    };
    let frame: LazyFrame = plan.into();
    Ok(PyLazyFrame(frame))
}

#[pymodule]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(pyo3::wrap_pyfunction!(custom_scan_delta, m)?)?;
    deltalake::azure::register_handlers(None);
    deltalake::gcp::register_handlers(None);
    deltalake::aws::register_handlers(None);
    Ok(())
}
