#![feature(iterator_try_collect)]

mod error;
use arrow_schema::Schema as ArrowSchema;
use deltalake::kernel::ReaderFeatures;
use deltalake::DeltaTableError;
use error::PythonError;
use polars::io::cloud::CloudOptions;
use polars::io::parquet::ParallelStrategy;
use polars::prelude::Schema as PolarsSchema;
use polars_arrow::datatypes::ArrowSchema as PolarsArrowSchema;
use polars_arrow::datatypes::Field;
use polars_lazy::dsl::functions::concat_lf_diagonal;
use polars_lazy::dsl::UnionArgs;
use polars_lazy::frame::ScanArgsParquet;
use polars_lazy::prelude::LazyFrame;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
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
    retries: Option<i64>,
) -> PyResult<PyLazyFrame> {
    let mut builder = deltalake::DeltaTableBuilder::from_uri(&uri);
    if let Some(storage_options) = storage_options.clone() {
        builder = builder.with_storage_options(storage_options)
    }
    if let Some(version) = version {
        builder = builder.with_version(version)
    }

    let table = rt()?.block_on(builder.load()).map_err(PythonError::from)?;

    if let Some(features) = table
        .protocol()
        .map_err(PythonError::from)?
        .to_owned()
        .reader_features
    {
        if features.contains(&ReaderFeatures::DeletionVectors) {
            return Err(PyNotImplementedError::new_err(
                "Deletion Vectors are not supported yet.",
            ));
        } else if features.contains(&ReaderFeatures::ColumnMapping) {
            return Err(PyNotImplementedError::new_err(
                "Column Mapping is not supported yet.",
            ));
        }
    }
    let file_paths = table
        .get_file_uris()
        .map_err(PythonError::from)?
        .map(PathBuf::from)
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
        .map_err(|_| {
            DeltaTableError::Generic("can't convert Delta schema into Arrow schema".to_string())
        })
        .map_err(PythonError::from)?;

    let cloud_options = storage_options
        .map(|opts| CloudOptions::from_untyped_config(&uri, &opts))
        .transpose()
        .map_err(PyPolarsErr::from)?
        .map(|mut options| {
            options.max_retries = retries.unwrap_or(10) as usize;
            options
        });

    let frames = file_paths
        .iter()
        .map(|path| {
            LazyFrame::scan_parquet(
                path,
                ScanArgsParquet {
                    n_rows: None,
                    cache: true,
                    parallel: ParallelStrategy::Auto,
                    rechunk: false,
                    row_index: None,
                    low_memory,
                    cloud_options: cloud_options.clone(),
                    use_statistics,
                    hive_partitioning,
                },
            )
        })
        .try_collect::<Vec<_>>();

    let mut final_frame = concat_lf_diagonal(
        frames.map_err(PyPolarsErr::from)?,
        UnionArgs {
            rechunk: false,
            parallel: true,
            ..Default::default()
        },
    )
    .map_err(PyPolarsErr::from)?;

    let polars_arrow_schema: PolarsArrowSchema = schema
        .fields()
        .iter()
        .map(|f| (*f).clone().into())
        .collect::<Vec<Field>>()
        .into();
    let polars_schema: PolarsSchema = polars_arrow_schema.clone().into();

    final_frame = final_frame.cast(
        polars_schema
            .iter()
            .map(|(field, dtype)| (field.as_str(), dtype.to_owned()))
            .collect(),
        true,
    );

    Ok(PyLazyFrame(final_frame))
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
