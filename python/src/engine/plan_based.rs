//! The handlers `PolarsEngine` hands kernel. Each is a thin wrapper that
//! turns the handler call into a plan `Operation` for `PolarsPlanExecutor`
//! — kernel's `PlanBasedEngine` layout, so the executor is the one surface
//! every read goes through. `parse_json` stays polars code: kernel's
//! plan-based handler parses with arrow.

use std::borrow::Cow;
use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::engine_data::{EngineData, FilteredEngineData};
use delta_kernel::expressions::PredicateRef;
use delta_kernel::plans::{IoOperation, Operation, PlanBuilder, PlanExecutor, PlanResult};
use delta_kernel::schema::{ColumnMetadataKey, SchemaRef, StructField};
use delta_kernel::transforms::{SchemaTransform, transform_output_type};
use delta_kernel::{
    FileDataReadResultIterator, FileMeta, FileSize, FileSlice, JsonHandler, KernelError,
    KernelResult, KernelResultIterator, KernelResultIteratorStatic, ParquetFooter, ParquetHandler,
    StorageHandler,
};
use url::Url;

use super::handlers::parse_json_strings;

const READ_ONLY: &str = "polars-deltalake is read-only; write support is not yet implemented";

pub(super) struct PlanBasedStorageHandler {
    executor: Arc<dyn PlanExecutor>,
}

impl PlanBasedStorageHandler {
    pub(super) fn new(executor: Arc<dyn PlanExecutor>) -> Self {
        Self { executor }
    }

    fn execute_io(&self, op: IoOperation) -> KernelResult<PlanResult> {
        self.executor.execute_op(Operation::IoOperation(op))
    }
}

impl StorageHandler for PlanBasedStorageHandler {
    fn list_from(&self, path: &Url) -> KernelResult<KernelResultIteratorStatic<FileMeta>> {
        self.execute_io(IoOperation::file_listing(path.clone()))?
            .into_file_meta()
    }

    fn read_files(&self, files: Vec<FileSlice>) -> KernelResult<KernelResultIteratorStatic<Bytes>> {
        self.execute_io(IoOperation::read_bytes(files))?
            .into_bytes()
    }

    fn copy_atomic(&self, src: &Url, dest: &Url) -> KernelResult<()> {
        self.execute_io(IoOperation::atomic_copy(src.clone(), dest.clone()))?
            .into_unit()
    }

    fn put(&self, path: &Url, data: Bytes, overwrite: bool) -> KernelResult<()> {
        self.execute_io(IoOperation::write_bytes(path.clone(), data, overwrite))?
            .into_unit()
    }

    fn head(&self, path: &Url) -> KernelResult<FileMeta> {
        self.execute_io(IoOperation::head_file(path.clone()))?
            .into_file_meta()?
            .next()
            .ok_or_else(|| KernelError::Generic(format!("head {path}: no file meta")))?
    }

    /// No `IoOperation` for delete yet (kernel #2820).
    fn delete(&self, _path: &Url) -> KernelResult<()> {
        Err(KernelError::Unsupported(READ_ONLY.into()))
    }
}

/// One plan per file, built when the iterator reaches it: the handler
/// contract orders batches by file and forbids merging across files, where
/// a scan node may read its files in any order.
fn scan_per_file(
    executor: Arc<dyn PlanExecutor>,
    files: &[FileMeta],
    schema: SchemaRef,
    scan: impl Fn(Vec<FileMeta>, SchemaRef) -> KernelResult<PlanBuilder> + Send + 'static,
) -> KernelResult<FileDataReadResultIterator> {
    let iter = Vec::from(files).into_iter().flat_map(move |file| {
        let batches: FileDataReadResultIterator = match scan(vec![file], schema.clone())
            .and_then(|builder| builder.build())
            .and_then(|plan| executor.execute_op(Operation::QueryPlan(plan)))
            .and_then(PlanResult::into_data)
        {
            Ok(batches) => batches,
            Err(e) => Box::new(std::iter::once(Err(e))),
        };
        batches
    });
    Ok(Box::new(iter))
}

pub(super) struct PlanBasedJsonHandler {
    executor: Arc<dyn PlanExecutor>,
}

impl PlanBasedJsonHandler {
    pub(super) fn new(executor: Arc<dyn PlanExecutor>) -> Self {
        Self { executor }
    }
}

impl JsonHandler for PlanBasedJsonHandler {
    fn parse_json(
        &self,
        json_strings: Box<dyn EngineData>,
        output_schema: SchemaRef,
    ) -> KernelResult<Box<dyn EngineData>> {
        parse_json_strings(json_strings, output_schema)
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        scan_per_file(
            self.executor.clone(),
            files,
            physical_schema,
            |files, schema| PlanBuilder::scan_json(files, &[], schema),
        )
    }

    fn write_json_file(
        &self,
        _path: &Url,
        _data: KernelResultIterator<'_, FilteredEngineData>,
        _overwrite: bool,
    ) -> KernelResult<FileSize> {
        Err(KernelError::Unsupported(READ_ONLY.into()))
    }
}

pub(super) struct PlanBasedParquetHandler {
    executor: Arc<dyn PlanExecutor>,
}

impl PlanBasedParquetHandler {
    pub(super) fn new(executor: Arc<dyn PlanExecutor>) -> Self {
        Self { executor }
    }
}

/// Drops `parquet.field.id` at every depth. The handler contract resolves
/// such a field by ID, which the executor's scan refuses; this handler has
/// always matched by name instead, and every column-mapped table we read
/// stores the physical names in its files.
struct DropFieldIds;

impl<'a> SchemaTransform<'a> for DropFieldIds {
    transform_output_type!(|'a, T| Cow<'a, T>);

    fn transform_struct_field(&mut self, field: &'a StructField) -> Cow<'a, StructField> {
        let key = ColumnMetadataKey::ParquetFieldId.as_ref();
        let data_type = self.transform(&field.data_type);
        if !field.metadata.contains_key(key) && matches!(data_type, Cow::Borrowed(_)) {
            return Cow::Borrowed(field);
        }
        Cow::Owned(
            StructField::new(field.name.clone(), data_type.into_owned(), field.nullable)
                .with_metadata(
                    field
                        .metadata
                        .iter()
                        .filter(|(k, _)| k.as_str() != key)
                        .map(|(k, v)| (k.clone(), v.clone())),
                ),
        )
    }
}

impl ParquetHandler for PlanBasedParquetHandler {
    /// The predicate is dropped, as in kernel's own plan-based handler; the
    /// one caller this engine serves (CDF) passes `None`.
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        let schema = match DropFieldIds.transform_struct(physical_schema.as_ref()) {
            Cow::Borrowed(_) => physical_schema,
            Cow::Owned(stripped) => Arc::new(stripped),
        };
        scan_per_file(self.executor.clone(), files, schema, |files, schema| {
            PlanBuilder::scan_parquet(files, &[], schema)
        })
    }

    fn write_parquet_file(
        &self,
        _location: Url,
        _data: KernelResultIteratorStatic<Box<dyn EngineData>>,
    ) -> KernelResult<FileSize> {
        Err(KernelError::Unsupported(READ_ONLY.into()))
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> KernelResult<ParquetFooter> {
        self.executor.read_parquet_footer(file.clone())
    }
}

#[cfg(test)]
mod parquet_handler_tests {
    use std::path::Path;

    use delta_kernel::schema::{DataType, MetadataColumnSpec, StructField, StructType};
    use polars::prelude::{DataFrame, ParquetWriter};

    use super::*;
    use crate::engine::handlers::ObjectStoreStorageHandler;
    use crate::engine::{PolarsEngineData, PolarsPlanExecutor};

    fn handler(base: &Url) -> PlanBasedParquetHandler {
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(base, std::iter::empty(), rt).unwrap());
        PlanBasedParquetHandler::new(Arc::new(PolarsPlanExecutor::new(storage, None)))
    }

    fn write_parquet(dir: &Path, name: &str, vals: Vec<i64>) {
        let mut df = polars::df!("x" => vals).unwrap();
        ParquetWriter::new(std::fs::File::create(dir.join(name)).unwrap())
            .finish(&mut df)
            .unwrap();
    }

    fn file_meta(base: &Url, dir: &Path, name: &str) -> FileMeta {
        FileMeta {
            location: base.join(name).unwrap(),
            last_modified: 0,
            size: std::fs::metadata(dir.join(name)).unwrap().len(),
        }
    }

    fn read(base: &Url, files: &[FileMeta], schema: SchemaRef) -> Vec<Box<dyn EngineData>> {
        handler(base)
            .read_parquet_files(files, schema, None)
            .unwrap()
            .collect::<KernelResult<Vec<_>>>()
            .unwrap()
    }

    fn frame(batch: &dyn EngineData) -> &DataFrame {
        batch
            .any_ref()
            .downcast_ref::<PolarsEngineData>()
            .unwrap()
            .dataframe()
    }

    /// The footer path synthesizes metadata columns without reading any data
    /// pages; each spec is built from its own kind.
    #[test]
    fn footer_fast_path_synthesizes_each_metadata_column() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(dir.path(), "f.parquet", vec![1, 2, 3]);
        let base = Url::from_directory_path(dir.path()).unwrap();

        // No data fields: the whole read schema is metadata columns.
        let schema = Arc::new(
            StructType::try_new([
                StructField::create_metadata_column("ri", MetadataColumnSpec::RowIndex),
                StructField::create_metadata_column("fp", MetadataColumnSpec::FilePath),
            ])
            .unwrap(),
        );
        let files = vec![file_meta(&base, dir.path(), "f.parquet")];
        let batches = read(&base, &files, schema);

        let df = frame(batches[0].as_ref());
        assert_eq!(df.height(), 3, "row count comes from the footer");
        let ri: Vec<i64> = df
            .column("ri")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(ri, vec![0, 1, 2]);
        assert_eq!(
            df.column("fp").unwrap().str().unwrap().get(0).unwrap(),
            files[0].location.as_str(),
            "the path column carries the file URL"
        );
    }

    /// A column-mapped table's physical schema carries `parquet.field.id`;
    /// the handler matches by physical name, as it did before it ran on
    /// plans (kernel's CDF reads hit this — Spark suite, PR #31).
    #[test]
    fn field_ids_are_matched_by_name() {
        use delta_kernel::schema::MetadataValue;

        let dir = tempfile::tempdir().unwrap();
        let mut df = polars::df!("col-abc" => [1i64, 2, 3]).unwrap();
        ParquetWriter::new(std::fs::File::create(dir.path().join("a.parquet")).unwrap())
            .finish(&mut df)
            .unwrap();
        let base = Url::from_directory_path(dir.path()).unwrap();

        let schema = Arc::new(
            StructType::try_new([StructField::nullable("col-abc", DataType::LONG)
                .with_metadata([("parquet.field.id", MetadataValue::Number(5))])])
            .unwrap(),
        );
        let files = vec![file_meta(&base, dir.path(), "a.parquet")];
        let batches = read(&base, &files, schema);
        let xs: Vec<i64> = frame(batches[0].as_ref())
            .column("col-abc")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(xs, vec![1, 2, 3]);
    }

    /// A schema with no fields at all (CDF projecting only its own metadata
    /// columns) still reports each file's row count.
    #[test]
    fn empty_schema_reports_each_files_row_count() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(dir.path(), "a.parquet", vec![1, 2, 3]);
        write_parquet(dir.path(), "b.parquet", vec![4, 5, 6]);
        let base = Url::from_directory_path(dir.path()).unwrap();

        let schema = Arc::new(StructType::try_new(Vec::<StructField>::new()).unwrap());
        let files = vec![
            file_meta(&base, dir.path(), "a.parquet"),
            file_meta(&base, dir.path(), "b.parquet"),
        ];
        let heights: Vec<usize> = read(&base, &files, schema)
            .iter()
            .map(|b| b.len())
            .collect();
        assert_eq!(heights, vec![3, 3]);
    }

    /// `FileDataReadResultIterator` contract: data arrives in file order and
    /// batches do not span files.
    #[test]
    fn batches_do_not_span_files() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(dir.path(), "a.parquet", vec![1, 2, 3]);
        write_parquet(dir.path(), "b.parquet", vec![4, 5, 6]);
        let base = Url::from_directory_path(dir.path()).unwrap();

        let schema =
            Arc::new(StructType::try_new([StructField::not_null("x", DataType::LONG)]).unwrap());
        let files = vec![
            file_meta(&base, dir.path(), "a.parquet"),
            file_meta(&base, dir.path(), "b.parquet"),
        ];
        let batches = read(&base, &files, schema);
        let heights: Vec<usize> = batches.iter().map(|b| b.len()).collect();
        assert_eq!(heights, vec![3, 3], "batches must not span files");
        let xs: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                frame(b.as_ref())
                    .column("x")
                    .unwrap()
                    .i64()
                    .unwrap()
                    .iter()
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(xs, vec![1, 2, 3, 4, 5, 6], "file and row order hold");
    }

    /// A RowIndex column restarts per file and a FilePath column carries the
    /// file's path — populated, not null-filled.
    #[test]
    fn metadata_columns_are_synthesized() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(dir.path(), "a.parquet", vec![10, 20, 30]);
        write_parquet(dir.path(), "b.parquet", vec![40]);
        let base = Url::from_directory_path(dir.path()).unwrap();

        let schema = Arc::new(
            StructType::try_new([
                StructField::not_null("x", DataType::LONG),
                StructField::create_metadata_column("ridx", MetadataColumnSpec::RowIndex),
                StructField::create_metadata_column("fname", MetadataColumnSpec::FilePath),
            ])
            .unwrap(),
        );
        let files = vec![
            file_meta(&base, dir.path(), "a.parquet"),
            file_meta(&base, dir.path(), "b.parquet"),
        ];
        let batches = read(&base, &files, schema);

        let mut ridx = Vec::new();
        let mut fnames = Vec::new();
        for b in &batches {
            let df = frame(b.as_ref());
            let col = df.column("ridx").unwrap();
            assert_eq!(col.dtype(), &polars::prelude::DataType::Int64);
            ridx.extend(col.i64().unwrap().iter().map(|v| v.unwrap()));
            fnames.extend(
                df.column("fname")
                    .unwrap()
                    .str()
                    .unwrap()
                    .iter()
                    .map(|v| v.unwrap().to_string()),
            );
        }
        assert_eq!(ridx, vec![0, 1, 2, 0], "row index restarts per file");
        assert!(
            fnames[0].ends_with("a.parquet") && fnames[3].ends_with("b.parquet"),
            "got: {fnames:?}"
        );
    }

    /// Unsupported metadata specs error loudly rather than come back as
    /// silently null-filled data columns.
    #[test]
    fn unsupported_metadata_spec_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(dir.path(), "a.parquet", vec![1]);
        let base = Url::from_directory_path(dir.path()).unwrap();

        let schema = Arc::new(
            StructType::try_new([
                StructField::not_null("x", DataType::LONG),
                StructField::create_metadata_column("rid", MetadataColumnSpec::RowId),
            ])
            .unwrap(),
        );
        let files = vec![file_meta(&base, dir.path(), "a.parquet")];
        let result = handler(&base)
            .read_parquet_files(&files, schema, None)
            .and_then(|it| it.collect::<KernelResult<Vec<_>>>());
        let err = match result {
            Ok(_) => panic!("RowId is unsupported and must error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("metadata column"), "got: {err}");
    }
}
