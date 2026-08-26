//! `delta_kernel::ParquetHandler` over polars-io. Each file gets its own
//! `DslBuilder::scan_parquet` plan (the kernel contract forbids merging
//! engine data across file boundaries) with the kernel physical schema
//! attached and polars-io's `missing_struct_fields=Insert` /
//! `extra_columns=Ignore` policies set; polars-io parallelises row groups
//! and columns within the file, and reads come back already shaped to the
//! kernel contract. RowIndex / FilePath metadata columns are synthesized
//! per file, never read from the data pages.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::{
    ColumnMetadataKey, DataType as KernelDataType, MetadataColumnSpec, SchemaRef, StructField,
    StructType,
};
use delta_kernel::{
    DeltaResult, Error, FileDataReadResultIterator, FileMeta, ParquetFooter, ParquetHandler,
    StorageHandler,
};
use polars::io::cloud::CloudOptions;
use polars::io::parquet::read::{ParquetOptions, infer_schema};
use polars::lazy::frame::LazyFrame;
use polars::prelude::{
    ChunkFull, Column, DataFrame, DataType as PlDataType, Expr, IntoColumn, IntoSeries, NamedFrom,
    Series, StringChunked, col, lit,
};
use polars_buffer::Buffer;
use polars_parquet::parquet::metadata::FileMetadata;
use polars_parquet::parquet::{FOOTER_SIZE, PARQUET_MAGIC, read::deserialize_metadata};
use polars_plan::dsl::{
    CastColumnsPolicy, DslBuilder, ExtraColumnsPolicy, MissingColumnsPolicy, ScanSources,
    UnifiedScanArgs,
};
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use polars_utils::pl_str::PlSmallStr;
use url::Url;

use crate::engine::PolarsEngineData;
use crate::errors::to_kernel_err;
use crate::translation::schema::{ArrowSchemaExt, KernelSchemaExt};

use super::storage::ObjectStoreStorageHandler;

/// The metadata columns this engine synthesizes per file.
#[derive(Clone, Copy)]
pub(crate) enum SyntheticColumn {
    RowIndex,
    FilePath,
}

/// The synthetic metadata columns a read schema requests.
#[derive(Default)]
pub(crate) struct MetadataColumns {
    /// In schema order, carrying the accepted kind rather than the raw
    /// spec, so a synthesis match cannot fall through.
    cols: Vec<(PlSmallStr, SyntheticColumn)>,
}

impl MetadataColumns {
    pub(crate) fn row_index(&self) -> Option<&PlSmallStr> {
        self.cols
            .iter()
            .find_map(|(name, kind)| matches!(kind, SyntheticColumn::RowIndex).then_some(name))
    }

    pub(crate) fn file_path(&self) -> Option<&PlSmallStr> {
        self.cols
            .iter()
            .find_map(|(name, kind)| matches!(kind, SyntheticColumn::FilePath).then_some(name))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }
}

/// Split `schema` into the fields read from the data pages and the
/// requested synthetic metadata columns. One implementation for the
/// classic `ParquetHandler` and the plan executor's scan nodes, so the two
/// read paths cannot drift on which specs the engine supports. RowIndex is
/// the 0-based position within each file, FilePath its URL; any other spec
/// errors loudly — a null-fill would silently misreport it.
pub(crate) fn split_metadata_columns(
    schema: &StructType,
    context: &str,
) -> DeltaResult<(StructType, MetadataColumns)> {
    let mut row_index: Option<PlSmallStr> = None;
    let mut file_path: Option<PlSmallStr> = None;
    let mut cols: Vec<(PlSmallStr, SyntheticColumn)> = Vec::new();
    for f in schema.fields() {
        let Some(spec) = f.get_metadata_column_spec() else {
            continue;
        };
        let name = PlSmallStr::from_str(f.name.as_str());
        let (kind, slot) = match spec {
            MetadataColumnSpec::RowIndex => (SyntheticColumn::RowIndex, &mut row_index),
            MetadataColumnSpec::FilePath => (SyntheticColumn::FilePath, &mut file_path),
            other => {
                return Err(Error::Unsupported(format!(
                    "{context}: metadata column {other:?} is not supported"
                )));
            }
        };
        if slot.replace(name.clone()).is_some() {
            return Err(Error::Unsupported(format!(
                "{context}: more than one {spec:?} metadata column"
            )));
        }
        cols.push((name, kind));
    }
    let read_fields: Vec<StructField> = schema
        .fields()
        .filter(|f| f.get_metadata_column_spec().is_none())
        .cloned()
        .collect();
    Ok((StructType::try_new(read_fields)?, MetadataColumns { cols }))
}

pub(crate) struct PolarsParquetHandler {
    storage: Arc<ObjectStoreStorageHandler>,
    /// Pre-built once per table; `None` for `file://`. Cloned once per
    /// `read_parquet_files` call into `UnifiedScanArgs`.
    cloud_opts: Option<CloudOptions>,
}

impl PolarsParquetHandler {
    pub(crate) fn new(
        storage: Arc<ObjectStoreStorageHandler>,
        storage_options: HashMap<String, String>,
    ) -> DeltaResult<Self> {
        let cloud_opts = cloud_options_for(storage.base_url(), storage_options)?;
        Ok(Self {
            storage,
            cloud_opts,
        })
    }

    /// Shared with the scan-driver's bulk-read path.
    pub(crate) fn cloud_options(&self) -> Option<&CloudOptions> {
        self.cloud_opts.as_ref()
    }
}

fn cloud_options_for(
    url: &Url,
    storage_options: HashMap<String, String>,
) -> DeltaResult<Option<CloudOptions>> {
    let scheme = match url.scheme() {
        "file" => return Ok(None),
        "s3" => CloudScheme::S3,
        "s3a" => CloudScheme::S3a,
        "az" => CloudScheme::Az,
        "azure" => CloudScheme::Azure,
        "abfs" => CloudScheme::Abfs,
        "abfss" => CloudScheme::Abfss,
        "adl" => CloudScheme::Adl,
        "gs" => CloudScheme::Gs,
        "gcs" => CloudScheme::Gcs,
        "http" => CloudScheme::Http,
        "https" => CloudScheme::Https,
        other => {
            return Err(Error::Unsupported(format!(
                "PolarsParquetHandler: unknown URL scheme '{other}'"
            )));
        }
    };
    CloudOptions::from_untyped_config(Some(scheme), storage_options)
        .map(Some)
        .map_err(to_kernel_err)
}

impl ParquetHandler for PolarsParquetHandler {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        let (read_schema, meta) =
            split_metadata_columns(&physical_schema, "read_parquet_files")?;
        let row_index = meta.row_index().cloned();
        let file_path = meta.file_path().cloned();
        let meta_cols = meta.cols;

        // Footer fast path — nothing needed from the data pages (empty or
        // metadata-only projection): `num_rows` drives synthesis, one batch
        // per file per the no-merging contract.
        if read_schema.fields().next().is_none() {
            let per_file = files
                .iter()
                .map(|f| {
                    fetch_parquet_metadata(self.storage.as_ref(), f)
                        .map(|m| (m.num_rows, f.location.clone()))
                })
                .collect::<DeltaResult<Vec<_>>>()?;
            return Ok(Box::new(per_file.into_iter().map(
                move |(rows, location)| {
                    let columns: Vec<Column> = meta_cols
                        .iter()
                        .map(|(name, kind)| match kind {
                            SyntheticColumn::RowIndex => {
                                Series::new(name.clone(), (0..rows as i64).collect::<Vec<i64>>())
                                    .into_column()
                            }
                            SyntheticColumn::FilePath => {
                                StringChunked::full(name.clone(), location.as_str(), rows)
                                    .into_series()
                                    .into_column()
                            }
                        })
                        .collect();
                    let df = if columns.is_empty() {
                        DataFrame::empty_with_height(rows)
                    } else {
                        DataFrame::new(rows, columns).map_err(to_kernel_err)?
                    };
                    Ok(Box::new(PolarsEngineData::new(df)) as Box<dyn EngineData>)
                },
            )));
        }

        // Translation failure is non-fatal: kernel already pruned files via
        // stats, so we just skip row-level pushdown and let polars filter.
        let polars_predicate: Option<Expr> = predicate.as_ref().and_then(|p| {
            crate::translation::from_kernel::translate_predicate(p.as_ref(), None).ok()
        });

        let select_exprs = crate::scan::select_exprs_for_schema(physical_schema.as_ref());

        let paths: Vec<(PlRefPath, String)> = files
            .iter()
            .map(|f| Ok((path_for_polars_io(&f.location)?, f.location.to_string())))
            .collect::<DeltaResult<_>>()?;

        read_batch(
            paths,
            FileScanArgs {
                cloud_opts: self.cloud_opts.clone(),
                select_exprs,
                predicate: polars_predicate,
                read_schema,
                row_index,
                file_path,
            },
        )
    }

    fn write_parquet_file(
        &self,
        _location: Url,
        _data: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>,
    ) -> DeltaResult<()> {
        Err(Error::Unsupported(
            "polars-deltalake is read-only; write support is not yet implemented".into(),
        ))
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> DeltaResult<ParquetFooter> {
        kernel_parquet_footer(self.storage.as_ref(), file)
    }
}

/// Footer → kernel schema, shared by the `ParquetHandler` and the plan
/// executor's `IoOperation::ParquetFooter` arm.
pub(crate) fn kernel_parquet_footer(
    storage: &ObjectStoreStorageHandler,
    file: &FileMeta,
) -> DeltaResult<ParquetFooter> {
    let metadata = fetch_parquet_metadata(storage, file)?;
    let arrow_schema = infer_schema(&metadata).map_err(to_kernel_err)?;
    let kernel_schema = arrow_schema.to_kernel().map_err(to_kernel_err)?;
    Ok(ParquetFooter {
        schema: Arc::new(kernel_schema),
    })
}

/// Fetch a parquet file's thrift footer via two range reads
fn fetch_parquet_metadata(
    storage: &ObjectStoreStorageHandler,
    file: &FileMeta,
) -> DeltaResult<FileMetadata> {
    if file.size < FOOTER_SIZE {
        return Err(Error::Generic(format!(
            "{}: too small ({} bytes) to be a parquet file",
            file.location, file.size,
        )));
    }
    let read_range = |range: std::ops::Range<u64>| -> DeltaResult<bytes::Bytes> {
        storage
            .read_files(vec![(file.location.clone(), Some(range))])?
            .next()
            .ok_or_else(|| Error::Generic(format!("{}: empty range read", file.location)))?
    };

    let trailer = read_range((file.size - FOOTER_SIZE)..file.size)?;
    // A short body (proxy error page, truncated presigned response) would
    // otherwise index out of bounds below.
    if trailer.len() != FOOTER_SIZE as usize {
        return Err(Error::Generic(format!(
            "{}: trailer read returned {} of {FOOTER_SIZE} bytes",
            file.location,
            trailer.len(),
        )));
    }
    if trailer[4..] != PARQUET_MAGIC {
        return Err(Error::Generic(format!(
            "{}: missing PAR1 magic in trailer",
            file.location,
        )));
    }
    let footer_len = u32::from_le_bytes(trailer[..4].try_into().unwrap()) as u64;
    if FOOTER_SIZE + footer_len > file.size {
        return Err(Error::Generic(format!(
            "{}: footer length ({}) exceeds file size ({})",
            file.location, footer_len, file.size,
        )));
    }

    let footer_thrift =
        read_range((file.size - FOOTER_SIZE - footer_len)..(file.size - FOOTER_SIZE))?;
    if footer_thrift.len() as u64 != footer_len {
        return Err(Error::Generic(format!(
            "{}: footer read returned {} of {footer_len} bytes",
            file.location,
            footer_thrift.len(),
        )));
    }
    deserialize_metadata(Buffer::from_owner(footer_thrift)).map_err(to_kernel_err)
}

/// `ParquetOptions` shaped by the kernel-declared physical schema. Shared
/// across the kernel `ParquetHandler` path and the scan-driver's bulk read.
pub(crate) fn parquet_options(physical_schema: &StructType) -> DeltaResult<ParquetOptions> {
    let target_schema = physical_schema.to_polars().map_err(to_kernel_err)?;
    Ok(ParquetOptions {
        schema: Some(target_schema),
        ..Default::default()
    })
}

/// The one shared `DslBuilder::scan_parquet` construction: schema-typed
/// options over `paths`, with the synthetic row index requested and typed
/// in one place — kernel's plan contract types metadata columns LONG,
/// polars' native row index is IDX_DTYPE (u32). Callers customize the
/// remaining `args` first (file-id column).
pub(crate) fn dsl_parquet_scan(
    paths: Vec<PlRefPath>,
    physical_schema: &StructType,
    mut args: UnifiedScanArgs,
    row_index: Option<&PlSmallStr>,
) -> DeltaResult<LazyFrame> {
    if let Some(name) = row_index {
        args.row_index = Some(polars::prelude::RowIndex {
            name: name.clone(),
            offset: 0,
        });
    }
    let options = parquet_options(physical_schema)?;
    let lazy: LazyFrame = DslBuilder::scan_parquet(ScanSources::Paths(paths.into()), options, args)
        .map_err(to_kernel_err)?
        .build()
        .into();
    Ok(match row_index {
        Some(name) => lazy.with_columns([col(name.clone()).cast(PlDataType::Int64)]),
        None => lazy,
    })
}

/// The ScanParquet plan contract resolves a field carrying
/// `parquet.field.id` metadata by field ID. polars' unified scan matches by
/// name only, which would silently null-fill renamed columns — the plan
/// executor refuses instead. The classic data path keeps name matching:
/// delta writers put physical names in the files, and Id-mode tables read
/// correctly by name there today.
pub(crate) fn ensure_no_field_id_matching(schema: &StructType) -> DeltaResult<()> {
    fn walk_dtype(dt: &KernelDataType) -> DeltaResult<()> {
        match dt {
            KernelDataType::Struct(s) => ensure_no_field_id_matching(s),
            KernelDataType::Array(a) => walk_dtype(a.element_type()),
            KernelDataType::Map(m) => {
                walk_dtype(m.key_type())?;
                walk_dtype(m.value_type())
            }
            _ => Ok(()),
        }
    }
    for field in schema.fields() {
        if field
            .get_config_value(&ColumnMetadataKey::ParquetFieldId)
            .is_some()
        {
            return Err(Error::Unsupported(format!(
                "parquet read: field {} carries parquet.field.id — field-ID matching is not implemented",
                field.name
            )));
        }
        walk_dtype(&field.data_type)?;
    }
    Ok(())
}

/// `Insert` / `Ignore` policies are load-bearing for the kernel contract:
/// null-fill missing fields, drop file columns kernel didn't ask for (e.g.
/// `txn` in checkpoints written with stats-as-struct disabled).
pub(crate) fn unified_scan_args(
    cloud_opts: Option<&CloudOptions>,
    include_file_paths: Option<PlSmallStr>,
) -> UnifiedScanArgs {
    UnifiedScanArgs {
        cloud_options: cloud_opts.cloned(),
        rechunk: false,
        glob: false,
        hive_options: polars::prelude::HiveOptions::new_disabled(),
        cast_columns_policy: CastColumnsPolicy {
            missing_struct_fields: MissingColumnsPolicy::Insert,
            extra_struct_fields: ExtraColumnsPolicy::Ignore,
            // INT96 downcast
            datetime_nanoseconds_downcast: true,
            ..CastColumnsPolicy::ERROR_ON_MISMATCH
        },
        missing_columns_policy: MissingColumnsPolicy::Insert,
        extra_columns_policy: ExtraColumnsPolicy::Ignore,
        include_file_paths,
        ..Default::default()
    }
}

/// Everything a per-file scan needs that does not vary between files.
struct FileScanArgs {
    cloud_opts: Option<CloudOptions>,
    select_exprs: Vec<Expr>,
    predicate: Option<Expr>,
    read_schema: StructType,
    row_index: Option<PlSmallStr>,
    file_path: Option<PlSmallStr>,
}

type BatchIter = Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>;

fn read_batch(
    paths: Vec<(PlRefPath, String)>,
    args: FileScanArgs,
) -> DeltaResult<FileDataReadResultIterator> {
    // Contract: engines must not merge engine data across file boundaries,
    // so each file gets its own scan. Construction is deferred inside the
    // flat_map so file N+1's streaming query starts only once file N drains;
    // polars still parallelises row groups within a file.
    let iter = paths.into_iter().flat_map(move |(path, location)| {
        match file_batches(path, &location, &args) {
            Ok(batches) => batches,
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    });
    Ok(Box::new(iter))
}

fn file_batches(path: PlRefPath, location: &str, args: &FileScanArgs) -> DeltaResult<BatchIter> {
    let scan_args = unified_scan_args(args.cloud_opts.as_ref(), None);
    let lazy = dsl_parquet_scan(
        vec![path],
        &args.read_schema,
        scan_args,
        args.row_index.as_ref(),
    )?;
    let lazy = match &args.file_path {
        Some(name) => lazy.with_columns([lit(location).alias(name.clone())]),
        None => lazy,
    };
    let mut plan = lazy.select(args.select_exprs.clone());
    if let Some(pred) = &args.predicate {
        plan = plan.filter(pred.clone());
    }
    let batches = crate::engine::collect_streaming_batches(plan).map_err(to_kernel_err)?;
    Ok(Box::new(batches.map(
        |r| -> DeltaResult<Box<dyn EngineData>> {
            let mut df = r.map_err(to_kernel_err)?;
            df.rechunk_mut();
            Ok(Box::new(PolarsEngineData::new(df)))
        },
    )))
}

/// polars-io's `Path::parse` doesn't decode, so Spark-style double-encoded
/// partition keys (`letter=%252F`) would re-encode and miss storage. Decode
/// once up front. Same fix as delta-rs's DataFusion table provider.
pub(crate) fn path_for_polars_io(url: &Url) -> DeltaResult<PlRefPath> {
    let decoded = object_store::path::Path::from_url_path(url.path())
        .map_err(|e| Error::Generic(format!("invalid object store path {url}: {e}")))?;
    let s = if url.scheme() == "file" {
        if cfg!(windows) {
            format!("{decoded}")
        } else {
            format!("/{decoded}")
        }
    } else {
        let authority = url.authority();
        if authority.is_empty() {
            format!("{}:///{decoded}", url.scheme())
        } else {
            format!("{}://{authority}/{decoded}", url.scheme())
        }
    };
    Ok(PlRefPath::new(s))
}

#[cfg(test)]
mod per_file_batch_tests {
    use std::collections::HashMap;

    use delta_kernel::schema::{DataType, StructField};
    use polars::prelude::ParquetWriter;

    use super::*;

    /// The footer fast path synthesizes metadata columns without reading any
    /// data pages. Both supported specs must be built from their own kind —
    /// a catch-all arm would emit the file path under a row-index column.
    #[test]
    fn footer_fast_path_synthesizes_each_metadata_column() {
        use delta_kernel::schema::MetadataColumnSpec;

        let dir = std::env::temp_dir().join(format!("pldl-footer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut df = polars::df!("x" => [1i64, 2, 3]).unwrap();
        ParquetWriter::new(std::fs::File::create(dir.join("f.parquet")).unwrap())
            .finish(&mut df)
            .unwrap();

        let base = Url::from_directory_path(&dir).unwrap();
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(&base, std::iter::empty(), rt).unwrap());
        let handler = PolarsParquetHandler::new(storage, HashMap::new()).unwrap();

        // No data fields: the whole read schema is metadata columns.
        let schema = Arc::new(
            StructType::try_new([
                StructField::create_metadata_column("ri", MetadataColumnSpec::RowIndex),
                StructField::create_metadata_column("fp", MetadataColumnSpec::FilePath),
            ])
            .unwrap(),
        );
        let location = base.join("f.parquet").unwrap();
        let files = vec![FileMeta {
            location: location.clone(),
            last_modified: 0,
            size: std::fs::metadata(dir.join("f.parquet")).unwrap().len(),
        }];
        let batches: Vec<_> = handler
            .read_parquet_files(&files, schema, None)
            .unwrap()
            .collect::<DeltaResult<Vec<_>>>()
            .unwrap();

        let df = batches[0]
            .any_ref()
            .downcast_ref::<PolarsEngineData>()
            .unwrap()
            .dataframe();
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
        let fp = df.column("fp").unwrap().str().unwrap().get(0).unwrap();
        assert_eq!(
            fp,
            location.as_str(),
            "the path column carries the file URL"
        );
    }

    /// `FileDataReadResultIterator` contract: data arrives in file order and
    /// engines must not merge engine data across file boundaries.
    #[test]
    fn batches_do_not_span_files() {
        let dir = std::env::temp_dir().join(format!("pldl-perfile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, vals) in [("a.parquet", [1i64, 2, 3]), ("b.parquet", [4, 5, 6])] {
            let mut df = polars::df!("x" => vals).unwrap();
            ParquetWriter::new(std::fs::File::create(dir.join(name)).unwrap())
                .finish(&mut df)
                .unwrap();
        }
        let base = Url::from_directory_path(&dir).unwrap();
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(&base, std::iter::empty(), rt).unwrap());
        let handler = PolarsParquetHandler::new(storage, HashMap::new()).unwrap();

        let schema =
            Arc::new(StructType::try_new([StructField::not_null("x", DataType::LONG)]).unwrap());
        let files: Vec<FileMeta> = ["a.parquet", "b.parquet"]
            .iter()
            .map(|n| FileMeta {
                location: base.join(n).unwrap(),
                last_modified: 0,
                size: std::fs::metadata(dir.join(n)).unwrap().len(),
            })
            .collect();
        let batches: Vec<_> = handler
            .read_parquet_files(&files, schema, None)
            .unwrap()
            .collect::<DeltaResult<Vec<_>>>()
            .unwrap();
        let heights: Vec<usize> = batches.iter().map(|b| b.len()).collect();
        assert_eq!(heights, vec![3, 3], "batches must not span files");
        let xs: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.any_ref()
                    .downcast_ref::<PolarsEngineData>()
                    .unwrap()
                    .dataframe()
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

    /// ParquetHandler contract: a RowIndex metadata column carries the
    /// 0-based row position within each file (restarting per file) and a
    /// FilePath column the file's path — populated, not null-filled.
    #[test]
    fn metadata_columns_are_synthesized() {
        use delta_kernel::schema::MetadataColumnSpec;

        let dir = std::env::temp_dir().join(format!("pldl-metacol-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, vals) in [("a.parquet", vec![10i64, 20, 30]), ("b.parquet", vec![40])] {
            let mut df = polars::df!("x" => vals).unwrap();
            ParquetWriter::new(std::fs::File::create(dir.join(name)).unwrap())
                .finish(&mut df)
                .unwrap();
        }
        let base = Url::from_directory_path(&dir).unwrap();
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(&base, std::iter::empty(), rt).unwrap());
        let handler = PolarsParquetHandler::new(storage, HashMap::new()).unwrap();

        let schema = Arc::new(
            StructType::try_new([
                StructField::not_null("x", DataType::LONG),
                StructField::create_metadata_column("ridx", MetadataColumnSpec::RowIndex),
                StructField::create_metadata_column("fname", MetadataColumnSpec::FilePath),
            ])
            .unwrap(),
        );
        let files: Vec<FileMeta> = ["a.parquet", "b.parquet"]
            .iter()
            .map(|n| FileMeta {
                location: base.join(n).unwrap(),
                last_modified: 0,
                size: std::fs::metadata(dir.join(n)).unwrap().len(),
            })
            .collect();
        let batches: Vec<_> = handler
            .read_parquet_files(&files, schema, None)
            .unwrap()
            .collect::<DeltaResult<Vec<_>>>()
            .unwrap();

        let mut ridx = Vec::new();
        let mut fnames = Vec::new();
        for b in &batches {
            let df = b
                .any_ref()
                .downcast_ref::<PolarsEngineData>()
                .unwrap()
                .dataframe();
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

    /// Unsupported metadata specs must error loudly rather than come back
    /// as silently null-filled data columns.
    #[test]
    fn unsupported_metadata_spec_errors() {
        use delta_kernel::schema::MetadataColumnSpec;

        let dir = std::env::temp_dir().join(format!("pldl-metaerr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut df = polars::df!("x" => [1i64]).unwrap();
        ParquetWriter::new(std::fs::File::create(dir.join("a.parquet")).unwrap())
            .finish(&mut df)
            .unwrap();
        let base = Url::from_directory_path(&dir).unwrap();
        let rt = crate::engine::rt();
        let storage =
            Arc::new(ObjectStoreStorageHandler::new(&base, std::iter::empty(), rt).unwrap());
        let handler = PolarsParquetHandler::new(storage, HashMap::new()).unwrap();

        let schema = Arc::new(
            StructType::try_new([
                StructField::not_null("x", DataType::LONG),
                StructField::create_metadata_column("rid", MetadataColumnSpec::RowId),
            ])
            .unwrap(),
        );
        let files = vec![FileMeta {
            location: base.join("a.parquet").unwrap(),
            last_modified: 0,
            size: std::fs::metadata(dir.join("a.parquet")).unwrap().len(),
        }];
        let result = handler
            .read_parquet_files(&files, schema, None)
            .and_then(|it| it.collect::<DeltaResult<Vec<_>>>());
        let err = match result {
            Ok(_) => panic!("RowId is unsupported and must error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("metadata column"), "got: {err}");
    }
}

#[cfg(test)]
mod field_id_tests {
    use delta_kernel::schema::{DataType, MetadataValue, StructField};

    use super::*;

    #[test]
    fn field_id_metadata_is_unsupported() {
        let schema = StructType::try_new([StructField::nullable("c1", DataType::LONG)
            .with_metadata([("parquet.field.id", MetadataValue::Number(5))])])
        .unwrap();
        let err =
            ensure_no_field_id_matching(&schema).expect_err("field-id matching is unimplemented");
        assert!(err.to_string().contains("parquet.field.id"), "got: {err}");
    }

    #[test]
    fn nested_field_id_metadata_is_unsupported() {
        let inner = StructField::nullable("x", DataType::LONG)
            .with_metadata([("parquet.field.id", MetadataValue::Number(7))]);
        let schema = StructType::try_new([StructField::nullable(
            "s",
            DataType::Struct(Box::new(StructType::try_new([inner]).unwrap())),
        )])
        .unwrap();
        assert!(ensure_no_field_id_matching(&schema).is_err());
    }

    #[test]
    fn plain_schema_passes() {
        let schema = StructType::try_new([StructField::nullable("c", DataType::LONG)]).unwrap();
        assert!(ensure_no_field_id_matching(&schema).is_ok());
    }
}
