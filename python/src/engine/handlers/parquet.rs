//! `delta_kernel::ParquetHandler` over polars-io. Every `read_parquet_files`
//! call hands the whole file batch to a single `DslBuilder::scan_parquet`
//! plan with the kernel physical schema attached and polars-io's
//! `missing_struct_fields=Insert` / `extra_columns=Ignore` policies set, so
//! polars-io's multi-file scan does cross-file / row-group / column
//! parallelism in its own scheduler and reads come back already shaped to
//! the kernel contract.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::{ColumnMetadataKey, DataType as KernelDataType, SchemaRef, StructType};
use delta_kernel::{
    DeltaResult, Error, FileDataReadResultIterator, FileMeta, ParquetFooter, ParquetHandler,
    StorageHandler,
};
use polars::io::cloud::CloudOptions;
use polars::io::parquet::read::{ParquetOptions, infer_schema};
use polars::lazy::frame::LazyFrame;
use polars::prelude::{DataFrame, Expr};
use polars_buffer::Buffer;
use polars_parquet::parquet::metadata::FileMetadata;
use polars_parquet::parquet::{FOOTER_SIZE, PARQUET_MAGIC, read::deserialize_metadata};
use polars_plan::dsl::{
    CastColumnsPolicy, DslBuilder, Engine as PolarsEngineMode, ExtraColumnsPolicy,
    MissingColumnsPolicy, ScanSources, UnifiedScanArgs,
};
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use polars_utils::pl_str::PlSmallStr;
use url::Url;

use crate::engine::{COLLECT_CHUNK_ROWS, PolarsEngineData};
use crate::errors::to_kernel_err;
use crate::translation::schema::{ArrowSchemaExt, KernelSchemaExt};

use super::storage::ObjectStoreStorageHandler;

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
        // Empty projection (kernel needs only the row count so its
        // logical-transform can emit literal columns — e.g. CDF metadata,
        // all-partition projections): read `num_rows` from each footer and
        // skip the data scan entirely
        if physical_schema.fields().next().is_none() {
            let total: usize = files
                .iter()
                .map(|f| fetch_parquet_metadata(self.storage.as_ref(), f).map(|m| m.num_rows))
                .sum::<DeltaResult<usize>>()?;
            let df = DataFrame::empty_with_height(total);
            let result: DeltaResult<Box<dyn EngineData>> = Ok(Box::new(PolarsEngineData::new(df)));
            return Ok(Box::new(std::iter::once(result)));
        }

        // Translation failure is non-fatal: kernel already pruned files via
        // stats, so we just skip row-level pushdown and let polars filter.
        let polars_predicate: Option<Expr> = predicate.as_ref().and_then(|p| {
            crate::translation::from_kernel::translate_predicate(p.as_ref(), None).ok()
        });

        let select_exprs = crate::scan::select_exprs_for_schema(physical_schema.as_ref());

        let paths: Vec<PlRefPath> = files
            .iter()
            .map(|f| path_for_polars_io(&f.location))
            .collect::<DeltaResult<_>>()?;

        read_batch(
            paths,
            self.cloud_opts.as_ref(),
            &select_exprs,
            polars_predicate.as_ref(),
            physical_schema.as_ref(),
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
/// options over `paths`. Callers customize `args` first (row index,
/// file-id column).
pub(crate) fn dsl_parquet_scan(
    paths: Vec<PlRefPath>,
    physical_schema: &StructType,
    args: UnifiedScanArgs,
) -> DeltaResult<LazyFrame> {
    let options = parquet_options(physical_schema)?;
    let lazy: LazyFrame = DslBuilder::scan_parquet(ScanSources::Paths(paths.into()), options, args)
        .map_err(to_kernel_err)?
        .build()
        .into();
    Ok(lazy)
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

fn read_batch(
    paths: Vec<PlRefPath>,
    cloud_opts: Option<&CloudOptions>,
    select_exprs: &[Expr],
    predicate: Option<&Expr>,
    physical_schema: &StructType,
) -> DeltaResult<FileDataReadResultIterator> {
    let lazy = dsl_parquet_scan(paths, physical_schema, unified_scan_args(cloud_opts, None))?;

    let mut plan = lazy.select(select_exprs);
    if let Some(pred) = predicate {
        plan = plan.filter(pred.clone());
    }
    let chunk_size = std::num::NonZeroUsize::new(COLLECT_CHUNK_ROWS);
    let batches = plan
        .collect_batches(PolarsEngineMode::Streaming, true, chunk_size, false)
        .map_err(to_kernel_err)?;
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
