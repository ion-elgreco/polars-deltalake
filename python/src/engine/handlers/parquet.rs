//! polars-io parquet helpers behind the plan executor's `ScanParquet` /
//! `DynamicScan` and the scan driver's bulk read: the one
//! `DslBuilder::scan_parquet` construction with the kernel physical schema
//! attached and polars-io's `missing_struct_fields=Insert` /
//! `extra_columns=Ignore` policies set, the suffix-range footer reader, and
//! the RowIndex / FilePath metadata columns synthesized per file, never read
//! from the data pages.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::schema::{
    ColumnMetadataKey, DataType as KernelDataType, MetadataColumnSpec, StructField, StructType,
};
use delta_kernel::{FileMeta, KernelError, KernelResult, ParquetFooter, StorageHandler};
use polars::io::cloud::CloudOptions;
use polars::io::parquet::read::{ParquetOptions, infer_schema};
use polars::lazy::frame::LazyFrame;
use polars::prelude::{
    ChunkFull, Column, DataFrame, DataType as PlDataType, IntoColumn, IntoSeries, NamedFrom,
    Series, StringChunked, col,
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
/// requested synthetic metadata columns. RowIndex is the 0-based position
/// within each file, FilePath its URL; any other spec errors loudly — a
/// null-fill would silently misreport it.
pub(crate) fn split_metadata_columns(
    schema: &StructType,
    context: &str,
) -> KernelResult<(StructType, MetadataColumns)> {
    let mut row_index: Option<PlSmallStr> = None;
    let mut file_path: Option<PlSmallStr> = None;
    let mut cols: Vec<(PlSmallStr, SyntheticColumn)> = Vec::new();
    let mut read_fields: Vec<StructField> = Vec::new();
    for f in schema.fields() {
        let Some(spec) = f.get_metadata_column_spec() else {
            read_fields.push(f.clone());
            continue;
        };
        let name = PlSmallStr::from_str(f.name.as_str());
        let (kind, slot) = match spec {
            MetadataColumnSpec::RowIndex => (SyntheticColumn::RowIndex, &mut row_index),
            MetadataColumnSpec::FilePath => (SyntheticColumn::FilePath, &mut file_path),
            other => {
                return Err(KernelError::Unsupported(format!(
                    "{context}: metadata column {other:?} is not supported"
                )));
            }
        };
        if slot.replace(name.clone()).is_some() {
            return Err(KernelError::Unsupported(format!(
                "{context}: more than one {spec:?} metadata column"
            )));
        }
        cols.push((name, kind));
    }
    Ok((StructType::try_new(read_fields)?, MetadataColumns { cols }))
}

/// `None` for `file://`; the executor holds the result for every scan it
/// builds.
pub(crate) fn cloud_options_for(
    url: &Url,
    storage_options: HashMap<String, String>,
) -> KernelResult<Option<CloudOptions>> {
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
            return Err(KernelError::Unsupported(format!(
                "cloud options: unknown URL scheme '{other}'"
            )));
        }
    };
    CloudOptions::from_untyped_config(Some(scheme), storage_options)
        .map(Some)
        .map_err(to_kernel_err)
}

/// A frame of `rows` rows holding only the synthesized metadata columns, for
/// a read that needs nothing from the data pages: polars would size the
/// frame from zero read columns, so the footer's row count sets the height
/// instead.
pub(crate) fn metadata_only_frame(
    meta_cols: &MetadataColumns,
    location: &Url,
    rows: usize,
) -> KernelResult<DataFrame> {
    let columns: Vec<Column> = meta_cols
        .cols
        .iter()
        .map(|(name, kind)| match kind {
            SyntheticColumn::RowIndex => {
                Series::new(name.clone(), (0..rows as i64).collect::<Vec<i64>>()).into_column()
            }
            SyntheticColumn::FilePath => StringChunked::full(name.clone(), location.as_str(), rows)
                .into_series()
                .into_column(),
        })
        .collect();
    if columns.is_empty() {
        Ok(DataFrame::empty_with_height(rows))
    } else {
        DataFrame::new(rows, columns).map_err(to_kernel_err)
    }
}

/// Footer → kernel schema for the plan executor's
/// `IoOperation::ParquetFooter` arm.
pub(crate) fn kernel_parquet_footer(
    storage: &ObjectStoreStorageHandler,
    file: &FileMeta,
) -> KernelResult<ParquetFooter> {
    let metadata = fetch_parquet_metadata(storage, file)?;
    let arrow_schema = infer_schema(&metadata).map_err(to_kernel_err)?;
    let kernel_schema = arrow_schema.to_kernel().map_err(to_kernel_err)?;
    Ok(ParquetFooter {
        schema: Arc::new(kernel_schema),
    })
}

/// Fetch a parquet file's thrift footer via two range reads
pub(crate) fn fetch_parquet_metadata(
    storage: &dyn StorageHandler,
    file: &FileMeta,
) -> KernelResult<FileMetadata> {
    if file.size < FOOTER_SIZE {
        return Err(KernelError::Generic(format!(
            "{}: too small ({} bytes) to be a parquet file",
            file.location, file.size,
        )));
    }
    let read_range = |range: std::ops::Range<u64>| -> KernelResult<bytes::Bytes> {
        storage
            .read_files(vec![(file.location.clone(), Some(range))])?
            .next()
            .ok_or_else(|| KernelError::Generic(format!("{}: empty range read", file.location)))?
    };

    let trailer = read_range((file.size - FOOTER_SIZE)..file.size)?;
    // A short body (proxy error page, truncated presigned response) would
    // otherwise index out of bounds below.
    if trailer.len() != FOOTER_SIZE as usize {
        return Err(KernelError::Generic(format!(
            "{}: trailer read returned {} of {FOOTER_SIZE} bytes",
            file.location,
            trailer.len(),
        )));
    }
    if trailer[4..] != PARQUET_MAGIC {
        return Err(KernelError::Generic(format!(
            "{}: missing PAR1 magic in trailer",
            file.location,
        )));
    }
    let footer_len = u32::from_le_bytes(trailer[..4].try_into().unwrap()) as u64;
    if FOOTER_SIZE + footer_len > file.size {
        return Err(KernelError::Generic(format!(
            "{}: footer length ({}) exceeds file size ({})",
            file.location, footer_len, file.size,
        )));
    }

    let footer_thrift =
        read_range((file.size - FOOTER_SIZE - footer_len)..(file.size - FOOTER_SIZE))?;
    if footer_thrift.len() as u64 != footer_len {
        return Err(KernelError::Generic(format!(
            "{}: footer read returned {} of {footer_len} bytes",
            file.location,
            footer_thrift.len(),
        )));
    }
    deserialize_metadata(Buffer::from_owner(footer_thrift)).map_err(to_kernel_err)
}

/// `ParquetOptions` shaped by the kernel-declared physical schema. Shared
/// across the plan executor's scans and the scan-driver's bulk read.
pub(crate) fn parquet_options(physical_schema: &StructType) -> KernelResult<ParquetOptions> {
    let target_schema = physical_schema.to_polars().map_err(to_kernel_err)?;
    Ok(ParquetOptions {
        schema: Some(target_schema),
        ..Default::default()
    })
}

/// The one shared `DslBuilder::scan_parquet` construction: schema-typed
/// options over `paths`, with the synthetic row index requested in one
/// place, in polars' native IDX_DTYPE. Callers customize the remaining
/// `args` first (file-id column).
pub(crate) fn dsl_parquet_scan(
    paths: Vec<PlRefPath>,
    physical_schema: &StructType,
    mut args: UnifiedScanArgs,
    row_index: Option<&PlSmallStr>,
) -> KernelResult<LazyFrame> {
    if let Some(name) = row_index {
        args.row_index = Some(polars::prelude::RowIndex {
            name: name.clone(),
            offset: 0,
        });
    }
    let options = parquet_options(physical_schema)?;
    Ok(
        DslBuilder::scan_parquet(ScanSources::Paths(paths.into()), options, args)
            .map_err(to_kernel_err)?
            .build()
            .into(),
    )
}

/// Kernel's plan contract types metadata columns LONG; polars' row index
/// is IDX_DTYPE (u32). Every kernel-facing arm casts its row index here.
pub(crate) fn row_index_as_long(lf: LazyFrame, name: &PlSmallStr) -> LazyFrame {
    lf.with_columns([col(name.clone()).cast(PlDataType::Int64)])
}

/// The ScanParquet plan contract resolves a field carrying
/// `parquet.field.id` metadata by field ID. polars' unified scan matches by
/// name only, which would silently null-fill renamed columns — the plan
/// executor refuses instead. The scan driver's data path keeps name
/// matching: delta writers put physical names in the files, and Id-mode
/// tables read correctly by name there today.
pub(crate) fn ensure_no_field_id_matching(schema: &StructType) -> KernelResult<()> {
    fn walk_dtype(dt: &KernelDataType) -> KernelResult<()> {
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
            return Err(KernelError::Unsupported(format!(
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
            // INT96 lands as nanoseconds; TIMESTAMP_MILLIS files predate
            // the microsecond Delta type.
            datetime_nanoseconds_downcast: true,
            datetime_milliseconds_upcast: true,
            // Type widening: files written before an ALTER COLUMN keep the
            // narrower type. Decimal and date widenings are not covered.
            integer_upcast: true,
            integer_to_float_cast: true,
            float_upcast: true,
            ..CastColumnsPolicy::ERROR_ON_MISMATCH
        },
        missing_columns_policy: MissingColumnsPolicy::Insert,
        extra_columns_policy: ExtraColumnsPolicy::Ignore,
        include_file_paths,
        ..Default::default()
    }
}

/// polars-io's `Path::parse` doesn't decode, so Spark-style double-encoded
/// partition keys (`letter=%252F`) would re-encode and miss storage. Decode
/// once up front. Same fix as delta-rs's DataFusion table provider.
pub(crate) fn path_for_polars_io(url: &Url) -> KernelResult<PlRefPath> {
    let decoded = object_store::path::Path::from_url_path(url.path())
        .map_err(|e| KernelError::Generic(format!("invalid object store path {url}: {e}")))?;
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
