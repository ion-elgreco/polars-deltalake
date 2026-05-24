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
use delta_kernel::schema::{SchemaRef, StructType};
use delta_kernel::{
    DeltaResult, Error, FileDataReadResultIterator, FileMeta, ParquetFooter, ParquetHandler,
    StorageHandler,
};
use polars::io::cloud::CloudOptions;
use polars::io::parquet::read::{ParquetOptions, infer_schema};
use polars::lazy::frame::LazyFrame;
use polars::prelude::{DataFrame, Expr};
use polars_parquet::parquet::metadata::FileMetadata;
use polars_parquet::parquet::{FOOTER_SIZE, PARQUET_MAGIC, read::deserialize_metadata};
use polars_plan::dsl::{
    CastColumnsPolicy, DslBuilder, ExtraColumnsPolicy, MissingColumnsPolicy, ScanSources,
    UnifiedScanArgs,
};
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use polars_utils::pl_str::PlSmallStr;
use tokio::runtime::Runtime;
use url::Url;

use crate::engine::PolarsEngineData;
use crate::errors::to_kernel_err;
use crate::translation::schema::{ArrowSchemaExt, KernelSchemaExt};

use super::storage::ObjectStoreStorageHandler;

pub(crate) struct PolarsParquetHandler {
    storage: Arc<ObjectStoreStorageHandler>,
    /// Pre-built once per table; `None` for `file://`. Cloned once per
    /// `read_parquet_files` call into `UnifiedScanArgs`.
    cloud_opts: Option<CloudOptions>,
    rt: &'static Runtime,
}

impl PolarsParquetHandler {
    pub(crate) fn new(
        storage: Arc<ObjectStoreStorageHandler>,
        storage_options: HashMap<String, String>,
        rt: &'static Runtime,
    ) -> DeltaResult<Self> {
        let cloud_opts = cloud_options_for(storage.base_url(), storage_options)?;
        Ok(Self {
            storage,
            cloud_opts,
            rt,
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

        let result = read_batch(
            paths,
            self.cloud_opts.as_ref(),
            &select_exprs,
            polars_predicate.as_ref(),
            physical_schema.as_ref(),
            self.rt,
        );
        Ok(Box::new(std::iter::once(result)))
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
        let metadata = fetch_parquet_metadata(self.storage.as_ref(), file)?;
        let arrow_schema = infer_schema(&metadata).map_err(to_kernel_err)?;
        let kernel_schema = arrow_schema.to_kernel().map_err(to_kernel_err)?;
        Ok(ParquetFooter {
            schema: Arc::new(kernel_schema),
        })
    }
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
    let max_size = footer_thrift.len() * 2 + 1024;
    deserialize_metadata(footer_thrift.as_ref(), max_size).map_err(to_kernel_err)
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
    rt: &'static Runtime,
) -> DeltaResult<Box<dyn EngineData>> {
    let parquet_options = parquet_options(physical_schema)?;
    let unified_scan_args = unified_scan_args(cloud_opts, None);

    let sources = ScanSources::Paths(paths.into());
    let lazy: LazyFrame = DslBuilder::scan_parquet(sources, parquet_options, unified_scan_args)
        .map_err(to_kernel_err)?
        .build()
        .into();

    // Bind our runtime so polars' async tasks reuse it instead of spinning
    // up a fresh per-call executor.
    let _enter = rt.enter();
    let mut plan = lazy.select(select_exprs);
    if let Some(pred) = predicate {
        plan = plan.filter(pred.clone());
    }
    let mut df = plan.collect().map_err(to_kernel_err)?;
    df.rechunk_mut();
    Ok(Box::new(PolarsEngineData::new(df)))
}

/// polars-io's `Path::parse` doesn't decode, so Spark-style double-encoded
/// partition keys (`letter=%252F`) would re-encode and miss storage. Decode
/// once up front. Same fix as delta-rs's DataFusion table provider.
pub(crate) fn path_for_polars_io(url: &Url) -> DeltaResult<PlRefPath> {
    let decoded = object_store::path::Path::from_url_path(url.path())
        .map_err(|e| Error::Generic(format!("invalid object store path {url}: {e}")))?;
    let s = if url.scheme() == "file" {
        // polars-io treats the post-`file:` remainder as an OS path. Plain
        // absolute path avoids `to_file_path`'s Windows-drive-letter quirks.
        format!("/{decoded}")
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
