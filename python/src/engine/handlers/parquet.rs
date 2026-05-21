//! `delta_kernel::ParquetHandler` over polars-io. Files stream through
//! `DslBuilder::scan_parquet` with the kernel physical schema attached and
//! polars-io's `missing_struct_fields=Insert` / `extra_columns=Ignore`
//! policies set, so reads come back already shaped to the kernel contract.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::engine_data::EngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::SchemaRef;
use delta_kernel::{
    DeltaResult, Error, FileDataReadResultIterator, FileMeta, ParquetFooter, ParquetHandler,
    StorageHandler,
};
use polars::io::cloud::CloudOptions;
use polars::io::parquet::read::{ParquetOptions, infer_schema};
use polars::lazy::frame::LazyFrame;
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
    /// Pre-built once per table; `None` for `file://`. Cloned per scan.
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
        // Translation failure is non-fatal: kernel already pruned files via
        // stats, so we just skip row-level pushdown and let polars filter.
        let polars_predicate: Option<polars::prelude::Expr> = predicate.as_ref().and_then(|p| {
            crate::translation::from_kernel::translate_predicate(p.as_ref(), None).ok()
        });

        let select_exprs: Vec<polars::prelude::Expr> = physical_schema
            .fields()
            .map(|f| polars::prelude::col(PlSmallStr::from_str(f.name.as_str())))
            .collect();

        let results: Vec<DeltaResult<Box<dyn EngineData>>> = files
            .iter()
            .map(|file| {
                self.read_one(
                    file,
                    &select_exprs,
                    polars_predicate.as_ref(),
                    physical_schema.as_ref(),
                )
            })
            .collect();

        Ok(Box::new(results.into_iter()))
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
        // Two range reads — the 8-byte trailer (footer length + PAR1 magic),
        // then the exact footer thrift bytes. Two RTTs but minimal bandwidth;
        // kernel calls this off the hot scan path.
        if file.size < FOOTER_SIZE {
            return Err(Error::Generic(format!(
                "{}: too small ({} bytes) to be a parquet file",
                file.location, file.size,
            )));
        }
        let read_range = |range: std::ops::Range<u64>| -> DeltaResult<bytes::Bytes> {
            self.storage
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
        let metadata =
            deserialize_metadata(footer_thrift.as_ref(), max_size).map_err(to_kernel_err)?;
        let arrow_schema = infer_schema(&metadata).map_err(to_kernel_err)?;
        let kernel_schema = arrow_schema.to_kernel().map_err(to_kernel_err)?;
        Ok(ParquetFooter {
            schema: Arc::new(kernel_schema),
        })
    }
}

impl PolarsParquetHandler {
    fn read_one(
        &self,
        file: &FileMeta,
        select_exprs: &[polars::prelude::Expr],
        predicate: Option<&polars::prelude::Expr>,
        physical_schema: &delta_kernel::schema::StructType,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let target_schema = physical_schema.to_polars().map_err(to_kernel_err)?;
        let parquet_options = ParquetOptions {
            schema: Some(target_schema),
            ..Default::default()
        };
        // `Insert` fills missing (top-level or nested) fields with nulls;
        // `Ignore` drops file columns kernel didn't ask for (e.g. `txn` in
        // checkpoints written with stats-as-struct disabled).
        let unified_scan_args = UnifiedScanArgs {
            cloud_options: self.cloud_opts.clone(),
            // visit_rows requires a single chunk per frame.
            rechunk: true,
            glob: false,
            hive_options: polars::prelude::HiveOptions::new_disabled(),
            cast_columns_policy: CastColumnsPolicy {
                missing_struct_fields: MissingColumnsPolicy::Insert,
                extra_struct_fields: ExtraColumnsPolicy::Ignore,
                ..CastColumnsPolicy::ERROR_ON_MISMATCH
            },
            missing_columns_policy: MissingColumnsPolicy::Insert,
            extra_columns_policy: ExtraColumnsPolicy::Ignore,
            ..Default::default()
        };

        // polars-io builds object-store paths via `Path::parse` (no decoding),
        // so Spark's double-encoded partition prefixes (`letter=%252F` in the
        // log) would re-encode and miss storage. Decode once up front — same
        // fix delta-rs applies in its DataFusion table provider.
        let path = path_for_polars_io(&file.location)?;
        let sources = ScanSources::Paths(vec![path].into());
        let lazy: LazyFrame = DslBuilder::scan_parquet(sources, parquet_options, unified_scan_args)
            .map_err(to_kernel_err)?
            .build()
            .into();

        // Bind our runtime so polars' async tasks reuse it instead of spinning
        // up a fresh per-call executor.
        let _enter = self.rt.enter();
        let mut plan = lazy.select(select_exprs);
        if let Some(pred) = predicate {
            plan = plan.filter(pred.clone());
        }
        let mut df = plan.collect().map_err(to_kernel_err)?;
        df.rechunk_mut();
        Ok(Box::new(PolarsEngineData::new(df)))
    }
}

/// Decode the URL once and rebuild a polars-io-friendly path: bare OS path
/// for `file://`, scheme-qualified with the decoded storage key otherwise.
fn path_for_polars_io(url: &Url) -> DeltaResult<PlRefPath> {
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
