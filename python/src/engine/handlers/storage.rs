//! `delta_kernel::StorageHandler` over `object_store`. Kernel is sync and
//! `object_store` is async, so every method `block_on`s the engine's
//! shared tokio runtime. Mirrors kernel 0.23's `default-engine`
//! `ObjectStoreStorageHandler` so behaviour stays aligned.

use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::{DeltaResult, Error, FileMeta, FileSlice, StorageHandler};
use futures_util::{StreamExt, TryStreamExt, stream};
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use tokio::runtime::Runtime;
use url::Url;

/// Match kernel's `default-engine` knob — at most this many concurrent
/// `read_files` futures in flight at once.
const READ_PARALLELISM: usize = 10;

pub(crate) struct ObjectStoreStorageHandler {
    store: Arc<dyn ObjectStore>,
    base_url: Url,
    rt: &'static Runtime,
}

impl ObjectStoreStorageHandler {
    pub(crate) fn base_url(&self) -> &Url {
        &self.base_url
    }

    pub(crate) fn new(
        table_url: &Url,
        storage_options: impl IntoIterator<Item = (String, String)>,
        rt: &'static Runtime,
    ) -> DeltaResult<Self> {
        let (store, _path) =
            object_store::parse_url_opts(table_url, storage_options).map_err(|e| {
                Error::Generic(format!("failed to build object store for {table_url}: {e}"))
            })?;
        Ok(Self {
            store: Arc::from(store),
            base_url: table_url.clone(),
            rt,
        })
    }

    fn url_to_path(&self, url: &Url) -> DeltaResult<Path> {
        if url.scheme() != self.base_url.scheme() {
            return Err(Error::Generic(format!(
                "url scheme mismatch: store={} url={}",
                self.base_url.scheme(),
                url.scheme()
            )));
        }
        if url.scheme() == "file" {
            // Match kernel: convert via `to_file_path` so symlinks /
            // platform-specific path quirks are handled by `url`, then by
            // `Path::from_absolute_path` (which strips the leading prefix).
            let file_path = url
                .to_file_path()
                .map_err(|_| Error::Generic(format!("invalid file URL: {url}")))?;
            Path::from_absolute_path(file_path)
                .map_err(|e| Error::Generic(format!("invalid file path: {e}")))
        } else {
            // Percent-decode before handing to object_store: kernel emits
            // URLs with doubly-encoded Hive partition prefixes (see kernel's
            // `WriteContext` docs), so `Path::from(url.path())` would keep
            // them literal and break interop with every other Delta writer.
            Path::from_url_path(url.path()).map_err(|e| {
                Error::Generic(format!("invalid object store path {}: {e}", url.path()))
            })
        }
    }
}

impl StorageHandler for ObjectStoreStorageHandler {
    fn list_from(
        &self,
        path: &Url,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<FileMeta>>>> {
        let offset = self.url_to_path(path)?;
        // Kernel pattern: the offset is the list-after marker; the prefix is
        // the scope. For a directory URL (trailing `/`) those are the same,
        // for a file URL the prefix is the file's parent. `Path` strips
        // trailing slashes so we have to peek at the original URL.
        let prefix = if path.path().ends_with('/') {
            offset.clone()
        } else {
            let mut parts: Vec<_> = offset.parts().collect();
            if parts.pop().is_none() {
                return Err(Error::Generic(format!(
                    "offset path must not be a root directory: '{path}'"
                )));
            }
            Path::from_iter(parts)
        };

        let store = self.store.clone();
        let mut metas: Vec<object_store::ObjectMeta> = self.rt.block_on(async move {
            store
                .list_with_offset(Some(&prefix), &offset)
                .try_collect()
                .await
                .map_err(|e| Error::Generic(format!("object_store list failed: {e}")))
        })?;

        // Cloud `list` is lexicographically ordered on real GCS/S3/Azure, so
        // we only pay the materialize+sort cost where it's actually needed
        // (local fs + S3 directory buckets).
        if !supports_ordered_listing(&self.base_url) {
            metas.sort_unstable_by(|a, b| a.location.cmp(&b.location));
        }

        let base = path.clone();
        let iter = metas.into_iter().map(move |m| {
            // Preserve port / userinfo / query on the input URL — only the
            // path part swaps in to point at the listed object.
            let mut location = base.clone();
            location.set_path(&format!("/{}", m.location.as_ref()));
            Ok(FileMeta {
                location,
                last_modified: m.last_modified.timestamp_millis(),
                size: m.size,
            })
        });
        Ok(Box::new(iter))
    }

    fn read_files(
        &self,
        files: Vec<FileSlice>,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<Bytes>>>> {
        // Resolve targets up front so URL-to-Path failures surface
        // synchronously. Presigned URLs (Databricks / R2 / SigV4
        // query-string auth) bypass object_store entirely and are fetched
        // via reqwest — the SAS / signature lives in the URL itself.
        let resolved: Vec<(ReadTarget, Option<std::ops::Range<u64>>)> = files
            .into_iter()
            .map(|(url, range)| -> DeltaResult<_> {
                let target = if is_presigned(&url) {
                    ReadTarget::Presigned(url)
                } else {
                    ReadTarget::Path(self.url_to_path(&url)?)
                };
                Ok((target, range))
            })
            .collect::<DeltaResult<_>>()?;

        let store = self.store.clone();
        let payloads: Vec<DeltaResult<Bytes>> = self.rt.block_on(async move {
            stream::iter(resolved)
                .map(move |(target, range)| {
                    let store = store.clone();
                    async move {
                        match target {
                            ReadTarget::Path(path) => match range {
                                Some(r) => store
                                    .get_range(&path, r)
                                    .await
                                    .map_err(|e| map_get_err(&path, e)),
                                None => match store.get(&path).await {
                                    Ok(g) => g.bytes().await.map_err(|e| {
                                        Error::Generic(format!("object_store bytes failed: {e}"))
                                    }),
                                    Err(e) => Err(map_get_err(&path, e)),
                                },
                            },
                            ReadTarget::Presigned(url) => fetch_presigned(url, range).await,
                        }
                    }
                })
                .buffered(READ_PARALLELISM)
                .collect()
                .await
        });

        Ok(Box::new(payloads.into_iter()))
    }

    fn copy_atomic(&self, src: &Url, dest: &Url) -> DeltaResult<()> {
        let src_path = self.url_to_path(src)?;
        let dest_path = self.url_to_path(dest)?;
        let store = self.store.clone();
        self.rt.block_on(async move {
            store
                .copy_if_not_exists(&src_path, &dest_path)
                .await
                .map_err(|e| match e {
                    object_store::Error::AlreadyExists { .. } => {
                        Error::FileAlreadyExists(dest_path.to_string())
                    }
                    other => {
                        Error::Generic(format!("object_store copy_if_not_exists failed: {other}"))
                    }
                })
        })
    }

    fn put(&self, _path: &Url, _data: Bytes, _overwrite: bool) -> DeltaResult<()> {
        Err(Error::Unsupported(
            "polars-deltalake is read-only; write support is not yet implemented".into(),
        ))
    }

    fn delete(&self, path: &Url) -> DeltaResult<()> {
        let p = self.url_to_path(path)?;
        let store = self.store.clone();
        self.rt.block_on(async move {
            match store.delete(&p).await {
                // The trait documents delete as idempotent.
                Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
                Err(other) => Err(Error::Generic(format!(
                    "object_store delete failed: {other}"
                ))),
            }
        })
    }

    fn head(&self, path: &Url) -> DeltaResult<FileMeta> {
        let p = self.url_to_path(path)?;
        let store = self.store.clone();
        let meta = self.rt.block_on(async move {
            store.head(&p).await.map_err(|e| match e {
                object_store::Error::NotFound { .. } => Error::FileNotFound(p.to_string()),
                other => Error::Generic(format!("object_store head failed: {other}")),
            })
        })?;

        Ok(FileMeta {
            location: path.clone(),
            last_modified: meta.last_modified.timestamp_millis(),
            size: meta.size,
        })
    }
}

/// Map object_store read errors onto kernel errors. NotFound is special-cased
/// so the kernel's "is `_last_checkpoint` present?" / commit-conflict checks
/// see the right variant instead of a generic catch-all.
fn map_get_err(path: &Path, e: object_store::Error) -> Error {
    match e {
        object_store::Error::NotFound { .. } => Error::FileNotFound(path.to_string()),
        other => Error::Generic(format!("object_store get failed for {path}: {other}")),
    }
}

/// What a single `read_files` slice resolves to. Normal URLs become an
/// `object_store::Path`; presigned URLs stay as the original `Url` so the
/// reqwest fetcher can hit them as-is.
enum ReadTarget {
    Path(Path),
    Presigned(Url),
}

/// Kernel-aligned: does this URL carry query-string auth (SigV4 / SAS /
/// Databricks)? If so, we have to fetch via HTTP directly — `object_store`
/// strips query params and would re-sign with whatever the credential
/// provider yields, producing a 403.
fn is_presigned(url: &Url) -> bool {
    // Same key set kernel checks — covers AWS SigV4, Cloudflare R2, Azure
    // SAS, Google Cloud Storage, Alibaba OSS, and Databricks UC.
    const PRESIGNED_KEYS: &[&str] = &[
        "X-Amz-Signature",
        "sp",
        "X-Goog-Credential",
        "X-OSS-Credential",
        "X-Databricks-Signature",
    ];
    matches!(url.scheme(), "http" | "https")
        && url
            .query_pairs()
            .any(|(k, _)| PRESIGNED_KEYS.iter().any(|p| k.eq_ignore_ascii_case(p)))
}

/// Fetch a presigned URL via reqwest. Range requests are forwarded as
/// `Range: bytes=start-end` so the parquet-footer suffix path stays cheap
/// when the kernel hands us presigned parquet URLs.
async fn fetch_presigned(url: Url, range: Option<std::ops::Range<u64>>) -> DeltaResult<Bytes> {
    let mut req = reqwest::Client::new().get(url.clone());
    if let Some(r) = range {
        // HTTP byte ranges are inclusive on both ends; kernel `Range<u64>` is
        // half-open, so subtract one from the end.
        req = req.header(
            reqwest::header::RANGE,
            format!("bytes={}-{}", r.start, r.end.saturating_sub(1)),
        );
    }
    let resp = req
        .send()
        .await
        .map_err(|e| Error::Generic(format!("presigned GET failed for {url}: {e}")))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(Error::FileNotFound(url.to_string()));
    }
    let resp = resp
        .error_for_status()
        .map_err(|e| Error::Generic(format!("presigned GET {url} returned {e}")))?;
    resp.bytes()
        .await
        .map_err(|e| Error::Generic(format!("presigned body read failed for {url}: {e}")))
}

/// Kernel-aligned: `true` iff `object_store::list` is guaranteed to return
/// lexicographically-ordered results for this URL. False for local fs
/// (`LocalFileSystem` lists in filesystem order) and S3 directory buckets
/// (`*--x-s3`, `*-xa-s3`); true for general-purpose S3 / GCS / Azure.
fn supports_ordered_listing(url: &Url) -> bool {
    !((url.scheme() == "file")
        || url.domain().map(|d| d.contains("--x-s3")).unwrap_or(false)
        || url.domain().map(|d| d.contains("-xa-s3")).unwrap_or(false))
}

#[cfg(test)]
mod delete_tests {
    use super::*;

    /// The trait documents delete as idempotent: a missing path is `Ok`.
    #[test]
    fn delete_missing_path_is_ok() {
        use delta_kernel::StorageHandler;

        let dir = std::env::temp_dir().join(format!("pldl-del-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = Url::from_directory_path(&dir).unwrap();
        let storage =
            ObjectStoreStorageHandler::new(&base, std::iter::empty(), crate::engine::rt()).unwrap();
        let missing = base.join("nope.json").unwrap();
        storage.delete(&missing).expect("idempotent delete");
    }
}
