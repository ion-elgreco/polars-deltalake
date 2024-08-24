use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::{DeltaResult, Error, FileMeta, FileSlice, FileSystemClient};
use futures::StreamExt;
use object_store::{path::Path, DynObjectStore};
use url::Url;

use super::executor::TaskExecutor;
pub(crate) struct PolarsEngineFilesystemClient;

#[derive(Debug)]
pub struct ObjectStoreFileSystemClient<E: TaskExecutor> {
    inner: Arc<DynObjectStore>,
    table_root: Path,
    task_executor: Arc<E>,
    readahead: usize,
}

impl<E: TaskExecutor> ObjectStoreFileSystemClient<E> {
    pub fn new(store: Arc<DynObjectStore>, table_root: Path, task_executor: Arc<E>) -> Self {
        Self {
            inner: store,
            table_root,
            task_executor,
            readahead: 10,
        }
    }

    /// Set the maximum number of files to read in parallel.
    pub fn with_readahead(mut self, readahead: usize) -> Self {
        self.readahead = readahead;
        self
    }
}

impl<E: TaskExecutor> FileSystemClient for ObjectStoreFileSystemClient<E> {
    fn list_from(
        &self,
        path: &Url,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<FileMeta>>>> {
        let url = path.clone();
        let offset = Path::from(path.path());
        // TODO properly handle table prefix
        let prefix = self.table_root.child("_delta_log");

        let store = self.inner.clone();

        // This channel will become the iterator
        let (sender, receiver) = std::sync::mpsc::sync_channel(4_000);

        self.task_executor.spawn(async move {
            let mut stream = store.list_with_offset(Some(&prefix), &offset);

            while let Some(meta) = stream.next().await {
                match meta {
                    Ok(meta) => {
                        let mut location = url.clone();
                        location.set_path(&format!("/{}", meta.location.as_ref()));
                        sender
                            .send(Ok(FileMeta {
                                location,
                                last_modified: meta.last_modified.timestamp(),
                                size: meta.size,
                            }))
                            .ok();
                    }
                    Err(e) => {
                        sender.send(Err(e.into())).ok();
                    }
                }
            }
        });

        Ok(Box::new(receiver.into_iter()))
    }

    /// Read data specified by the start and end offset from the file.
    ///
    /// This will return the data in the same order as the provided file slices.
    ///
    /// Multiple reads may occur in parallel, depending on the configured readahead.
    /// See [`Self::with_readahead`].
    fn read_files(
        &self,
        files: Vec<FileSlice>,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<Bytes>>>> {
        let store = self.inner.clone();

        // This channel will become the output iterator.
        // Because there will already be buffering in the stream, we set the
        // buffer size to 0.
        let (sender, receiver) = std::sync::mpsc::sync_channel(0);

        self.task_executor.spawn(
            futures::stream::iter(files)
                .map(move |(url, range)| {
                    // Wasn't checking the scheme before calling to_file_path causing the url path to
                    // be eaten in a strange way. Now, if not a file scheme, just blindly convert to a path.
                    // https://docs.rs/url/latest/url/struct.Url.html#method.to_file_path has more
                    // details about why this check is necessary
                    let path = if url.scheme() == "file" {
                        let file_path = url.to_file_path().expect("Not a valid file path");
                        Path::from_absolute_path(file_path).expect("Not able to be made into Path")
                    } else {
                        Path::from(url.path())
                    };
                    let store = store.clone();
                    async move {
                        match url.scheme() {
                            "http" | "https" => {
                                // have to annotate type here or rustc can't figure it out
                                Ok::<bytes::Bytes, Error>(reqwest::get(url).await?.bytes().await?)
                            }
                            _ => {
                                if let Some(rng) = range {
                                    Ok(store.get_range(&path, rng).await?)
                                } else {
                                    let result = store.get(&path).await?;
                                    Ok(result.bytes().await?)
                                }
                            }
                        }
                    }
                })
                // We allow executing up to `readahead` futures concurrently and
                // buffer the results. This allows us to achieve async concurrency
                // within a synchronous method.
                .buffered(self.readahead)
                .for_each(move |res| {
                    sender.send(res).ok();
                    futures::future::ready(())
                }),
        );

        Ok(Box::new(receiver.into_iter()))
    }
}
