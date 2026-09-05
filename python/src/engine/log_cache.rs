//! Process-wide cache of parsed Delta log JSON files.
//!
//! A commit file is never rewritten once it exists, so location, size and
//! modification time identify its content. Kernel replays the log twice per
//! scan (protocol/metadata for the snapshot, add/remove for the file list)
//! and again on every later scan of the same table; each replay after the
//! first is then a lookup instead of a fetch and parse.

use std::sync::{Arc, LazyLock};

use delta_kernel::FileMeta;
use foyer_memory::{Cache, CacheBuilder};
use polars::prelude::DataFrame;

const CAPACITY_BYTES: usize = 64 << 20;

/// One or more commit files parsed as a single NDJSON document, in file
/// order. `rows_per_file` recovers per-file columns after the parse.
pub(crate) struct ParsedLog {
    pub(crate) df: DataFrame,
    pub(crate) rows_per_file: Vec<usize>,
}

impl ParsedLog {
    pub(crate) fn single(df: DataFrame) -> Self {
        let rows_per_file = vec![df.height()];
        Self { df, rows_per_file }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct FileKey {
    location: String,
    size: u64,
    last_modified: i64,
}

type Key = Vec<FileKey>;

fn key_of(files: &[&FileMeta]) -> Key {
    files
        .iter()
        .map(|meta| FileKey {
            location: meta.location.to_string(),
            size: meta.size,
            last_modified: meta.last_modified,
        })
        .collect()
}

pub(crate) struct LogFileCache {
    cache: Cache<Key, Arc<ParsedLog>>,
    capacity: usize,
}

impl LogFileCache {
    pub(crate) fn global() -> &'static LogFileCache {
        static CACHE: LazyLock<LogFileCache> = LazyLock::new(|| LogFileCache::new(CAPACITY_BYTES));
        &CACHE
    }

    /// Weighted by the frame's estimated size, so `capacity` is in bytes.
    fn new(capacity: usize) -> Self {
        Self {
            cache: CacheBuilder::new(capacity)
                .with_weighter(|_: &Key, parsed: &Arc<ParsedLog>| parsed.df.estimated_size().max(1))
                .build(),
            capacity,
        }
    }

    pub(crate) fn get(&self, files: &[&FileMeta]) -> Option<Arc<ParsedLog>> {
        self.cache.get(&key_of(files)).map(|e| e.value().clone())
    }

    /// Stores the parse and returns it shared. A parse above the whole
    /// capacity is returned without being kept; foyer would otherwise hold
    /// it past the budget.
    pub(crate) fn insert(&self, files: &[&FileMeta], parsed: ParsedLog) -> Arc<ParsedLog> {
        let parsed = Arc::new(parsed);
        if parsed.df.estimated_size() <= self.capacity {
            self.cache.insert(key_of(files), parsed.clone());
        }
        parsed
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::{NamedFrom, Series};
    use url::Url;

    use super::*;

    fn meta(name: &str, size: u64) -> FileMeta {
        FileMeta {
            location: Url::parse(&format!("memory:///_delta_log/{name}.json")).unwrap(),
            last_modified: 1,
            size,
        }
    }

    fn frame(rows: usize) -> ParsedLog {
        ParsedLog::single(
            DataFrame::new(rows, vec![Series::new("a".into(), vec![1i64; rows]).into()]).unwrap(),
        )
    }

    #[test]
    fn same_files_hit_and_a_changed_file_misses() {
        let cache = LogFileCache::new(1 << 20);
        let (a, b) = (meta("1", 10), meta("2", 10));
        assert!(cache.get(&[&a, &b]).is_none());
        let stored = cache.insert(&[&a, &b], frame(3));
        assert!(Arc::ptr_eq(&cache.get(&[&a, &b]).unwrap(), &stored));
        assert!(
            cache.get(&[&a]).is_none(),
            "a different file list is a different key"
        );
        let b2 = meta("2", 11);
        assert!(
            cache.get(&[&a, &b2]).is_none(),
            "a different size is a different file"
        );
    }

    /// The byte budget holds: three frames of one unit into a budget of two
    /// and a half leave at most two resident.
    #[test]
    fn stays_within_capacity() {
        let one = frame(1000).df.estimated_size();
        let cache = LogFileCache::new(one * 2 + one / 2);
        let metas = [meta("1", 1), meta("2", 1), meta("3", 1)];
        for m in &metas {
            cache.insert(&[m], frame(1000));
        }
        let resident = metas.iter().filter(|m| cache.get(&[m]).is_some()).count();
        assert!(resident <= 2, "{resident} resident");
    }

    #[test]
    fn oversized_parse_is_returned_but_not_kept() {
        let cache = LogFileCache::new(16);
        let parsed = cache.insert(&[&meta("1", 1)], frame(1000));
        assert_eq!(parsed.df.height(), 1000);
        assert!(cache.get(&[&meta("1", 1)]).is_none());
    }
}
