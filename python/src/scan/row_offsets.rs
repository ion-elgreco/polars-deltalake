//! Places each DV file in the scan-wide `ROW_INDEX_COL` (see its doc).

use std::ops::Range;

use delta_kernel::StorageHandler;
use polars::prelude::IdxSize;
use rayon::prelude::*;

use crate::scan::plan::{LazyDv, ScanFileMeta};

/// The physical row range of every DV file, aligned with `files`. Every
/// file up to the last DV file needs a count: earlier files place a DV
/// file, its own count bounds the index → file mapping. `row_count` fills
/// in files whose add action carries no `numRecords`.
pub(crate) fn place_dvs(
    files: &[ScanFileMeta],
    row_count: impl Fn(&ScanFileMeta) -> anyhow::Result<u64> + Sync,
) -> anyhow::Result<Vec<Option<Range<u64>>>> {
    let mut spans = vec![None; files.len()];
    let Some(last_dv) = files.iter().rposition(|f| f.rewrite.dv.is_some()) else {
        return Ok(spans);
    };
    // Footer reads are round trips; run them side by side.
    let counts: Vec<u64> = files[..=last_dv]
        .par_iter()
        .map(|f| match f.num_records {
            Some(n) => Ok(n),
            None => row_count(f),
        })
        .collect::<anyhow::Result<_>>()?;
    let mut offset: u64 = 0;
    for ((file, span), num_rows) in files.iter().zip(spans.iter_mut()).zip(counts) {
        if file.rewrite.dv.is_some() {
            *span = Some(offset..offset + num_rows);
        }
        offset = offset.saturating_add(num_rows);
    }
    // polars' row index is `IdxSize`; a scan past it has no valid index.
    if offset > IdxSize::MAX as u64 {
        anyhow::bail!(
            "deletion vectors need a scan-wide row index, and the {offset} physical rows up \
             to the last DV file exceed polars' row index limit of {}",
            IdxSize::MAX
        );
    }
    Ok(spans)
}

pub(crate) fn footer_row_count(
    storage: &dyn StorageHandler,
    file: &ScanFileMeta,
) -> anyhow::Result<u64> {
    crate::engine::fetch_parquet_metadata(storage, &file.file)
        .map(|m| m.num_rows as u64)
        .map_err(|e| {
            anyhow::anyhow!(
                "row count from parquet footer of {}: {e:#}",
                file.file.location
            )
        })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use delta_kernel::FileMeta;
    use polars_utils::pl_path::PlRefPath;
    use url::Url;

    use super::*;
    use crate::scan::plan::LogicalRewrite;

    fn file(name: &str, num_records: Option<u64>, dv: bool) -> ScanFileMeta {
        ScanFileMeta {
            file: FileMeta {
                location: Url::parse(&format!("memory:///{name}")).unwrap(),
                last_modified: 0,
                size: 0,
            },
            path: PlRefPath::new(name),
            rewrite: LogicalRewrite {
                select: None,
                dv: dv.then(|| LazyDv::loaded(vec![1])),
            },
            partition_values: HashMap::new(),
            num_records,
        }
    }

    /// Spans accumulate over every earlier file, DV or not, and only files
    /// without stats up to the last DV file cost a footer read.
    #[test]
    fn spans_accumulate_and_footers_fill_missing_counts() {
        let files = vec![
            file("a", Some(10), false),
            file("b", None, true),
            file("c", Some(7), false),
            file("d", None, true),
            file("e", None, false),
        ];
        let fetched = Mutex::new(Vec::new());
        let spans = place_dvs(&files, |f| {
            let name = f.path.to_string();
            fetched.lock().unwrap().push(name.clone());
            Ok(match name.as_str() {
                "b" => 5,
                "d" => 3,
                other => panic!("unexpected footer read for {other}"),
            })
        })
        .unwrap();
        let mut fetched = fetched.into_inner().unwrap();
        fetched.sort();
        assert_eq!(fetched, ["b", "d"], "e is after the last DV file");
        assert_eq!(spans, [None, Some(10..15), None, Some(22..25), None]);
    }

    #[test]
    fn no_dv_needs_no_counts() {
        let files = vec![file("a", None, false), file("b", None, false)];
        let spans = place_dvs(&files, |f| panic!("footer read for {}", f.path)).unwrap();
        assert_eq!(spans, [None, None]);
    }

    #[test]
    fn footer_error_propagates() {
        let files = vec![file("a", None, false), file("b", Some(1), true)];
        let err =
            place_dvs(&files, |f| Err(anyhow::anyhow!("no footer for {}", f.path))).unwrap_err();
        assert!(err.to_string().contains("no footer for a"), "{err}");
    }

    #[test]
    fn past_the_index_limit_errors() {
        let files = vec![
            file("a", Some(IdxSize::MAX as u64), false),
            file("b", Some(1), true),
        ];
        let err = place_dvs(&files, |_| unreachable!()).unwrap_err();
        assert!(err.to_string().contains("row index limit"), "{err}");
    }
}
