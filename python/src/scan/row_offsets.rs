//! Places each DV file in the scan's physical row index.
//!
//! polars numbers physical rows across every file in path order, so a DV
//! file's rows start at the row count of all files before it. The count
//! comes from the log's `numRecords` stat; a writer that left stats out
//! costs one footer read per file.

use polars::io::cloud::CloudOptions;
use polars::prelude::{IdxSize, ParquetObjectStore};
use polars_utils::pl_path::PlRefPath;

use crate::scan::plan::ScanFileMeta;

pub(crate) fn assign_dv_row_offsets(
    files: &mut [ScanFileMeta],
    cloud_opts: Option<&CloudOptions>,
) -> anyhow::Result<()> {
    assign_with(files, |path| footer_row_count(path, cloud_opts))
}

/// Every file up to the last DV file needs a count: earlier files place a
/// DV file, and its own count bounds the index → file mapping.
fn assign_with(
    files: &mut [ScanFileMeta],
    mut row_count: impl FnMut(&PlRefPath) -> anyhow::Result<u64>,
) -> anyhow::Result<()> {
    let Some(last_dv) = files.iter().rposition(|f| f.rewrite.dv.is_some()) else {
        return Ok(());
    };
    let mut offset: u64 = 0;
    for file in &mut files[..=last_dv] {
        let num_rows = match file.num_records {
            Some(n) => n,
            None => row_count(&file.path)?,
        };
        if let Some(dv) = file.rewrite.dv.as_mut() {
            dv.row_offset = offset;
            dv.num_rows = num_rows;
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
    Ok(())
}

fn footer_row_count(path: &PlRefPath, cloud_opts: Option<&CloudOptions>) -> anyhow::Result<u64> {
    let uri = path.clone();
    let n = crate::engine::rt()
        .block_on(async move {
            let mut store = ParquetObjectStore::from_uri(uri, cloud_opts, None).await?;
            store.num_rows_only().await
        })
        .map_err(|e| anyhow::anyhow!("row count from parquet footer of {path}: {e:#}"))?;
    u64::try_from(n).map_err(|_| anyhow::anyhow!("parquet footer of {path} reports {n} rows"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::scan::plan::{DvState, LogicalRewrite};

    fn file(name: &str, num_records: Option<u64>, dv: bool) -> ScanFileMeta {
        ScanFileMeta {
            path: PlRefPath::new(name),
            rewrite: LogicalRewrite {
                select: None,
                dv: dv.then(|| DvState::new(vec![1])),
            },
            partition_values: HashMap::new(),
            num_records,
        }
    }

    fn placement(f: &ScanFileMeta) -> Option<(u64, u64)> {
        f.rewrite.dv.as_ref().map(|dv| (dv.row_offset, dv.num_rows))
    }

    /// Offsets accumulate over every earlier file, DV or not, and only files
    /// without stats up to the last DV file cost a footer read.
    #[test]
    fn offsets_accumulate_and_footers_fill_missing_counts() {
        let mut files = vec![
            file("a", Some(10), false),
            file("b", None, true),
            file("c", Some(7), false),
            file("d", None, true),
            file("e", None, false),
        ];
        let mut fetched = Vec::new();
        assign_with(&mut files, |p| {
            fetched.push(p.to_string());
            Ok(match p.as_str() {
                "b" => 5,
                "d" => 3,
                other => panic!("unexpected footer read for {other}"),
            })
        })
        .unwrap();
        assert_eq!(fetched, ["b", "d"], "e is after the last DV file");
        assert_eq!(placement(&files[1]), Some((10, 5)));
        assert_eq!(placement(&files[3]), Some((22, 3)));
    }

    #[test]
    fn no_dv_needs_no_counts() {
        let mut files = vec![file("a", None, false), file("b", None, false)];
        assign_with(&mut files, |p| panic!("footer read for {p}")).unwrap();
    }

    #[test]
    fn footer_error_propagates() {
        let mut files = vec![file("a", None, false), file("b", Some(1), true)];
        let err =
            assign_with(&mut files, |p| Err(anyhow::anyhow!("no footer for {p}"))).unwrap_err();
        assert!(err.to_string().contains("no footer for a"), "{err}");
    }

    #[test]
    fn past_the_index_limit_errors() {
        let mut files = vec![
            file("a", Some(IdxSize::MAX as u64), false),
            file("b", Some(1), true),
        ];
        let err = assign_with(&mut files, |_| unreachable!()).unwrap_err();
        assert!(err.to_string().contains("row index limit"), "{err}");
    }
}
