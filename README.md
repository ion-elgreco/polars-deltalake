# polars-deltalake

Native [Polars](https://pola.rs) I/O plugin for [Delta Lake](https://delta.io), backed by [delta-kernel-rs](https://github.com/delta-io/delta-kernel-rs).

```python
import polars as pl
from polars_deltalake import scan_delta

lf = scan_delta("s3://bucket/path/to/table")
df = lf.filter(pl.col("region") == "eu").select("id", "value").collect()
```

`scan_delta` returns a `LazyFrame`; `read_delta(uri)` is the eager shortcut for `scan_delta(uri).collect()`.

### Change Data Feed

```python
from polars_deltalake import scan_cdf

lf = scan_cdf("s3://bucket/path/to/table", start_version=10, end_version=20)
df = lf.filter(pl.col("_change_type") == "update_postimage").collect()
```

Streams the Change Data Feed between two commit versions. Each row carries the table's columns plus `_change_type` (`insert` / `delete` / `update_preimage` / `update_postimage`), `_commit_version`, and `_commit_timestamp`. Requires `delta.enableChangeDataFeed = true` on the table. `read_cdf` is the eager shortcut.

## What it does

- Reads Delta tables (local `file://`, S3, Azure, GCS, http(s)) entirely through polars-io + object_store — no `pyarrow`, no `deltalake` package required at runtime.
- Honors **projection**, **slice** (`n_rows`), and **predicate** pushdown into the kernel scan; predicates that don't have a kernel-supported shape fall back to polars-side filtering.
- Reads **deletion vectors** and **column-mapped** tables (modes `id` and `name`).
- Streams the **Change Data Feed** between two versions via `scan_cdf` / `read_cdf`.

## What it doesn't (yet)

- No write path — `scan_delta` / `scan_cdf` / `read_delta` / `read_cdf` only.

## Install

```bash
pip install polars-deltalake
```
