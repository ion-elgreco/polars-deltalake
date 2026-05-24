# polars-deltalake

Native [Polars](https://pola.rs) I/O plugin for [Delta Lake](https://delta.io), backed by [delta-kernel-rs](https://github.com/delta-io/delta-kernel-rs).

```python
import polars as pl
from polars_deltalake import scan_delta

lf = scan_delta("s3://bucket/path/to/table")
df = lf.filter(pl.col("region") == "eu").select("id", "value").collect()
```

The plugin pushes **projection**, **slice** (`n_rows`), and **predicate** filters into the kernel scan. Predicates that don't map to a kernel-supported shape fall back to Polars-side filtering after the scan.

`scan_cdf` streams a Delta table's **Change Data Feed** as a `LazyFrame` between two commit versions:

```python
from polars_deltalake import scan_cdf

lf = scan_cdf("s3://bucket/path/to/table", start_version=10, end_version=20)
df = lf.filter(pl.col("_change_type") == "update_postimage").collect()
```

Output rows carry the table's columns plus `_change_type`, `_commit_version`, and `_commit_timestamp`. Requires `delta.enableChangeDataFeed = true` on the table.
