# polars-deltalake

Native [Polars](https://pola.rs) I/O plugin for [Delta Lake](https://delta.io), backed by [delta-kernel-rs](https://github.com/delta-io/delta-kernel-rs).

```python
import polars as pl
from polars_deltalake import scan_delta

lf = scan_delta("s3://bucket/path/to/table")
df = lf.filter(pl.col("region") == "eu").select("id", "value").collect()
```

The plugin pushes **projection**, **slice** (`n_rows`), and **predicate** filters into the kernel scan. Predicates that don't map to a kernel-supported shape fall back to Polars-side filtering after the scan.
