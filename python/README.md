A native deltalake reader built with Pola-rs and Delta-RS crates.


## Example

```python
import polars as pl
import polars_deltalake as pldl

df = pl.DataFrame(
    [
        pl.Series("foo", [1, 2, 3], dtype=pl.Int64),
        pl.Series("bar", ['1', '2', '3'], dtype=pl.String),
        pl.Series("datetime", [datetime(2010, 1, 1, 0, 0)]*3, dtype=pl.Datetime(time_unit='us', time_zone=None)),
        pl.Series("datetime_tz", [datetime(2010, 1, 1, 0, 0, tzinfo=timezone.utc)]*3, dtype=pl.Datetime(time_unit='us', time_zone='UTC')),
        pl.Series("date_month", [201001, 201002, 201003], dtype=pl.Int32),
        pl.Series("static_part", ['A', 'A', 'A'], dtype=pl.String),
        pl.Series("list", [['5', 'B'], ['5', 'B'], ['5', 'B']], dtype=pl.List(pl.String)),
    ]
)

df.write_delta("test_table", delta_write_options={"partition_by":["foo"]})

table = pldl.scan_delta("test_table")

table.filter(pl.col("foo") == 1).collect()

shape: (1, 7)
┌─────┬─────────────────────┬─────────────────────┬────────────┬─────────────┬────────────┬─────┐
│ bar ┆ datetime            ┆ datetime_tz         ┆ date_month ┆ static_part ┆ list       ┆ foo │
│ --- ┆ ---                 ┆ ---                 ┆ ---        ┆ ---         ┆ ---        ┆ --- │
│ str ┆ datetime[μs]        ┆ datetime[μs, UTC]   ┆ i32        ┆ str         ┆ list[str]  ┆ i64 │
╞═════╪═════════════════════╪═════════════════════╪════════════╪═════════════╪════════════╪═════╡
│ 1   ┆ 2010-01-01 00:00:00 ┆ 2010-01-01 00:00:00 ┆ 201001     ┆ A           ┆ ["5", "B"] ┆ 1   │
│     ┆                     ┆ UTC                 ┆            ┆             ┆            ┆     │
└─────┴─────────────────────┴─────────────────────┴────────────┴─────────────┴────────────┴─────┘
```
