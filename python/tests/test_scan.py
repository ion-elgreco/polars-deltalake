"""Roundtrip tests for `polars_deltalake.scan_delta`.

Writes a Delta table with the `deltalake` package, then reads it back via
our polars-backed plugin and asserts data/shape equality.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_deltalake import TableScan, TableState, scan_delta


def _drain(scan: TableScan) -> pl.DataFrame:
    """Collect every morsel a directly-driven `TableScan` yields."""
    frames = []
    while (df := scan.next()) is not None:
        frames.append(df)
    return pl.concat(frames) if frames else pl.DataFrame()


@pytest.fixture
def simple_table(tmp_path):
    """Tiny non-partitioned table with three primitive columns."""
    from deltalake import write_deltalake

    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "carol", "dan", "eve"],
            "active": [True, False, True, True, False],
        }
    )
    write_deltalake(str(tmp_path / "tbl"), df.to_arrow())
    return tmp_path / "tbl"


@pytest.fixture
def multi_file_partitioned(tmp_path):
    """Three commits — each writes a separate parquet file partitioned by `g`."""
    from deltalake import write_deltalake

    table_path = str(tmp_path / "multi_part")
    for g, ids in [("a", [1, 2]), ("b", [3, 4]), ("c", [5, 6])]:
        write_deltalake(
            table_path,
            pl.DataFrame({"g": [g] * len(ids), "id": ids}).to_arrow(),
            mode="append" if g != "a" else "error",
            partition_by=["g"],
        )
    return table_path


class TestBasicScan:
    def test_full_table(self, simple_table):
        lf = scan_delta(str(simple_table))
        out = lf.collect().sort("id")
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "name": ["alice", "bob", "carol", "dan", "eve"],
                    "active": [True, False, True, True, False],
                }
            ),
        )

    def test_returns_lazyframe(self, simple_table):
        lf = scan_delta(str(simple_table))
        assert isinstance(lf, pl.LazyFrame)

    def test_projection_pushdown(self, simple_table):
        out = scan_delta(str(simple_table)).select("id", "name").collect().sort("id")
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "name": ["alice", "bob", "carol", "dan", "eve"],
                }
            ),
        )

    def test_slice_pushdown(self, simple_table):
        out = scan_delta(str(simple_table)).head(2).collect()
        assert out.height == 2

    def test_predicate_filter_postscan(self, simple_table):
        out = (
            scan_delta(str(simple_table)).filter(pl.col("active")).collect().sort("id")
        )
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "id": [1, 3, 4],
                    "name": ["alice", "carol", "dan"],
                    "active": [True, True, True],
                }
            ),
        )

    def test_multiple_files(self, tmp_path):
        """Multi-commit table — exercises the iterator across files."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "multi")
        a = pl.DataFrame({"x": [1, 2, 3]})
        b = pl.DataFrame({"x": [4, 5, 6]})
        write_deltalake(table_path, a.to_arrow())
        write_deltalake(table_path, b.to_arrow(), mode="append")

        out = scan_delta(table_path).collect().sort("x")
        assert_frame_equal(out, pl.DataFrame({"x": [1, 2, 3, 4, 5, 6]}))

    def test_time_travel(self, tmp_path):
        """Reading at version=0 should see only the first commit's rows."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "tt")
        a = pl.DataFrame({"x": [1, 2]})
        b = pl.DataFrame({"x": [3, 4]})
        write_deltalake(table_path, a.to_arrow())
        write_deltalake(table_path, b.to_arrow(), mode="append")

        v0 = scan_delta(table_path, version=0).collect().sort("x")
        assert_frame_equal(v0, pl.DataFrame({"x": [1, 2]}))

        latest = scan_delta(table_path).collect().sort("x")
        assert_frame_equal(latest, pl.DataFrame({"x": [1, 2, 3, 4]}))


class TestPartitionedScan:
    def test_partitioned_table(self, tmp_path):
        """Exercises the EvaluationHandler's Transform path (partition values
        get injected into each parquet batch as literal columns)."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "partitioned")
        df = pl.DataFrame(
            {
                "region": ["eu", "eu", "us", "us"],
                "id": [1, 2, 3, 4],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        write_deltalake(table_path, df.to_arrow(), partition_by=["region"])

        out = scan_delta(table_path).collect().sort("id")
        assert_frame_equal(out, df)

    def test_partition_col_only_projection(self, tmp_path):
        """Projecting only the partition column still works (the partition
        value comes from the Delta log via the Transform, not the parquet
        file)."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "partitioned_proj")
        df = pl.DataFrame({"region": ["eu", "us"], "id": [1, 2]})
        write_deltalake(table_path, df.to_arrow(), partition_by=["region"])

        out = scan_delta(table_path).select("region").collect().sort("region")
        assert_frame_equal(out, pl.DataFrame({"region": ["eu", "us"]}))

    def test_multi_column_partition(self, tmp_path):
        """Multi-column partition layout — `Transform` injects two literal cols."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "partitioned_multi")
        df = pl.DataFrame(
            {
                "year": [2024, 2024, 2025],
                "month": [1, 2, 1],
                "value": [10, 20, 30],
            }
        )
        write_deltalake(table_path, df.to_arrow(), partition_by=["year", "month"])

        out = scan_delta(table_path).collect().sort(["year", "month"])
        assert_frame_equal(out, df)


class TestPredicatePushdown:
    """Exact row results prove the Rust-side pushdown chain
    (kernel file-skip → polars-io row-group/row filter → option-2
    partition skip → orphan post-`transform_to_logical` eval) handles
    every case end-to-end. There is no Python-side correctness filter."""

    def test_partition_filter(self, multi_file_partitioned):
        out = (
            scan_delta(multi_file_partitioned)
            .filter(pl.col("g") == "b")
            .collect()
            .sort("id")
        )
        assert_frame_equal(out, pl.DataFrame({"g": ["b", "b"], "id": [3, 4]}))

    def test_and_chain(self, multi_file_partitioned):
        """Partition + data AND — kernel file-skips `g`, polars-io row-filters `id`."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter((pl.col("g") == "b") & (pl.col("id") >= 4))
            .collect()
        )
        assert_frame_equal(out, pl.DataFrame({"g": ["b"], "id": [4]}))

    def test_or_chain(self, multi_file_partitioned):
        """OR across partitions — single conjunct, translatable, kernel
        file-skips to the union of matching partitions."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter((pl.col("g") == "a") | (pl.col("g") == "c"))
            .collect()
            .sort("id")
        )
        assert_frame_equal(
            out, pl.DataFrame({"g": ["a", "a", "c", "c"], "id": [1, 2, 5, 6]})
        )

    def test_is_null(self, tmp_path):
        """Kernel `UnaryPredicate`."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "nulls")
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", None, "c"]})
        write_deltalake(table_path, df.to_arrow())

        out = scan_delta(table_path).filter(pl.col("name").is_null()).collect()
        assert_frame_equal(
            out,
            pl.DataFrame(
                {"id": [2], "name": [None]}, schema_overrides={"name": pl.String}
            ),
        )

    def test_is_between(self, multi_file_partitioned):
        """is_between → kernel `>=` AND `<=`."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter(pl.col("id").is_between(2, 5))
            .collect()
            .sort("id")
        )
        assert_frame_equal(
            out, pl.DataFrame({"g": ["a", "b", "b", "c"], "id": [2, 3, 4, 5]})
        )

    def test_is_in(self, multi_file_partitioned):
        """is_in → kernel `In` binary predicate."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter(pl.col("g").is_in(["a", "c"]))
            .collect()
            .sort("id")
        )
        assert_frame_equal(
            out, pl.DataFrame({"g": ["a", "a", "c", "c"], "id": [1, 2, 5, 6]})
        )

    def test_mixed_untranslatable_data_leg(self, multi_file_partitioned):
        """Kernel handles partition `g == 'b'`, polars-io row-filters the
        untranslatable `id.abs() >= 4`."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter((pl.col("g") == "b") & (pl.col("id").abs() >= 4))
            .collect()
        )
        assert_frame_equal(out, pl.DataFrame({"g": ["b"], "id": [4]}))

    def test_mixed_untranslatable_partition_leg(self, multi_file_partitioned):
        """Option-2 polars-driven file-skip handles untranslatable
        `g.upper() == 'B'`; kernel + polars-io handle `id >= 4`."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter((pl.col("g").str.to_uppercase() == "B") & (pl.col("id") >= 4))
            .collect()
        )
        assert_frame_equal(out, pl.DataFrame({"g": ["b"], "id": [4]}))

    def test_three_way_and(self, multi_file_partitioned):
        """Three-conjunct AND — partition + translatable data + untranslatable data."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter(
                (pl.col("g") == "b") & (pl.col("id") >= 3) & (pl.col("id").abs() <= 5)
            )
            .collect()
            .sort("id")
        )
        assert_frame_equal(out, pl.DataFrame({"g": ["b", "b"], "id": [3, 4]}))

    def test_bare_bool_column(self, tmp_path):
        """Kernel `Predicate::BooleanExpression(Column(...))`."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "boolfilter")
        df = pl.DataFrame({"id": [1, 2, 3, 4], "active": [True, False, True, None]})
        write_deltalake(table_path, df.to_arrow())

        out = scan_delta(table_path).filter(pl.col("active")).collect().sort("id")
        assert_frame_equal(out, pl.DataFrame({"id": [1, 3], "active": [True, True]}))

    def test_ne_missing_distinct(self, tmp_path):
        """`ne_missing` → kernel `Predicate::Binary(Distinct)`. Unlike `!=`,
        returns TRUE for `NULL ne_missing 5`, so the null row is kept."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "nemiss")
        df = pl.DataFrame({"id": [1, 2, 3, 4], "v": [5, 5, 7, None]})
        write_deltalake(table_path, df.to_arrow())

        out = (
            scan_delta(table_path)
            .filter(pl.col("v").ne_missing(5))
            .collect()
            .sort("id")
        )
        assert_frame_equal(out, pl.DataFrame({"id": [3, 4], "v": [7, None]}))

    def test_not_via_neq(self, multi_file_partitioned):
        """`!=` → kernel `Predicate::Not(Binary(Equal))`."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter(pl.col("g") != "b")
            .collect()
            .sort("id")
        )
        assert_frame_equal(
            out, pl.DataFrame({"g": ["a", "a", "c", "c"], "id": [1, 2, 5, 6]})
        )

    def test_not_around_is_null(self, tmp_path):
        """`~is_null()` → kernel `Predicate::Not(Unary(IsNull))`."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "notnull")
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", None, "c"]})
        write_deltalake(table_path, df.to_arrow())

        out = (
            scan_delta(table_path)
            .filter(pl.col("name").is_null().not_())
            .collect()
            .sort("id")
        )
        assert_frame_equal(out, pl.DataFrame({"id": [1, 3], "name": ["a", "c"]}))

    def test_untranslatable_data_op(self, simple_table):
        """Untranslatable string op on a data column — polars-io's row-level
        `.filter` handles it."""
        out = (
            scan_delta(str(simple_table))
            .filter(pl.col("name").str.starts_with("a"))
            .collect()
        )
        assert_frame_equal(
            out, pl.DataFrame({"id": [1], "name": ["alice"], "active": [True]})
        )

    @pytest.mark.xfail(
        reason="polars optimizer floor-casts sub-µs Datetime(ns) literals to "
        "the µs source-schema precision at plan time, so the predicate that "
        "reaches our Rust callback is already lossy. Needs upstream polars "
        "fix to operator-aware (or refuse) the implicit narrowing cast.",
        strict=True,
    )
    def test_ns_subus_lt_keeps_matching_row(self, tmp_path):
        """Sub-µs ns literal on `<` against a µs column. Row at 1_000 ns
        satisfies `< 1_500 ns`; polars's plan-time floor-cast turns this into
        `1 µs < 1 µs` (false) and drops the row."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "ns_subus")
        df = pl.DataFrame(
            {"id": [1], "ts": [1_000]},
            schema={"id": pl.Int64, "ts": pl.Datetime("ns")},
        )
        write_deltalake(table_path, df.to_arrow())

        out = (
            scan_delta(table_path)
            .filter(pl.col("ts") < pl.lit(1_500).cast(pl.Datetime("ns")))
            .collect()
        )
        assert_frame_equal(out, df)


class TestPartitionSkip:
    """Option-2 file skipping via polars evaluation of partition values for
    predicates kernel can't translate."""

    def test_untranslatable_str_op(self, multi_file_partitioned):
        out = (
            scan_delta(multi_file_partitioned)
            .filter(pl.col("g").str.to_uppercase() == "B")
            .collect()
            .sort("id")
        )
        assert_frame_equal(out, pl.DataFrame({"g": ["b", "b"], "id": [3, 4]}))

    def test_untranslatable_dt_year(self, tmp_path):
        """Date partition with `dt.year()` — non-string-cast partition col."""
        from datetime import date

        from deltalake import write_deltalake

        table_path = str(tmp_path / "date_partitioned")
        for d, ids in [
            (date(2023, 6, 1), [1, 2]),
            (date(2024, 3, 15), [3, 4]),
            (date(2025, 1, 1), [5, 6]),
        ]:
            write_deltalake(
                table_path,
                pl.DataFrame({"d": [d] * len(ids), "id": ids}).to_arrow(),
                mode="append" if d != date(2023, 6, 1) else "error",
                partition_by=["d"],
            )

        out = (
            scan_delta(table_path)
            .filter(pl.col("d").dt.year() == 2024)
            .collect()
            .sort("id")
        )
        assert_frame_equal(
            out, pl.DataFrame({"d": [date(2024, 3, 15)] * 2, "id": [3, 4]})
        )

    def test_drops_every_file(self, multi_file_partitioned):
        out = (
            scan_delta(multi_file_partitioned)
            .filter(pl.col("g").str.to_uppercase() == "Z")
            .collect()
        )
        assert out.height == 0


class TestPredicateRouting:
    """`_classify_predicate` returns per-bucket counts. Buckets are
    **non-disjoint**: every kernel-translatable conjunct appears in
    `kernel` (file-level stats) AND in whichever other bucket handles its
    row-level evaluation."""

    def test_translatable_data(self, multi_file_partitioned):
        """`id >= 3` — kernel file-stats + parquet row-group/row filter."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate(pl.col("id") >= 3) == {  # type: ignore
            "kernel": 1,
            "parquet_filter": 1,
            "partition_prune": 0,
            "post_transform": 0,
        }

    def test_translatable_partition(self, multi_file_partitioned):
        """`g == 'b'` — kernel file-skips, and `partition_prune` evaluates it
        exactly. Kernel's pruning alone keeps every file it cannot decide, and
        polars drops its own filter once we accept the predicate."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate(pl.col("g") == "b") == {  # type: ignore
            "kernel": 1,
            "parquet_filter": 0,
            "partition_prune": 1,
            "post_transform": 0,
        }

    def test_untranslatable_data(self, multi_file_partitioned):
        """`id.abs() >= 4` — only parquet_filter; kernel can't translate `abs`."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate(pl.col("id").abs() >= 4) == {  # type: ignore
            "kernel": 0,
            "parquet_filter": 1,
            "partition_prune": 0,
            "post_transform": 0,
        }

    def test_untranslatable_partition(self, multi_file_partitioned):
        """`g.str.to_uppercase() == 'B'` — option-2 partition prune only."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate(pl.col("g").str.to_uppercase() == "B") == {  # type: ignore
            "kernel": 0,
            "parquet_filter": 0,
            "partition_prune": 1,
            "post_transform": 0,
        }

    def test_translatable_mixed_atomic(self, multi_file_partitioned):
        """`(g == 'a') | (id == 3)` — kernel best-effort file-skips, but
        rows in surviving files need post-transform eval."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate(  # type: ignore
            (pl.col("g") == "a") | (pl.col("id") == 3)
        ) == {
            "kernel": 1,
            "parquet_filter": 0,
            "partition_prune": 0,
            "post_transform": 1,
        }

    def test_untranslatable_mixed_atomic(self, multi_file_partitioned):
        """`(g.upper() == 'A') | (id == 3)` — kernel can't translate; only
        post-transform eval can handle it."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate(  # type: ignore
            (pl.col("g").str.to_uppercase() == "A") | (pl.col("id") == 3)
        ) == {
            "kernel": 0,
            "parquet_filter": 0,
            "partition_prune": 0,
            "post_transform": 1,
        }

    def test_and_chain_routes_per_conjunct(self, multi_file_partitioned):
        """Top-level AND splits; both conjuncts are translatable so both go
        to kernel. `id >= 4` additionally goes to parquet_filter."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate((pl.col("g") == "b") & (pl.col("id") >= 4)) == {  # type: ignore
            "kernel": 2,
            "parquet_filter": 1,
            "partition_prune": 1,
            "post_transform": 0,
        }

    def test_three_way_and_routes_per_conjunct(self, multi_file_partitioned):
        """3 conjuncts: 2 translatable (kernel) + 2 data (parquet_filter)."""
        src = TableState(multi_file_partitioned)
        assert src._classify_predicate(  # type: ignore
            (pl.col("g") == "b") & (pl.col("id") >= 3) & (pl.col("id").abs() <= 5)
        ) == {
            "kernel": 2,  # g == 'b', id >= 3
            "parquet_filter": 2,  # id >= 3, id.abs() <= 5
            "partition_prune": 1,  # g == 'b'
            "post_transform": 0,
        }


class TestMixedAtomicConjunct:
    """Atomic OR conjuncts touching both partition and data cols. Neither
    kernel, polars-io, nor option-2 can pre-filter them — they're evaluated
    in `LogicalScanIter::apply_rewrite` after `transform_to_logical`
    materializes the partition columns."""

    def test_or_partition_and_data(self, multi_file_partitioned):
        """`(g.upper() == 'A') | (id == 3)`: partition leg untranslatable,
        OR'd with a data leg → only post-`transform_to_logical` eval works."""
        out = (
            scan_delta(multi_file_partitioned)
            .filter((pl.col("g").str.to_uppercase() == "A") | (pl.col("id") == 3))
            .collect()
            .sort("id")
        )
        assert_frame_equal(out, pl.DataFrame({"g": ["a", "a", "b"], "id": [1, 2, 3]}))

    def test_unprojected_partition_column_is_read_then_dropped(
        self, multi_file_partitioned
    ):
        """The scan widens past the projection to give the conjunct its
        partition column, then projects back down. `scan_delta` never asks for
        this — polars projects what its own pushdown references — so drive
        `TableScan` directly."""
        scan = TableScan(TableState(multi_file_partitioned))
        scan.configure(
            ["id"], None, (pl.col("g").str.to_uppercase() == "A") | (pl.col("id") == 3)
        )
        out = _drain(scan).sort("id")
        assert_frame_equal(out, pl.DataFrame({"id": [1, 2, 3]}))

    def test_unprojected_data_column_is_read_then_dropped(self, multi_file_partitioned):
        """Same widening for the data leg: `id` is read only to evaluate the
        conjunct and never reaches the caller."""
        scan = TableScan(TableState(multi_file_partitioned))
        scan.configure(
            ["g"], None, (pl.col("g").str.to_uppercase() == "A") | (pl.col("id") == 3)
        )
        out = _drain(scan).sort("g")
        assert_frame_equal(out, pl.DataFrame({"g": ["a", "a", "b"]}))

    def test_widening_survives_full_partition_pruning(self, multi_file_partitioned):
        """AND-ing a partition conjunct that prunes every file returns no rows
        rather than failing on the widened column."""
        scan = TableScan(TableState(multi_file_partitioned))
        scan.configure(
            ["id"],
            None,
            ((pl.col("g").str.to_uppercase() == "A") | (pl.col("id") == 3))
            & (pl.col("g") == "nonexistent"),
        )
        assert _drain(scan).height == 0

    def test_predicate_column_absent_from_table_errors(self, multi_file_partitioned):
        """Widening can only reach columns the table declares."""
        scan = TableScan(TableState(multi_file_partitioned))
        scan.configure(
            ["id"], None, (pl.col("g") == "a") | (pl.col("nope").str.len_chars() == 3)
        )
        with pytest.raises(RuntimeError, match="nope which the table does not have"):
            scan.next()


class TestDeletionVectors:
    """`deltalake` never writes deletion vectors — it rewrites the file even
    with `delta.enableDeletionVectors` set. Only `test_dv_with_predicate`,
    which reads the DAT fixture, reaches the keep-mask."""

    def test_delete_via_rewrite(self, tmp_path):
        """Plain DELETE (no DV protocol) — deltalake rewrites the file."""
        from deltalake import DeltaTable, write_deltalake

        table_path = str(tmp_path / "deleted")
        df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
        write_deltalake(table_path, df.to_arrow())

        DeltaTable(table_path).delete("id >= 3")

        out = scan_delta(table_path).collect().sort("id")
        assert_frame_equal(out, pl.DataFrame({"id": [1, 2], "name": ["a", "b"]}))

    def test_dv_enabled_delete(self, tmp_path):
        """DELETE on a DV-enabled table — `deltalake` still rewrites the
        file, so this covers the rewrite path with the DV protocol on."""
        from deltalake import DeltaTable, write_deltalake

        table_path = str(tmp_path / "dv")
        df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
        write_deltalake(
            table_path,
            df.to_arrow(),
            configuration={"delta.enableDeletionVectors": "true"},
        )

        DeltaTable(table_path).delete("id == 2 OR id == 4")

        out = scan_delta(table_path).collect().sort("id")
        assert_frame_equal(
            out, pl.DataFrame({"id": [1, 3, 5], "name": ["a", "c", "e"]})
        )

    def test_dv_with_projection(self, tmp_path):
        """DELETE + projection on a DV-enabled table — deleted rows must not
        leak into the projected output."""
        from deltalake import DeltaTable, write_deltalake

        table_path = str(tmp_path / "dv_proj")
        df = pl.DataFrame({"id": [1, 2, 3, 4, 5, 6], "value": [10, 20, 30, 40, 50, 60]})
        write_deltalake(
            table_path,
            df.to_arrow(),
            configuration={"delta.enableDeletionVectors": "true"},
        )

        DeltaTable(table_path).delete("id < 4")

        out = scan_delta(table_path).select("value").collect().sort("value")
        assert_frame_equal(out, pl.DataFrame({"value": [40, 50, 60]}))

    def test_dv_with_predicate(self):
        """A DV indexes the file's *physical* rows, so a pushed-down predicate
        must not run before the keep-mask. Uses the DAT fixture because
        `deltalake` cannot write a deletion vector."""
        from _dat_helper import reader_cases

        case = next(c for c in reader_cases() if c.name == "deletion_vectors")
        table = str(case / "delta")
        # The DELETE behind this DV removed every `letter == 'a'` row.
        survivor = [{"letter": "b", "int": 228}]

        def read(pred=None):
            lf = scan_delta(table)
            if pred is not None:
                lf = lf.filter(pred)
            return lf.select("letter", "int").collect().to_dicts()

        assert read() == survivor
        # Filtering before the mask resurrected the deleted `a, 692` row.
        assert read(pl.col("int") > 100) == survivor
        # ...and dropped the one live row entirely.
        assert read(pl.col("letter") == "b") == survivor


class TestEagerRead:
    def test_read_delta_eager(self, simple_table):
        """`read_delta` is the eager equivalent of `scan_delta(...).collect()`."""
        from polars_deltalake import read_delta

        out = read_delta(str(simple_table)).sort("id")
        assert isinstance(out, pl.DataFrame)
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "name": ["alice", "bob", "carol", "dan", "eve"],
                    "active": [True, False, True, True, False],
                }
            ),
        )

    def test_version_arg(self, tmp_path):
        """`read_delta` honours the `version` kwarg for time travel."""
        from deltalake import write_deltalake

        from polars_deltalake import read_delta

        table_path = str(tmp_path / "ver")
        write_deltalake(table_path, pl.DataFrame({"x": [1]}).to_arrow())
        write_deltalake(table_path, pl.DataFrame({"x": [2]}).to_arrow(), mode="append")

        v0 = read_delta(table_path, version=0)
        assert_frame_equal(v0, pl.DataFrame({"x": [1]}))

    @pytest.mark.parametrize("engine", ["auto", "in-memory", "streaming"])
    def test_engine_arg(self, simple_table, engine):
        """`read_delta` forwards `engine` to `LazyFrame.collect`."""
        from polars_deltalake import read_delta

        out = read_delta(str(simple_table), engine=engine).sort("id")
        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "name": ["alice", "bob", "carol", "dan", "eve"],
                    "active": [True, False, True, True, False],
                }
            ),
        )


class TestScanOrderStability:
    """Kernel leaves row order unspecified, so no particular order is promised.
    Repeated reads of an unchanged table must still agree, or `head(n)` returns
    different rows every run."""

    @pytest.fixture
    def many_files(self, tmp_path):
        """Twelve commits, one parquet file each."""
        from deltalake import write_deltalake

        table_path = str(tmp_path / "many")
        for i in range(12):
            write_deltalake(
                table_path,
                pl.DataFrame({"id": [i * 10, i * 10 + 1]}).to_arrow(),
                mode="error" if i == 0 else "append",
            )
        return table_path

    def test_full_scan_order_is_stable(self, many_files):
        orders = {
            tuple(scan_delta(many_files).collect()["id"].to_list()) for _ in range(5)
        }
        assert len(orders) == 1

    def test_head_returns_the_same_rows(self, many_files):
        heads = {
            tuple(scan_delta(many_files).head(4).collect()["id"].to_list())
            for _ in range(5)
        }
        assert len(heads) == 1
