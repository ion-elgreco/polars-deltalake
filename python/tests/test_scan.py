"""Roundtrip tests for `polars_deltalake.scan_delta`.

Writes a Delta table with the `deltalake` package, then reads it back via
our polars-backed plugin and asserts data/shape equality.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_deltalake import scan_delta


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


def test_scan_full_table(simple_table):
    lf = scan_delta(str(simple_table))
    out = lf.collect().sort("id")
    assert out.shape == (5, 3)
    assert out["id"].to_list() == [1, 2, 3, 4, 5]
    assert out["name"].to_list() == ["alice", "bob", "carol", "dan", "eve"]
    assert out["active"].to_list() == [True, False, True, True, False]


def test_scan_returns_lazyframe(simple_table):
    lf = scan_delta(str(simple_table))
    assert isinstance(lf, pl.LazyFrame)


def test_projection_pushdown(simple_table):
    out = scan_delta(str(simple_table)).select("id", "name").collect().sort("id")
    assert out.columns == ["id", "name"]
    assert out["id"].to_list() == [1, 2, 3, 4, 5]


def test_slice_pushdown(simple_table):
    out = scan_delta(str(simple_table)).head(2).collect()
    assert out.height == 2


def test_predicate_filter_postscan(simple_table):
    # The plugin doesn't push predicates yet — polars filters them after the
    # scan. Verify the result is still correct.
    out = scan_delta(str(simple_table)).filter(pl.col("active")).collect().sort("id")
    assert out["id"].to_list() == [1, 3, 4]


def test_multiple_files(tmp_path):
    """Multi-commit table — exercises the iterator across files."""
    from deltalake import write_deltalake

    table_path = str(tmp_path / "multi")
    a = pl.DataFrame({"x": [1, 2, 3]})
    b = pl.DataFrame({"x": [4, 5, 6]})
    write_deltalake(table_path, a.to_arrow())
    write_deltalake(table_path, b.to_arrow(), mode="append")

    out = scan_delta(table_path).collect().sort("x")
    assert out["x"].to_list() == [1, 2, 3, 4, 5, 6]


def test_time_travel(tmp_path):
    """Reading at version=0 should see only the first commit's rows."""
    from deltalake import write_deltalake

    table_path = str(tmp_path / "tt")
    a = pl.DataFrame({"x": [1, 2]})
    b = pl.DataFrame({"x": [3, 4]})
    write_deltalake(table_path, a.to_arrow())
    write_deltalake(table_path, b.to_arrow(), mode="append")

    v0 = scan_delta(table_path, version=0).collect().sort("x")
    assert v0["x"].to_list() == [1, 2]

    latest = scan_delta(table_path).collect().sort("x")
    assert latest["x"].to_list() == [1, 2, 3, 4]


def test_partitioned_table(tmp_path):
    """Exercises the EvaluationHandler's Transform path (partition values get
    injected into each parquet batch as literal columns)."""
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
    assert set(out.columns) == {"region", "id", "value"}
    assert out["region"].to_list() == ["eu", "eu", "us", "us"]
    assert out["id"].to_list() == [1, 2, 3, 4]
    assert out["value"].to_list() == [10.0, 20.0, 30.0, 40.0]


def test_partitioned_table_projection(tmp_path):
    """Projecting only the partition column still works (the partition value
    comes from the Delta log via the Transform, not the parquet file)."""
    from deltalake import write_deltalake

    table_path = str(tmp_path / "partitioned_proj")
    df = pl.DataFrame(
        {
            "region": ["eu", "us"],
            "id": [1, 2],
        }
    )
    write_deltalake(table_path, df.to_arrow(), partition_by=["region"])

    out = scan_delta(table_path).select("region").collect().sort("region")
    assert out.columns == ["region"]
    assert out["region"].to_list() == ["eu", "us"]


def test_partitioned_table_multi_column(tmp_path):
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
    assert out["year"].to_list() == [2024, 2024, 2025]
    assert out["month"].to_list() == [1, 2, 1]
    assert out["value"].to_list() == [10, 20, 30]


# ---------------------------------------------------------------------------
# Predicate pushdown (kernel-side data skipping)
# ---------------------------------------------------------------------------


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


def test_predicate_pushdown_partition_filter(multi_file_partitioned):
    """`g == 'b'` should prune the other partitions via kernel data skipping
    AND produce the right rows post-filter."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter(pl.col("g") == "b")
        .collect()
        .sort("id")
    )
    assert out["g"].to_list() == ["b", "b"]
    assert out["id"].to_list() == [3, 4]


def test_predicate_pushdown_and_chain(multi_file_partitioned):
    """Two-clause AND predicate — both sides translate, both kernel and polars
    apply correctly."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter((pl.col("g") == "b") & (pl.col("id") >= 4))
        .collect()
    )
    assert out["id"].to_list() == [4]


def test_predicate_pushdown_or_chain(multi_file_partitioned):
    """OR across partitions — kernel should keep partitions matching either
    side; polars filters to the union."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter((pl.col("g") == "a") | (pl.col("g") == "c"))
        .collect()
        .sort("id")
    )
    assert out["g"].to_list() == ["a", "a", "c", "c"]
    assert out["id"].to_list() == [1, 2, 5, 6]


def test_predicate_pushdown_is_null(tmp_path):
    """IS NULL pushdown — verify the translator builds the right kernel
    UnaryPredicate and polars still surfaces the null rows."""
    from deltalake import write_deltalake

    table_path = str(tmp_path / "nulls")
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", None, "c"]})
    write_deltalake(table_path, df.to_arrow())

    out = scan_delta(table_path).filter(pl.col("name").is_null()).collect()
    assert out["id"].to_list() == [2]


def test_predicate_pushdown_is_between(multi_file_partitioned):
    """is_between (closed interval) decomposes into >= AND <= for kernel."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter(pl.col("id").is_between(2, 5))
        .collect()
        .sort("id")
    )
    assert out["id"].to_list() == [2, 3, 4, 5]


def test_predicate_pushdown_is_in(multi_file_partitioned):
    """is_in pushes through to kernel as the In binary predicate."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter(pl.col("g").is_in(["a", "c"]))
        .collect()
        .sort("id")
    )
    assert out["g"].to_list() == ["a", "a", "c", "c"]


def test_predicate_pushdown_mixed_untranslatable_data_leg(multi_file_partitioned):
    """Mixed predicate where the data leg is untranslatable (`abs()`).
    Before conjunct splitting the whole predicate would have been dropped
    from polars-io (partition col present) AND kernel (untranslatable
    conjunct present), forcing a full scan. With splitting:
      - `g == 'b'` → kernel file-skip
      - `id.abs() >= 4` → polars-io row-group skip
    """
    out = (
        scan_delta(multi_file_partitioned)
        .filter((pl.col("g") == "b") & (pl.col("id").abs() >= 4))
        .collect()
    )
    assert out["id"].to_list() == [4]


def test_predicate_pushdown_mixed_untranslatable_partition_leg(multi_file_partitioned):
    """Mixed predicate where the partition leg is untranslatable
    (`str.to_uppercase()`). The partition conjunct is orphaned (only
    Python-side filter handles it); the data conjunct still gets pushed
    to kernel and polars-io."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter((pl.col("g").str.to_uppercase() == "B") & (pl.col("id") >= 4))
        .collect()
    )
    assert out["id"].to_list() == [4]


def test_predicate_pushdown_three_way_and(multi_file_partitioned):
    """Three-conjunct AND mixing partition, translatable data, and
    untranslatable data — verifies the flatten walker recurses through
    nested BinaryExpr::And nodes (polars associates left-to-right, so
    this is parsed as `((g == 'b') AND (id >= 3)) AND (id.abs() <= 5)`)."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter(
            (pl.col("g") == "b") & (pl.col("id") >= 3) & (pl.col("id").abs() <= 5)
        )
        .collect()
        .sort("id")
    )
    assert out["id"].to_list() == [3, 4]


def test_predicate_pushdown_bare_bool_column(tmp_path):
    """A bare bool column used as a filter (`lf.filter(pl.col("active"))`)
    pushes as kernel `Predicate::BooleanExpression(Column(...))` — exercises
    translate_predicate's `BooleanExpression` arm."""
    from deltalake import write_deltalake

    table_path = str(tmp_path / "boolfilter")
    df = pl.DataFrame({"id": [1, 2, 3, 4], "active": [True, False, True, None]})
    write_deltalake(table_path, df.to_arrow())

    out = scan_delta(table_path).filter(pl.col("active")).collect().sort("id")
    # SQL WHERE keeps only rows where the predicate evaluates TRUE; null
    # rows are excluded by kernel's null-safe expansion.
    assert out["id"].to_list() == [1, 3]
    assert out["active"].to_list() == [True, True]


def test_predicate_pushdown_ne_missing_distinct(tmp_path):
    """Null-aware inequality (`ne_missing`) pushes as kernel
    `Predicate::Binary(Distinct)` — exercises translate_predicate's
    `Distinct` arm. Unlike `!=`, this returns TRUE for `NULL ne_missing 5`
    instead of NULL, so the null row is kept."""
    from deltalake import write_deltalake

    table_path = str(tmp_path / "nemiss")
    df = pl.DataFrame({"id": [1, 2, 3, 4], "v": [5, 5, 7, None]})
    write_deltalake(table_path, df.to_arrow())

    out = scan_delta(table_path).filter(pl.col("v").ne_missing(5)).collect().sort("id")
    # null vs 5 → DISTINCT → kept; 7 vs 5 → DISTINCT → kept; 5 vs 5 → not distinct → dropped.
    assert out["id"].to_list() == [3, 4]
    assert out["v"].to_list() == [7, None]


def test_predicate_pushdown_not_via_neq(multi_file_partitioned):
    """`!=` round-trips as kernel `Predicate::Not(Binary(Equal))` —
    exercises translate_predicate's `Not` arm via parquet-level pushdown."""
    out = (
        scan_delta(multi_file_partitioned)
        .filter(pl.col("g") != "b")
        .collect()
        .sort("id")
    )
    assert out["g"].to_list() == ["a", "a", "c", "c"]
    assert out["id"].to_list() == [1, 2, 5, 6]


def test_predicate_pushdown_not_around_is_null(tmp_path):
    """`~is_null()` round-trips as kernel `Predicate::Not(Unary(IsNull))` —
    exercises Not wrapping a Unary predicate."""
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
    assert out["id"].to_list() == [1, 3]
    assert out["name"].to_list() == ["a", "c"]


def test_predicate_pushdown_unsupported_falls_back(simple_table):
    """A predicate the translator can't handle (here: a string operation we
    don't translate) should still produce the right rows — polars keeps its
    post-scan filter regardless of whether kernel got pushdown."""
    out = (
        scan_delta(str(simple_table))
        .filter(pl.col("name").str.starts_with("a"))
        .collect()
    )
    assert out["name"].to_list() == ["alice"]


# ---------------------------------------------------------------------------
# Deletion vectors
# ---------------------------------------------------------------------------


def test_delete_via_rewrite(tmp_path):
    """Plain DELETE (no DV protocol) — deltalake rewrites the file. Reads
    should see only the remaining rows."""
    from deltalake import DeltaTable, write_deltalake

    table_path = str(tmp_path / "deleted")
    df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    write_deltalake(table_path, df.to_arrow())

    DeltaTable(table_path).delete("id >= 3")

    out = scan_delta(table_path).collect().sort("id")
    assert out["id"].to_list() == [1, 2]
    assert out["name"].to_list() == ["a", "b"]


def test_deletion_vectors(tmp_path):
    """DV-enabled DELETE — table is configured with
    `delta.enableDeletionVectors=true`, so the delete writes a deletion-vector
    file instead of rewriting parquet. Exercises kernel's selection-vector
    application path and our `EngineData::apply_selection_vector`."""
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
    assert out["id"].to_list() == [1, 3, 5]


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def test_read_delta_eager(simple_table):
    """`read_delta` is the eager equivalent of `scan_delta(...).collect()`."""
    from polars_deltalake import read_delta

    out = read_delta(str(simple_table)).sort("id")
    assert isinstance(out, pl.DataFrame)
    assert out["id"].to_list() == [1, 2, 3, 4, 5]


def test_read_delta_version_arg(tmp_path):
    """`read_delta` honours the `version` kwarg for time travel."""
    from deltalake import write_deltalake

    from polars_deltalake import read_delta

    table_path = str(tmp_path / "ver")
    write_deltalake(table_path, pl.DataFrame({"x": [1]}).to_arrow())
    write_deltalake(table_path, pl.DataFrame({"x": [2]}).to_arrow(), mode="append")

    v0 = read_delta(table_path, version=0)
    assert v0["x"].to_list() == [1]


def test_deletion_vectors_with_projection(tmp_path):
    """DV + projection — verifies that selection vectors apply before the
    polars projection so deleted rows don't leak into the projected output."""
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
    assert out["value"].to_list() == [40, 50, 60]
