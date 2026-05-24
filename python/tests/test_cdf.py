"""Roundtrip tests for `polars_deltalake.scan_cdf` / `read_cdf` against
``deltalake``-authored CDF tables."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from polars_deltalake import CdfTableState, read_cdf, scan_cdf
from tests._cdf_helpers import drop_ts

# Polars dtype for kernel `_commit_timestamp` (file mtime as UTC µs).
CDF_TS_DTYPE = pl.Datetime("us", "UTC")


@pytest.fixture
def cdf_table(tmp_path):
    """Three commits: initial insert, update one row, delete one row."""
    from deltalake import DeltaTable, write_deltalake

    table_path = str(tmp_path / "cdf_tbl")
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    write_deltalake(
        table_path,
        df.to_arrow(),
        configuration={"delta.enableChangeDataFeed": "true"},
    )
    dt = DeltaTable(table_path)
    dt.update(predicate="id = 2", updates={"name": "'B'"})
    dt.delete(predicate="id = 3")
    return table_path


@pytest.fixture
def partitioned_cdf_table(tmp_path):
    """CDF-enabled partitioned table — initial insert + an update."""
    from deltalake import DeltaTable, write_deltalake

    table_path = str(tmp_path / "cdf_part")
    df = pl.DataFrame(
        {
            "region": ["eu", "eu", "us", "us"],
            "id": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
        }
    )
    write_deltalake(
        table_path,
        df.to_arrow(),
        partition_by=["region"],
        configuration={"delta.enableChangeDataFeed": "true"},
    )
    dt = DeltaTable(table_path)
    dt.update(predicate="id = 3", updates={"value": "300"})
    return table_path


class TestBasicCdf:
    def test_returns_lazyframe(self, cdf_table):
        lf = scan_cdf(cdf_table, start_version=0)
        assert isinstance(lf, pl.LazyFrame)

    def test_schema_carries_metadata_columns(self, cdf_table):
        expected = pl.Schema(
            {
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
                "_commit_timestamp": CDF_TS_DTYPE,
            }
        )
        assert scan_cdf(cdf_table, start_version=0).collect_schema() == expected

    def test_full_range(self, cdf_table):
        out = drop_ts(
            read_cdf(cdf_table, start_version=0).sort(
                "_commit_version", "_change_type", "id"
            )
        )
        expected = pl.DataFrame(
            {
                "id": [1, 2, 3, 3, 2, 2],
                "name": ["a", "b", "c", "c", "B", "b"],
                "_change_type": [
                    "insert",
                    "insert",
                    "insert",
                    "delete",
                    "update_postimage",
                    "update_preimage",
                ],
                "_commit_version": [0, 0, 0, 2, 1, 1],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        ).sort("_commit_version", "_change_type", "id")
        assert_frame_equal(out, expected)
        ts_carrier = read_cdf(cdf_table, start_version=0)
        assert ts_carrier.schema["_commit_timestamp"] == CDF_TS_DTYPE

    def test_update_preimage_postimage_values(self, cdf_table):
        out = drop_ts(
            read_cdf(cdf_table, start_version=0)
            .filter(pl.col("id") == 2)
            .sort("_commit_version", "_change_type")
        )
        expected = pl.DataFrame(
            {
                "id": [2, 2, 2],
                "name": ["b", "B", "b"],
                "_change_type": ["insert", "update_postimage", "update_preimage"],
                "_commit_version": [0, 1, 1],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)


class TestVersionRange:
    def test_explicit_end_version(self, cdf_table):
        out = drop_ts(read_cdf(cdf_table, start_version=0, end_version=0).sort("id"))
        expected = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "_change_type": ["insert", "insert", "insert"],
                "_commit_version": [0, 0, 0],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)

    def test_mid_range(self, cdf_table):
        out = drop_ts(
            read_cdf(cdf_table, start_version=1, end_version=1).sort("_change_type")
        )
        expected = pl.DataFrame(
            {
                "id": [2, 2],
                "name": ["B", "b"],
                "_change_type": ["update_postimage", "update_preimage"],
                "_commit_version": [1, 1],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)

    def test_start_only_runs_to_latest(self, cdf_table):
        out = drop_ts(read_cdf(cdf_table, start_version=2))
        expected = pl.DataFrame(
            {
                "id": [3],
                "name": ["c"],
                "_change_type": ["delete"],
                "_commit_version": [2],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)

    @pytest.mark.parametrize("engine", ["auto", "in-memory", "streaming"])
    def test_engine_arg(self, cdf_table, engine):
        """`read_cdf` forwards `engine` to `LazyFrame.collect`."""
        out = read_cdf(cdf_table, start_version=0, engine=engine)
        assert isinstance(out, pl.DataFrame)
        assert out.height == 6


class TestProjection:
    def test_projection_pushdown(self, cdf_table):
        out = (
            scan_cdf(cdf_table, start_version=0)
            .select("id", "_change_type")
            .collect()
            .sort("_change_type", "id")
        )
        expected = pl.DataFrame(
            {
                "id": [3, 1, 2, 3, 2, 2],
                "_change_type": [
                    "delete",
                    "insert",
                    "insert",
                    "insert",
                    "update_postimage",
                    "update_preimage",
                ],
            },
            schema={"id": pl.Int64, "_change_type": pl.String},
        )
        assert_frame_equal(out, expected)

    def test_projection_only_metadata(self, cdf_table):
        # Exercises the empty-physical-schema row-count probe in the
        # parquet handler
        out = (
            scan_cdf(cdf_table, start_version=0)
            .select("_commit_version", "_change_type")
            .collect()
            .sort("_commit_version", "_change_type")
        )
        expected = pl.DataFrame(
            {
                "_commit_version": [0, 0, 0, 1, 1, 2],
                "_change_type": [
                    "insert",
                    "insert",
                    "insert",
                    "update_postimage",
                    "update_preimage",
                    "delete",
                ],
            },
            schema={"_commit_version": pl.Int64, "_change_type": pl.String},
        )
        assert_frame_equal(out, expected)


class TestPredicatePushdown:
    def test_filter_on_data_column(self, cdf_table):
        out = drop_ts(
            scan_cdf(cdf_table, start_version=0)
            .filter(pl.col("id") == 2)
            .collect()
            .sort("_commit_version", "_change_type")
        )
        expected = pl.DataFrame(
            {
                "id": [2, 2, 2],
                "name": ["b", "B", "b"],
                "_change_type": ["insert", "update_postimage", "update_preimage"],
                "_commit_version": [0, 1, 1],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)

    def test_filter_on_change_type(self, cdf_table):
        # Predicates on CDF metadata cols don't push to kernel but must
        # still filter correctly polars-side.
        out = drop_ts(
            scan_cdf(cdf_table, start_version=0)
            .filter(pl.col("_change_type") == "delete")
            .collect()
        )
        expected = pl.DataFrame(
            {
                "id": [3],
                "name": ["c"],
                "_change_type": ["delete"],
                "_commit_version": [2],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)

    def test_filter_combined(self, cdf_table):
        out = drop_ts(
            scan_cdf(cdf_table, start_version=0)
            .filter(
                (pl.col("id") == 2) & (pl.col("_change_type") == "update_postimage")
            )
            .collect()
        )
        expected = pl.DataFrame(
            {
                "id": [2],
                "name": ["B"],
                "_change_type": ["update_postimage"],
                "_commit_version": [1],
            },
            schema={
                "id": pl.Int64,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)


class TestSlicePushdown:
    def test_head(self, cdf_table):
        out = scan_cdf(cdf_table, start_version=0).head(2).collect()
        assert out.height == 2


class TestPartitioned:
    def test_partitioned_full_cdf(self, partitioned_cdf_table):
        out = drop_ts(
            read_cdf(partitioned_cdf_table, start_version=0).sort(
                "_commit_version", "_change_type", "id"
            )
        )
        expected = pl.DataFrame(
            {
                "region": ["eu", "eu", "us", "us", "us", "us"],
                "id": [1, 2, 3, 4, 3, 3],
                "value": [10, 20, 30, 40, 300, 30],
                "_change_type": [
                    "insert",
                    "insert",
                    "insert",
                    "insert",
                    "update_postimage",
                    "update_preimage",
                ],
                "_commit_version": [0, 0, 0, 0, 1, 1],
            },
            schema={
                "region": pl.String,
                "id": pl.Int64,
                "value": pl.Int64,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)


class TestCdfTableState:
    def test_start_and_end_version_accessors(self, cdf_table):
        state = CdfTableState(cdf_table, 0, 1)
        assert state.start_version() == 0
        assert state.end_version() == 1

    def test_end_version_defaults_to_latest(self, cdf_table):
        state = CdfTableState(cdf_table, 0)
        assert state.end_version() == 2

    def test_schema_matches_lazyframe(self, cdf_table):
        state = CdfTableState(cdf_table, 0)
        # pyo3-polars surfaces the schema as a plain {name: dtype} dict.
        assert_series_equal(
            pl.Series("cols", list(state.schema())),
            pl.Series(
                "cols",
                ["id", "name", "_change_type", "_commit_version", "_commit_timestamp"],
            ),
        )


class TestErrors:
    def test_cdf_disabled_table_errors(self, tmp_path):
        from deltalake import write_deltalake

        table_path = str(tmp_path / "no_cdf")
        write_deltalake(table_path, pl.DataFrame({"id": [1, 2]}).to_arrow())
        with pytest.raises(RuntimeError):
            read_cdf(table_path, start_version=0)
