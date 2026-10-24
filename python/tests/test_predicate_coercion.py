"""Predicates Spark accepts that polars alone would refuse or mis-evaluate:
naive timestamp literals against a UTC column, and nested fields a file
predates (DAT v0.0.8-preview `data_skipping_*`, `schema_evolution_*`)."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_deltalake import scan_delta


@pytest.fixture
def ts_table(tmp_path):
    from deltalake import write_deltalake

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "ts": [datetime(2024, 1, 1), datetime(2024, 3, 1), datetime(2024, 6, 1)],
        }
    ).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    for row in df.iter_slices(1):
        write_deltalake(str(tmp_path / "ts"), row.to_arrow(), mode="append")
    return str(tmp_path / "ts")


class TestNaiveTimestampLiteral:
    def test_comparison(self, ts_table):
        out = scan_delta(ts_table).filter(pl.col("ts") < datetime(2024, 3, 1)).collect()
        assert out["id"].to_list() == [1]

    def test_sql_timestamp_literal(self, ts_table):
        out = (
            scan_delta(ts_table)
            .filter(pl.sql_expr("ts >= TIMESTAMP'2024-03-01 00:00:00'"))
            .collect()
            .sort("id")
        )
        assert out["id"].to_list() == [2, 3]

    def test_is_between_and_is_in(self, ts_table):
        between = (
            scan_delta(ts_table)
            .filter(pl.col("ts").is_between(datetime(2024, 2, 1), datetime(2024, 4, 1)))
            .collect()
        )
        assert between["id"].to_list() == [2]
        in_set = (
            scan_delta(ts_table)
            .filter(pl.col("ts").is_in([datetime(2024, 1, 1), datetime(2024, 6, 1)]))
            .collect()
            .sort("id")
        )
        assert in_set["id"].to_list() == [1, 3]

    def test_aware_literal_is_unchanged(self, ts_table):
        out = (
            scan_delta(ts_table)
            .filter(pl.col("ts") < datetime(2024, 3, 1, tzinfo=timezone.utc))
            .collect()
        )
        assert out["id"].to_list() == [1]


class TestNestedFieldAddedLater:
    @pytest.fixture
    def evolved(self, tmp_path):
        from deltalake import write_deltalake

        path = str(tmp_path / "evolved")
        v0 = pl.DataFrame({"id": [1, 2], "info": [{"name": "a"}, {"name": "b"}]})
        write_deltalake(path, v0.to_arrow())
        v1 = pl.DataFrame(
            {"id": [3], "info": [{"name": "c", "email": "c@x"}]},
            schema={"id": pl.Int64, "info": pl.Struct({"name": pl.String, "email": pl.String})},
        )
        write_deltalake(path, v1.to_arrow(), mode="append", schema_mode="merge")
        return path

    def test_filter_on_field_older_files_lack(self, evolved):
        lf = scan_delta(evolved)
        present = lf.filter(pl.col("info").struct.field("email").is_not_null()).collect()
        assert present["id"].to_list() == [3]
        absent = lf.filter(pl.col("info").struct.field("email").is_null()).collect().sort("id")
        assert absent["id"].to_list() == [1, 2]
        assert_frame_equal(
            absent.select(pl.col("info").struct.field("email")),
            pl.DataFrame({"email": [None, None]}, schema={"email": pl.String}),
        )
