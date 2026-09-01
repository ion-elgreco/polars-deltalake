"""Scan tests against tables authored with PySpark + delta-spark."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_deltalake import scan_delta

# Re-export so pytest discovers the session-scoped fixture in this module.
from tests._spark_helper import spark_session  # noqa: F401

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


pytestmark = pytest.mark.spark


def test_int96_timestamps(
    spark_session: SparkSession,  # noqa: F811
    tmp_path,
):
    """Verifies if INT96 is downcasted to 'us'"""
    path = str(tmp_path / "int96")
    spark_session.conf.set("spark.sql.session.timeZone", "UTC")
    spark_session.conf.set("spark.sql.parquet.outputTimestampType", "INT96")

    spark_session.sql(f"CREATE TABLE delta.`{path}` (id INT, ts TIMESTAMP) USING DELTA")
    spark_session.sql(f"""
        INSERT INTO delta.`{path}` VALUES
            (1, TIMESTAMP '2024-01-15 12:30:45.123456'),
            (2, TIMESTAMP '1999-12-31 23:59:59.999999')
    """)

    out = scan_delta(path).collect().sort("id")

    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "id": [1, 2],
                "ts": [
                    datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc),
                    datetime(1999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
                ],
            },
            schema={"id": pl.Int32, "ts": pl.Datetime("us", "UTC")},
        ),
    )
