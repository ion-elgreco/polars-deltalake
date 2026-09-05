"""Scan tests against tables authored with PySpark + delta-spark."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

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


def test_dv_predicate_pushdown_across_row_groups(
    spark_session: SparkSession,  # noqa: F811
    tmp_path,
):
    """Spark writes real deletion vectors (`deltalake` cannot). Small row
    groups let a selective predicate skip most of each file, so the keep-mask
    has to address rows by physical position, not by batch order. Every
    deleted row must stay deleted and a pushed predicate must match the same
    filter applied after a full scan."""
    import json

    from pyspark.sql import functions as F

    import pyarrow.parquet as pq

    path = str(tmp_path / "dv_rg")
    # Small row groups so a selective predicate can skip most of each file.
    # Only the JVM-side Hadoop conf reaches the parquet writer.
    hadoop_conf = cast(Any, spark_session.sparkContext._jsc).hadoopConfiguration()
    hadoop_conf.set("parquet.block.size", str(64 * 1024))
    try:
        df = spark_session.range(0, 200_000, 1, 2).withColumn("v", F.col("id") % 97)
        df.write.format("delta").save(path)
        spark_session.sql(
            f"ALTER TABLE delta.`{path}` "
            "SET TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')"
        )
        spark_session.sql(f"DELETE FROM delta.`{path}` WHERE id % 7 = 0")
    finally:
        hadoop_conf.unset("parquet.block.size")

    data_files = list((tmp_path / "dv_rg").glob("*.parquet"))
    assert data_files and all(
        pq.ParquetFile(f).metadata.num_row_groups > 4 for f in data_files
    ), "files came out as one row group; nothing for the predicate to skip"

    last_commit = sorted((tmp_path / "dv_rg" / "_delta_log").glob("*.json"))[-1]
    adds = [json.loads(line) for line in last_commit.read_text().splitlines()]
    assert any(a.get("add", {}).get("deletionVector") for a in adds), (
        "Spark rewrote the files instead of writing a deletion vector"
    )

    full = scan_delta(path).collect()
    assert full.height == 200_000 - len(range(0, 200_000, 7))
    assert full.filter(pl.col("id") % 7 == 0).height == 0

    for pred in [
        pl.col("id") > 199_000,
        pl.col("id").is_between(100_000, 100_100),
        pl.col("v") == 3,
        pl.col("id") == 7,
        pl.col("id") == 8,
    ]:
        pushed = scan_delta(path).filter(pred).collect().sort("id")
        assert_frame_equal(pushed, full.filter(pred).sort("id"))
