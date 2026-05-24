"""CDF tests against tables authored with PySpark + delta-spark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_deltalake import read_cdf
from tests._cdf_helpers import drop_ts

# Re-export so pytest discovers the session-scoped fixture in this module.
from tests._spark_helper import spark_session  # noqa: F401

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


pytestmark = pytest.mark.spark


def _build_column_mapped_cdf_table(spark: SparkSession, path: str) -> None:
    """3 commits on a CDF + column-mapping=name table: insert, update,
    delete. Same shape as the deltalake-py fixture in ``test_cdf.py`` so
    the asserts compare apples-to-apples."""
    spark.sql(f"""
        CREATE TABLE delta.`{path}` (id INT, name STRING) USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true',
            'delta.columnMapping.mode' = 'name',
            'delta.minReaderVersion' = '2',
            'delta.minWriterVersion' = '5'
        )
    """)
    spark.sql(f"INSERT INTO delta.`{path}` VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    spark.sql(f"UPDATE delta.`{path}` SET name = 'B' WHERE id = 2")
    spark.sql(f"DELETE FROM delta.`{path}` WHERE id = 3")


class TestColumnMappedCdf:
    @pytest.fixture(scope="class")
    def column_mapped_cdf_table(
        self,
        spark_session: SparkSession,  # noqa: F811
        tmp_path_factory: pytest.TempPathFactory,
    ) -> str:
        path = str(tmp_path_factory.mktemp("cm_cdf"))
        _build_column_mapped_cdf_table(spark_session, path)
        return path

    def test_full_range(self, column_mapped_cdf_table):
        out = drop_ts(
            read_cdf(column_mapped_cdf_table, start_version=1).sort(
                "_commit_version", "_change_type", "id"
            )
        )
        # v0 is the CREATE (no data); v1 inserts, v2 updates id=2, v3 deletes id=3.
        expected = pl.DataFrame(
            {
                "id": [1, 2, 3, 2, 2, 3],
                "name": ["a", "b", "c", "B", "b", "c"],
                "_change_type": [
                    "insert",
                    "insert",
                    "insert",
                    "update_postimage",
                    "update_preimage",
                    "delete",
                ],
                "_commit_version": [1, 1, 1, 2, 2, 3],
            },
            schema={
                "id": pl.Int32,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)

    def test_projection_with_logical_name(self, column_mapped_cdf_table):
        # `name`'s physical name is `col-<uuid>` in the parquet — projecting
        # by the logical name must still work.
        out = (
            read_cdf(column_mapped_cdf_table, start_version=1)
            .select("name", "_change_type")
            .sort("_change_type", "name")
        )
        expected = pl.DataFrame(
            {
                "name": ["c", "a", "b", "c", "B", "b"],
                "_change_type": [
                    "delete",
                    "insert",
                    "insert",
                    "insert",
                    "update_postimage",
                    "update_preimage",
                ],
            },
            schema={"name": pl.String, "_change_type": pl.String},
        )
        assert_frame_equal(out, expected)

    def test_filter_on_logical_name_column(self, column_mapped_cdf_table):
        # Predicates use logical names; the translator must rewrite them
        # to physical names before pushdown.
        out = drop_ts(
            read_cdf(column_mapped_cdf_table, start_version=1)
            .filter(pl.col("id") == 2)
            .sort("_commit_version", "_change_type")
        )
        expected = pl.DataFrame(
            {
                "id": [2, 2, 2],
                "name": ["b", "B", "b"],
                "_change_type": ["insert", "update_postimage", "update_preimage"],
                "_commit_version": [1, 2, 2],
            },
            schema={
                "id": pl.Int32,
                "name": pl.String,
                "_change_type": pl.String,
                "_commit_version": pl.Int64,
            },
        )
        assert_frame_equal(out, expected)
