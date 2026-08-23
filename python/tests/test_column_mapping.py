"""Column-mapping name resolution, including nested fields.

The parquet files carry physical (``col-<uuid>``) names; the logical names live
only in the schema metadata, and kernel renames at *every* nesting level.

``deltalake`` cannot write column-mapped tables and the DAT fixture is flat, so
the table comes from delta-spark.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_deltalake import scan_delta

# Re-export so pytest discovers the session-scoped fixture in this module.
from tests._spark_helper import spark_session  # noqa: F401

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def column_mapped_table(
    spark_session: SparkSession,  # noqa: F811
    tmp_path_factory,
) -> str:
    """Enabling column mapping at CREATE time makes delta-spark assign a
    ``col-<uuid>`` physical name to every field, nested ones included."""
    path = str(tmp_path_factory.mktemp("cm") / "nested")
    spark_session.sql(f"""
        CREATE TABLE delta.`{path}` (
            id BIGINT,
            person STRUCT<name: STRING, age: BIGINT>,
            address STRUCT<geo: STRUCT<lat: DOUBLE>>
        ) USING DELTA
        TBLPROPERTIES ('delta.columnMapping.mode' = 'name')
    """)
    spark_session.sql(f"""
        INSERT INTO delta.`{path}` VALUES
            (1,
             named_struct('name', 'alice', 'age', 30L),
             named_struct('geo', named_struct('lat', 1.5))),
            (2,
             named_struct('name', 'bob', 'age', 40L),
             named_struct('geo', named_struct('lat', 2.5)))
    """)
    return path


class TestFixture:
    def test_physical_names_differ_at_every_level(self, column_mapped_table):
        """Without this the suite would pass on an unmapped table."""
        # Column mapping makes delta-spark write under a random prefix dir.
        data = next(Path(column_mapped_table).rglob("*.parquet"))
        frame = pl.read_parquet(data)
        assert all(name.startswith("col-") for name in frame.columns)
        nested = [
            dtype for dtype in frame.schema.values() if isinstance(dtype, pl.Struct)
        ]
        assert nested, "fixture lost its struct columns"
        assert all(f.name.startswith("col-") for dtype in nested for f in dtype.fields)


class TestTopLevelNames:
    """The part that already works, kept as a guard."""

    def test_top_level_columns_use_logical_names(self, column_mapped_table):
        out = scan_delta(column_mapped_table).collect()
        assert out.columns == ["id", "person", "address"]

    def test_top_level_values(self, column_mapped_table):
        out = scan_delta(column_mapped_table).collect().sort("id")
        assert out["id"].to_list() == [1, 2]


class TestNestedNames:
    def test_lazy_schema_matches_collected_schema(self, column_mapped_table):
        """The LazyFrame must not advertise a schema its data does not have."""
        lf = scan_delta(column_mapped_table)
        assert dict(lf.collect_schema()) == dict(lf.collect().schema)

    def test_nested_fields_use_logical_names(self, column_mapped_table):
        person = scan_delta(column_mapped_table).collect().schema["person"]
        assert isinstance(person, pl.Struct)
        assert [f.name for f in person.fields] == ["name", "age"]

    def test_doubly_nested_fields_use_logical_names(self, column_mapped_table):
        address = scan_delta(column_mapped_table).collect().schema["address"]
        assert isinstance(address, pl.Struct)
        geo = address.fields[0]
        assert geo.name == "geo"
        assert isinstance(geo.dtype, pl.Struct)
        assert [f.name for f in geo.dtype.fields] == ["lat"]

    def test_nested_values_survive_the_rename(self, column_mapped_table):
        """A cast-based rename would produce this schema with all-null values."""
        out = scan_delta(column_mapped_table).collect().sort("id")
        assert out["person"].to_list() == [
            {"name": "alice", "age": 30},
            {"name": "bob", "age": 40},
        ]
        assert out["address"].to_list() == [
            {"geo": {"lat": 1.5}},
            {"geo": {"lat": 2.5}},
        ]

    def test_select_nested_logical_field(self, column_mapped_table):
        out = (
            scan_delta(column_mapped_table)
            .select(pl.col("person").struct.field("name").alias("name"))
            .collect()
        )
        assert sorted(out["name"].to_list()) == ["alice", "bob"]

    def test_filter_on_nested_logical_field(self, column_mapped_table):
        out = (
            scan_delta(column_mapped_table)
            .filter(pl.col("person").struct.field("name") == "alice")
            .collect()
        )
        assert out.height == 1
        assert out["id"].to_list() == [1]
