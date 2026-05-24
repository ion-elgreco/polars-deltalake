"""Spark + delta-spark bootstrap for fixtures ``deltalake`` can't write
(column mapping, complex CDF shapes). Requires JDK 17/21 for spark 4+"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def build_spark_session() -> SparkSession:
    """Create a SparkSession configured for Delta Lake.

    Pulls ``io.delta:delta-spark`` from Maven on first call (cached in
    ``~/.ivy2``). Single local worker — fixtures are small.
    """
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master("local[1]")
        .appName("polars-deltalake-tests")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.showConsoleProgress", "false")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


@pytest.fixture(scope="session")
def spark_session() -> Iterator[SparkSession]:
    """Session-scoped SparkSession. Skips the test if PySpark can't start."""
    try:
        from pyspark.sql import SparkSession  # noqa: F401
    except ImportError:
        pytest.skip("pyspark not installed (install the `spark` dep group)")
    try:
        session = build_spark_session()
    except Exception as e:
        pytest.skip(f"SparkSession failed to start (no compatible JDK?): {e}")
    yield session
    session.stop()
