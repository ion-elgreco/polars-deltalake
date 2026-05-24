"""Native Polars I/O plugin for Delta Lake, backed by delta-kernel-rs with a polars-io engine."""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
from polars.io.plugins import register_io_source

from polars_deltalake._internal import (
    CdfTableScan,
    CdfTableState,
    TableScan,
    TableState,
)

__all__ = [
    "CdfTableScan",
    "CdfTableState",
    "TableScan",
    "TableState",
    "read_cdf",
    "read_delta",
    "scan_cdf",
    "scan_delta",
]


def read_delta(
    uri: str,
    *,
    version: int | None = None,
    storage_options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Eagerly read a Delta Lake table into a Polars ``DataFrame``.

    Equivalent to ``scan_delta(uri, ...).collect()``. Use ``scan_delta`` for
    lazy evaluation when you need projection / predicate pushdown or to
    chain further lazy operations.
    """
    return scan_delta(uri, version=version, storage_options=storage_options).collect()


def scan_delta(
    uri: str,
    *,
    version: int | None = None,
    storage_options: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """Scan a Delta Lake table into a Polars ``LazyFrame``.

    Native polars scan backed by ``delta-kernel-rs``.

    Args:
        uri: Path or fully qualified URL of the Delta table (``s3://``,
            ``az://``, ``gs://``, ``file://``, or a bare local path).
        version: Optional snapshot version for time travel. Defaults to the
            latest commit.
        storage_options: Cloud credentials forwarded to ``object_store``
            (e.g. ``aws_access_key_id``, ``azure_storage_account_name``).

    Returns:
        A Polars ``LazyFrame`` that streams the table's rows as
        ``pl.DataFrame`` batches when collected.
    """
    table = TableState(uri, version, storage_options)
    schema = table.schema()

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        _batch_size_hint: int | None,
    ) -> Iterator[pl.DataFrame]:
        scan = TableScan(table)
        scan.configure(with_columns, n_rows, predicate)
        while (df := scan.next()) is not None:
            yield df

    return register_io_source(source, schema=schema)


def read_cdf(
    uri: str,
    *,
    start_version: int,
    end_version: int | None = None,
    storage_options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Eagerly read a Delta Lake Change Data Feed into a Polars ``DataFrame``.

    Equivalent to ``scan_cdf(uri, ...).collect()``.
    """
    return scan_cdf(
        uri,
        start_version=start_version,
        end_version=end_version,
        storage_options=storage_options,
    ).collect()


def scan_cdf(
    uri: str,
    *,
    start_version: int,
    end_version: int | None = None,
    storage_options: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """Scan a Delta Lake Change Data Feed into a Polars ``LazyFrame``.

    Streams the Change Data Feed between two commit versions. The returned
    frame carries the table's data columns plus three CDF metadata columns:

    - ``_change_type``: one of ``insert``, ``delete``, ``update_preimage``,
      ``update_postimage``.
    - ``_commit_version``: commit version the change belongs to.
    - ``_commit_timestamp``: commit timestamp (file mtime of the log file).

    CDF must be enabled for the entire requested range
    (``delta.enableChangeDataFeed = true``).

    Args:
        uri: Path or fully qualified URL of the Delta table.
        start_version: First commit version to include (inclusive).
        end_version: Last commit version to include (inclusive). Defaults to
            the latest commit.
        storage_options: Cloud credentials forwarded to ``object_store``.

    Returns:
        A Polars ``LazyFrame``.
    """
    table = CdfTableState(uri, start_version, end_version, storage_options)
    schema = table.schema()

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        _batch_size_hint: int | None,
    ) -> Iterator[pl.DataFrame]:
        scan = CdfTableScan(table)
        scan.configure(with_columns, n_rows, predicate)
        while (df := scan.next()) is not None:
            yield df

    return register_io_source(source, schema=schema)
