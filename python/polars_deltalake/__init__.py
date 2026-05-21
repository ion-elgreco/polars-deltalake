"""Native Polars I/O plugin for Delta Lake, backed by delta-kernel-rs with a polars-io engine."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import polars as pl
from polars.io.plugins import register_io_source

from polars_deltalake._internal import DeltaSource

if TYPE_CHECKING:
    pass

__all__ = ["DeltaSource", "read_delta", "scan_delta"]


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
    # Snapshot the delta log once; reuse the same `DeltaSource` for the
    # schema probe and every io-source invocation. `configure` resets all
    # pushdown state per call so polars can re-collect the LazyFrame without
    # bleeding state across invocations.
    src = DeltaSource(uri, version, storage_options)
    schema = src.schema()

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        _batch_size_hint: int | None,
    ) -> Iterator[pl.DataFrame]:
        src.configure(with_columns, n_rows, predicate)
        while (df := src.next()) is not None:
            yield df

    return register_io_source(source, schema=schema)
