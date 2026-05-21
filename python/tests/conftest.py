"""Shared test helpers."""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
from polars.io.plugins import register_io_source

from polars_deltalake import DeltaSource


def scan_delta_unfiltered(
    uri: str,
    *,
    version: int | None = None,
    storage_options: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """Same as `polars_deltalake.scan_delta` but **without** the
    Python-side `df.filter(predicate)` correctness backstop.

    Tests using this helper assert exact rows after only the Rust-side
    pushdown — if any rows survive that shouldn't, the pushdown is
    incomplete (kernel didn't prune, polars-io didn't filter, or our
    partition skip missed). With the backstop in place those bugs would
    be silently corrected.
    """
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
