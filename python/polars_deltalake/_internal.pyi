"""Type stubs for the native ``_internal`` extension module."""

from __future__ import annotations

import polars as pl

class DeltaSource:
    """Drives a delta-kernel scan against the polars-backed engine.

    A single source can be reused across ``register_io_source`` invocations:
    construction snapshots the delta log once, then ``configure`` resets the
    per-call pushdown state and rewinds the scan iterator.
    """

    def __init__(
        self,
        uri: str,
        version: int | None = ...,
        storage_options: dict[str, str] | None = ...,
    ) -> None: ...
    def schema(self) -> pl.Schema:
        """Logical schema of the table, as a polars ``Schema``."""
        ...
    def configure(
        self,
        with_columns: list[str] | None = ...,
        n_rows: int | None = ...,
        predicate: pl.Expr | None = ...,
    ) -> None:
        """Apply per-call pushdown atomically and rewind. ``predicate`` acts
        as a kernel data-skipping hint over file stats; row filtering still
        has to run on the polars side. Predicate translation failures fall
        through silently (scan reads every file)."""
        ...
    def next(self) -> pl.DataFrame | None:
        """Return the next ``DataFrame`` batch, or ``None`` when the scan is done."""
        ...
