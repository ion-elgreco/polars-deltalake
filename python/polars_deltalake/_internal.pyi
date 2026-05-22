"""Type stubs for the native ``_internal`` extension module."""

from __future__ import annotations

import polars as pl

class TableState:
    """Opened table snapshot — kernel snapshot + polars-backed engine."""

    def __init__(
        self,
        uri: str,
        version: int | None = ...,
        storage_options: dict[str, str] | None = ...,
    ) -> None: ...
    def schema(self) -> pl.Schema: ...

class TableScan:
    def __init__(self, state: TableState) -> None: ...
    def configure(
        self,
        with_columns: list[str] | None = ...,
        n_rows: int | None = ...,
        predicate: pl.Expr | None = ...,
    ) -> None:
        """Set projection / row cap / predicate and rewind. Unsupported
        predicate shapes fall through silently — no kernel file-skip."""
        ...
    def next(self) -> pl.DataFrame | None: ...
