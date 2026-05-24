"""Shared helpers for CDF tests."""

from __future__ import annotations

import polars as pl


def drop_ts(df: pl.DataFrame) -> pl.DataFrame:
    """Strip the non-deterministic ``_commit_timestamp`` column (it's the
    log-file mtime, varies per run). Caller asserts schema presence
    separately when relevant."""
    return df.drop("_commit_timestamp") if "_commit_timestamp" in df.columns else df
