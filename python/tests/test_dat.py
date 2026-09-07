"""Reader-conformance tests against the Delta Acceptance Tests (DAT) bundle.

Each case reads `case/delta/` via `read_delta` and compares against the
parquet under `case/expected/latest/table_content/`. Unsupported features
are listed in `_XFAIL_REASONS` so a regression on a passing case stays
distinguishable from a known gap.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from _dat_helper import reader_cases
from polars_deltalake import read_delta

_XFAIL_REASONS: dict[str, str] = {}


_CASES = reader_cases()


@pytest.fixture(params=_CASES, ids=lambda p: p.name)
def case(request: pytest.FixtureRequest) -> Path:
    return request.param


def _read_expected(case: Path) -> pl.DataFrame:
    """Read all parquet shards under `expected/latest/table_content/`."""
    content_dir = case / "expected" / "latest" / "table_content"
    parts = sorted(content_dir.glob("*.parquet"))
    if not parts:
        pytest.fail(
            f"DAT case {case.name!r}: no expected parquet shards under {content_dir}"
        )
    return pl.concat([pl.read_parquet(p) for p in parts])


def test_dat_latest(case: Path):
    # DAT doesn't pin a row order in the expected parquet, so sort both sides.
    if reason := _XFAIL_REASONS.get(case.name):
        pytest.xfail(reason)

    actual = read_delta(str(case / "delta"))
    expected = _read_expected(case)

    sort_cols = expected.columns
    assert_frame_equal(actual.sort(sort_cols), expected.sort(sort_cols))
