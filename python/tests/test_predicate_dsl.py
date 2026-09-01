from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from deltalake import write_deltalake
from polars.testing import assert_frame_equal

from polars_deltalake import scan_delta

_NUMERIC_DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "i": [-3, -1, 0, 2, 4, 8],
        "f": [-1.5, -0.5, 0.0, 0.25, 1.5, 3.14],
        "p": [0.1, 0.5, 1.0, 2.5, 10.0, 100.0],  # positives for log
    }
)

_STRING_DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "s": ["abc", "  hello  ", "FOO123", "BAR-42"],
        "json": [
            '{"x": 1}',
            '{"x": 2}',
            '{"x": 3}',
            '{"x": 4}',
        ],
    }
)

_DATE_DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "d": [
            date(2023, 1, 15),
            date(2023, 6, 30),
            date(2024, 3, 10),
            date(2024, 12, 31),
        ],
    }
)


@pytest.fixture
def numeric_table(tmp_path: Path) -> Path:
    path = tmp_path / "num"
    write_deltalake(str(path), _NUMERIC_DF.to_arrow())
    return path


@pytest.fixture
def string_table(tmp_path: Path) -> Path:
    path = tmp_path / "str"
    write_deltalake(str(path), _STRING_DF.to_arrow())
    return path


@pytest.fixture
def date_table(tmp_path: Path) -> Path:
    path = tmp_path / "dates"
    write_deltalake(str(path), _DATE_DF.to_arrow())
    return path


def _assert_rows(out: pl.DataFrame, source: pl.DataFrame, ids: list[int]) -> None:
    assert_frame_equal(out.sort("id"), source.filter(pl.col("id").is_in(ids)))


class TestTemporal:
    def test_year(self, date_table):
        out = (
            scan_delta(str(date_table)).filter(pl.col("d").dt.year() == 2024).collect()
        )
        _assert_rows(out, _DATE_DF, [3, 4])

    def test_month(self, date_table):
        out = scan_delta(str(date_table)).filter(pl.col("d").dt.month() >= 6).collect()
        _assert_rows(out, _DATE_DF, [2, 4])


class TestBitwise:
    def test_and(self, numeric_table):
        """`&` on integer Series is bitwise AND under the `bitwise` feature."""
        out = scan_delta(str(numeric_table)).filter((pl.col("i") & 1) != 0).collect()
        # i values: -3, -1, 0, 2, 4, 8 → odd ones are -3 (id=1), -1 (id=2)
        _assert_rows(out, _NUMERIC_DF, [1, 2])


class TestMath:
    def test_abs(self, numeric_table):
        out = scan_delta(str(numeric_table)).filter(pl.col("i").abs() >= 3).collect()
        _assert_rows(out, _NUMERIC_DF, [1, 5, 6])

    def test_log(self, numeric_table):
        """log(1.0) = 0 — strict gt → rows with p > 1."""
        out = scan_delta(str(numeric_table)).filter(pl.col("p").log() > 0).collect()
        _assert_rows(out, _NUMERIC_DF, [4, 5, 6])

    def test_trigonometry(self, numeric_table):
        """sin(-1.5)=-0.997, sin(-0.5)=-0.479, sin(0)=0, sin(0.25)=0.247,
        sin(1.5)=0.997, sin(3.14)≈0 → |sin|<0.5 → ids 2, 3, 4, 6."""
        out = (
            scan_delta(str(numeric_table))
            .filter(pl.col("f").sin().abs() < 0.5)
            .collect()
        )
        _assert_rows(out, _NUMERIC_DF, [2, 3, 4, 6])

    def test_sign(self, numeric_table):
        out = scan_delta(str(numeric_table)).filter(pl.col("f").sign() < 0).collect()
        _assert_rows(out, _NUMERIC_DF, [1, 2])

    def test_round(self, numeric_table):
        """round(-0.5) = 0 (bankers rounding); round(0)=0; round(0.25)=0."""
        out = scan_delta(str(numeric_table)).filter(pl.col("f").round(0) == 0).collect()
        _assert_rows(out, _NUMERIC_DF, [2, 3, 4])


class TestDistinctness:
    def test_is_close(self, numeric_table):
        out = (
            scan_delta(str(numeric_table))
            .filter(pl.col("p").is_close(1.0, abs_tol=0.01))
            .collect()
        )
        _assert_rows(out, _NUMERIC_DF, [3])

    def test_is_first_distinct(self, numeric_table):
        """All numeric values are unique here, so is_first_distinct → all true."""
        out = (
            scan_delta(str(numeric_table))
            .filter(pl.col("id").is_first_distinct())
            .collect()
        )
        _assert_rows(out, _NUMERIC_DF, [1, 2, 3, 4, 5, 6])

    def test_is_last_distinct(self, numeric_table):
        out = (
            scan_delta(str(numeric_table))
            .filter(pl.col("id").is_last_distinct())
            .collect()
        )
        _assert_rows(out, _NUMERIC_DF, [1, 2, 3, 4, 5, 6])


class TestStringOps:
    def test_concat_str(self, string_table):
        out = (
            scan_delta(str(string_table))
            .filter(pl.concat_str([pl.col("s"), pl.lit("!")]).str.contains("FOO"))
            .collect()
        )
        _assert_rows(out, _STRING_DF, [3])

    def test_pad_start(self, string_table):
        """`"abc"→"*****abc"`, `"FOO123"→"**FOO123"`, `"BAR-42"→"**BAR-42"` all
        start with `"*"`; `"  hello  "` is already 9 chars so no padding."""
        out = (
            scan_delta(str(string_table))
            .filter(pl.col("s").str.pad_start(8, "*").str.starts_with("*"))
            .collect()
        )
        _assert_rows(out, _STRING_DF, [1, 3, 4])

    def test_normalize(self, string_table):
        """NFC of plain ASCII is idempotent; passes for every row."""
        out = (
            scan_delta(str(string_table))
            .filter(pl.col("s").str.normalize("NFC") == pl.col("s"))
            .collect()
        )
        _assert_rows(out, _STRING_DF, [1, 2, 3, 4])

    def test_reverse(self, string_table):
        out = (
            scan_delta(str(string_table))
            .filter(pl.col("s").str.reverse() == "cba")
            .collect()
        )
        _assert_rows(out, _STRING_DF, [1])

    def test_to_integer(self, string_table):
        """Extract trailing digits with regex; cast to int; filter on it."""
        out = (
            scan_delta(str(string_table))
            .filter(
                pl.col("s").str.extract(r"(\d+)").str.to_integer(strict=False) >= 42
            )
            .collect()
        )
        _assert_rows(out, _STRING_DF, [3, 4])  # FOO123, BAR-42

    def test_extract_groups(self, string_table):
        out = (
            scan_delta(str(string_table))
            .filter(
                pl.col("s")
                .str.extract_groups(r"(?<prefix>[A-Z]+)(?<num>\d+)")
                .struct.field("prefix")
                == "FOO"
            )
            .collect()
        )
        _assert_rows(out, _STRING_DF, [3])

    def test_contains_any(self, string_table):
        out = (
            scan_delta(str(string_table))
            .filter(pl.col("s").str.contains_any(["FOO", "BAR"]))
            .collect()
        )
        _assert_rows(out, _STRING_DF, [3, 4])


class TestJsonPath:
    def test_extract(self, string_table):
        out = (
            scan_delta(str(string_table))
            .filter(pl.col("json").str.json_path_match("$.x").cast(pl.Int64) >= 3)
            .collect()
        )
        _assert_rows(out, _STRING_DF, [3, 4])
