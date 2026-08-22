"""Partition-value parsing contract tests.

Delta serializes partition values as strings in the commit log, and the
kernel's `MapToStruct` contract fixes how they parse back: a missing key or
the empty string yields null, and anything else that cannot be parsed fails
the scan instead of silently becoming null.

Tables are written with `deltalake` and read back through `scan_delta`. A
conforming writer never emits the values the contract cases need, so those
are injected by rewriting the committed `partitionValues` map in place.
"""

from __future__ import annotations

import datetime
import decimal
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl
import pytest

from polars_deltalake import scan_delta

UTC = datetime.timezone.utc

_DROP = object()


def _write_partitioned(path: Path, values: list[Any], dtype: pl.DataType) -> str:
    """One commit, one add action per distinct partition value."""
    from deltalake import write_deltalake

    df = pl.DataFrame(
        {"p": values, "v": list(range(len(values)))},
        schema_overrides={"p": dtype},
    )
    write_deltalake(str(path), df.to_arrow(), partition_by=["p"])
    return str(path)


def _rewrite_partition_values(
    table_path: str,
    rewrite: Callable[[Any], Any],
    version: int = 0,
) -> None:
    """Replace the committed partition value of column `p` via `rewrite`.

    `rewrite` receives the value `deltalake` wrote and returns the value to
    commit instead; returning `_DROP` removes the key entirely.
    """
    log = Path(table_path) / "_delta_log" / f"{version:020d}.json"
    lines = []
    for line in log.read_text().splitlines():
        action = json.loads(line)
        add = action.get("add")
        if add is not None and "p" in add["partitionValues"]:
            new = rewrite(add["partitionValues"]["p"])
            if new is _DROP:
                del add["partitionValues"]["p"]
            else:
                add["partitionValues"]["p"] = new
        lines.append(json.dumps(action))
    log.write_text("\n".join(lines) + "\n")


def _read_p(table_path: str) -> list[Any]:
    return scan_delta(table_path).collect().sort("v")["p"].to_list()


class TestRoundtrip:
    """Every partition dtype `deltalake` can write survives a roundtrip."""

    @pytest.mark.parametrize(
        ("dtype", "value"),
        [
            (pl.String(), "a"),
            (pl.Int32(), 7),
            (pl.Int64(), 2**40),
            (pl.Boolean(), True),
            (pl.Date(), datetime.date(2021, 1, 2)),
            (
                pl.Datetime("us", "UTC"),
                datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC),
            ),
            (pl.Datetime("us"), datetime.datetime(2021, 1, 2, 8, 45)),
            (pl.Decimal(10, 2), decimal.Decimal("12.34")),
        ],
        ids=[
            "string",
            "int",
            "long",
            "boolean",
            "date",
            "timestamp",
            "timestamp_ntz",
            "decimal",
        ],
    )
    def test_dtype_roundtrip(self, tmp_path, dtype, value):
        table = _write_partitioned(tmp_path / "t", [value], dtype)
        out = scan_delta(table).collect()
        assert out["p"].to_list() == [value]
        assert out.schema["p"] == dtype

    def test_multi_value_partition_roundtrip(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [3, 1, 2], pl.Int32())
        assert sorted(_read_p(table)) == [1, 2, 3]


class TestNullProducingValues:
    """The contract's only null results: a missing key and the empty string."""

    def test_missing_key_is_null(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [7], pl.Int32())
        _rewrite_partition_values(table, lambda _: _DROP)
        assert _read_p(table) == [None]

    def test_json_null_is_null(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [7], pl.Int32())
        _rewrite_partition_values(table, lambda _: None)
        assert _read_p(table) == [None]

    def test_empty_string_is_null_for_non_string(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [7], pl.Int32())
        _rewrite_partition_values(table, lambda _: "")
        assert _read_p(table) == [None]

    def test_empty_string_is_itself_for_string(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", ["a"], pl.String())
        _rewrite_partition_values(table, lambda _: "")
        assert _read_p(table) == [""]


class TestUnparseableValueFails:
    """An unparseable value is a broken table: fail loudly, never silently."""

    @pytest.mark.parametrize(
        ("dtype", "value", "garbage"),
        [
            (pl.Int32(), 7, "12x"),
            (pl.Int64(), 2**40, "not-a-number"),
            (pl.Boolean(), True, "yes"),
            (pl.Date(), datetime.date(2021, 1, 2), "20x1-01-02"),
            (
                pl.Datetime("us", "UTC"),
                datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC),
                "not-a-time",
            ),
            (pl.Decimal(10, 2), decimal.Decimal("12.34"), "12.3.4"),
        ],
        ids=["int", "long", "boolean", "date", "timestamp", "decimal"],
    )
    def test_garbage_value_raises(self, tmp_path, dtype, value, garbage):
        table = _write_partitioned(tmp_path / "t", [value], dtype)
        _rewrite_partition_values(table, lambda _: garbage)
        with pytest.raises(Exception, match=garbage.replace(".", r"\.")):
            scan_delta(table).collect()

    def test_garbage_value_raises_instead_of_dropping_the_file(self, tmp_path):
        """A null parse would make the pruning predicate null and skip the file."""
        table = _write_partitioned(tmp_path / "t", [7], pl.Int32())
        _rewrite_partition_values(table, lambda _: "12x")
        with pytest.raises(Exception, match="12x"):
            scan_delta(table).filter(pl.col("p") == 7).collect()

    def test_unaffected_files_still_fail_the_scan(self, tmp_path):
        """One broken add poisons the scan; it does not just lose that file."""
        table = _write_partitioned(tmp_path / "t", [1, 2], pl.Int32())
        _rewrite_partition_values(table, lambda old: "12x" if old == "2" else old)
        with pytest.raises(Exception, match="12x"):
            scan_delta(table).collect()


class TestAcceptedSpellings:
    """Spellings the kernel accepts that polars' own parsing would not."""

    @pytest.mark.parametrize("spelling", ["true", "True", "TRUE", "tRuE"])
    def test_boolean_is_case_insensitive(self, tmp_path, spelling):
        table = _write_partitioned(tmp_path / "t", [False], pl.Boolean())
        _rewrite_partition_values(table, lambda _: spelling)
        assert _read_p(table) == [True]

    @pytest.mark.parametrize(
        "spelling",
        [
            "2021-01-02 08:45:00",
            "2021-01-02 08:45:00.000000",
            "2021-01-02T08:45:00Z",
            "2021-01-02 08:45:00Z",
            "2021-01-02t08:45:00z",
        ],
    )
    def test_timestamp_spellings(self, tmp_path, spelling):
        expected = datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC)
        table = _write_partitioned(tmp_path / "t", [expected], pl.Datetime("us", "UTC"))
        _rewrite_partition_values(table, lambda _: spelling)
        assert _read_p(table) == [expected]

    def test_timestamp_offset_is_normalized_to_utc(self, tmp_path):
        table = _write_partitioned(
            tmp_path / "t",
            [datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC)],
            pl.Datetime("us", "UTC"),
        )
        _rewrite_partition_values(table, lambda _: "2021-01-02T14:15:00+05:30")
        assert _read_p(table) == [datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC)]

    @pytest.mark.parametrize(
        ("dtype", "value", "spelling"),
        [
            (pl.Date(), datetime.date(2021, 1, 2), "2021-1-2"),
            (pl.Date(), datetime.date(2021, 1, 2), "2021-01-2"),
            (pl.Int32(), 7, "+7"),
            (pl.Decimal(10, 2), decimal.Decimal("12.34"), "1.234e1"),
        ],
        ids=["unpadded_date", "unpadded_day", "signed_int", "exponent_decimal"],
    )
    def test_kernel_grammar_spellings(self, tmp_path, dtype, value, spelling):
        """Spellings `PrimitiveType::parse_scalar` accepts beyond the canonical form."""
        table = _write_partitioned(tmp_path / "t", [value], dtype)
        _rewrite_partition_values(table, lambda _: spelling)
        assert _read_p(table) == [value]

    @pytest.mark.parametrize(
        ("dtype", "value", "spelling"),
        [
            (pl.Int32(), 7, "7.0"),
            (pl.Int32(), 7, " 7"),
            (pl.Boolean(), True, "1"),
            (pl.Date(), datetime.date(2021, 1, 2), "20210102"),
            (
                pl.Datetime("us", "UTC"),
                datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC),
                "2021-01-02T08:45:00",
            ),
        ],
        ids=[
            "float_for_int",
            "padded_int",
            "one_for_bool",
            "compact_date",
            "no_offset",
        ],
    )
    def test_spellings_kernel_rejects(self, tmp_path, dtype, value, spelling):
        """`parse_scalar` rejects these, so the scan fails rather than nulling."""
        table = _write_partitioned(tmp_path / "t", [value], dtype)
        _rewrite_partition_values(table, lambda _: spelling)
        with pytest.raises(Exception, match="Failed to parse"):
            scan_delta(table).collect()

    def test_mixed_spellings_in_one_commit(self, tmp_path):
        """Both adds share a batch, so a single inferred format would null one.

        polars' format inference locks onto the first value's pattern family;
        the kernel instead parses each value on its own.
        """
        first = datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC)
        second = datetime.datetime(2021, 3, 4, 10, 30, tzinfo=UTC)
        table = _write_partitioned(
            tmp_path / "t", [first, second], pl.Datetime("us", "UTC")
        )
        _rewrite_partition_values(
            table,
            lambda old: "2021-03-04T10:30:00Z" if old.startswith("2021-03-04") else old,
        )
        assert _read_p(table) == [first, second]
