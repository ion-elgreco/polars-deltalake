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
from pathlib import Path
from typing import Any, Callable

import polars as pl
import pytest
from _log_helpers import rewrite_log_actions
from polars.testing import assert_frame_equal

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

    def mutate(action: dict) -> None:
        add = action.get("add")
        if add is not None and "p" in add["partitionValues"]:
            new = rewrite(add["partitionValues"]["p"])
            if new is _DROP:
                del add["partitionValues"]["p"]
            else:
                add["partitionValues"]["p"] = new

    rewrite_log_actions(table_path, mutate, version)


def _read(table_path: str) -> pl.DataFrame:
    return scan_delta(table_path).collect().sort("v")


def _p_frame(values: list[Any], dtype: pl.DataType) -> pl.DataFrame:
    """The frame `_write_partitioned(_, values, dtype)` reads back as."""
    return pl.DataFrame(
        {"p": values, "v": list(range(len(values)))},
        schema_overrides={"p": dtype},
    )


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
        assert_frame_equal(_read(table), _p_frame([value], dtype))

    def test_multi_value_partition_roundtrip(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [3, 1, 2], pl.Int32())
        assert_frame_equal(_read(table), _p_frame([3, 1, 2], pl.Int32()))


class TestNullProducingValues:
    """The contract's only null results: a missing key and the empty string."""

    def test_missing_key_is_null(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [7], pl.Int32())
        _rewrite_partition_values(table, lambda _: _DROP)
        assert_frame_equal(_read(table), _p_frame([None], pl.Int32()))

    def test_json_null_is_null(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [7], pl.Int32())
        _rewrite_partition_values(table, lambda _: None)
        assert_frame_equal(_read(table), _p_frame([None], pl.Int32()))

    def test_empty_string_is_null_for_non_string(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", [7], pl.Int32())
        _rewrite_partition_values(table, lambda _: "")
        assert_frame_equal(_read(table), _p_frame([None], pl.Int32()))

    def test_empty_string_is_itself_for_string(self, tmp_path):
        table = _write_partitioned(tmp_path / "t", ["a"], pl.String())
        _rewrite_partition_values(table, lambda _: "")
        assert_frame_equal(_read(table), _p_frame([""], pl.String()))


class TestUnparsableValueFails:
    """An unparsable value is a broken table: fail loudly, never silently."""

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
        assert_frame_equal(_read(table), _p_frame([True], pl.Boolean()))

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
        assert_frame_equal(_read(table), _p_frame([expected], pl.Datetime("us", "UTC")))

    def test_timestamp_offset_is_normalized_to_utc(self, tmp_path):
        table = _write_partitioned(
            tmp_path / "t",
            [datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC)],
            pl.Datetime("us", "UTC"),
        )
        _rewrite_partition_values(table, lambda _: "2021-01-02T14:15:00+05:30")
        assert_frame_equal(
            _read(table),
            _p_frame(
                [datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC)],
                pl.Datetime("us", "UTC"),
            ),
        )

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
        assert_frame_equal(_read(table), _p_frame([value], dtype))

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
        assert_frame_equal(
            _read(table), _p_frame([first, second], pl.Datetime("us", "UTC"))
        )


class TestPruningParsesLikeProjection:
    """Partition pruning for predicates kernel cannot translate builds its own
    frame from the raw `partitionValues` strings. It has to use the same
    kernel grammar the select list does: a polars cast nulls spellings Delta
    mandates, and a pruning frame of nulls drops every matching file."""

    @pytest.mark.parametrize(
        ("dtype", "value"),
        [
            (
                pl.Datetime("us", "UTC"),
                datetime.datetime(2021, 1, 2, 8, 45, tzinfo=UTC),
            ),
            (pl.Datetime("us"), datetime.datetime(2021, 1, 2, 8, 45)),
            (pl.Date(), datetime.date(2021, 1, 2)),
        ],
        ids=["timestamp", "timestamp_ntz", "date"],
    )
    def test_untranslatable_conjunct_keeps_matching_files(self, tmp_path, dtype, value):
        table = _write_partitioned(tmp_path / "t", [value], dtype)
        # `.dt.year()` has no kernel translation, so this routes to the
        # polars-side partition-pruning frame.
        got = scan_delta(table).filter(pl.col("p").dt.year() == 2021).collect()
        assert_frame_equal(got, _p_frame([value], dtype))

    def test_unreferenced_partition_column_is_not_materialized(self, tmp_path):
        """A boolean partition column polars cannot cast from string must not
        abort a skip that never reads it."""
        from deltalake import write_deltalake

        table = str(tmp_path / "t")
        df = pl.DataFrame(
            {"g": ["a", "b"], "flag": [True, False], "v": [0, 1]},
            schema_overrides={"flag": pl.Boolean()},
        )
        write_deltalake(table, df.to_arrow(), partition_by=["g", "flag"])
        got = scan_delta(table).filter(pl.col("g").str.to_uppercase() == "A").collect()
        assert_frame_equal(got, pl.DataFrame({"g": ["a"], "flag": [True], "v": [0]}))

    def test_stale_partition_key_is_ignored(self, tmp_path):
        """An extra key a foreign writer left in `partitionValues` names no
        logical column; a skip that never references it must still run."""
        table = _write_partitioned(tmp_path / "t", ["a", "b"], pl.String())

        def add_stale_key(action: dict) -> None:
            if "add" in action:
                action["add"]["partitionValues"]["dropped_col"] = "x"

        rewrite_log_actions(table, add_stale_key)

        got = scan_delta(table).filter(pl.col("p").str.to_uppercase() == "A").collect()
        assert_frame_equal(got, pl.DataFrame({"p": ["a"], "v": [0]}))


class TestTranslatablePartitionPredicateIsExact:
    """Accepting a predicate makes polars delete its own filter node, so every
    conjunct we accept has to be evaluated somewhere. Kernel translating a
    partition conjunct is not enough on its own: its pruning evaluator has no
    rule for some shapes and then prunes nothing at all.
    """

    def test_column_to_column_comparison_does_not_leak_rows(self, tmp_path):
        """`eval_pred_binary_columns` returns no verdict, so kernel keeps every
        file and the conjunct has to be evaluated on the polars side."""
        from deltalake import write_deltalake

        table = str(tmp_path / "t")
        df = pl.DataFrame(
            {"a": [1, 1, 2], "b": [1, 2, 2], "v": [0, 1, 2]},
            schema_overrides={"a": pl.Int32(), "b": pl.Int32()},
        )
        write_deltalake(table, df.to_arrow(), partition_by=["a", "b"])
        got = scan_delta(table).filter(pl.col("a") == pl.col("b")).collect().sort("v")
        assert_frame_equal(
            got,
            pl.DataFrame(
                {"a": [1, 2], "b": [1, 2], "v": [0, 2]},
                schema_overrides={"a": pl.Int32, "b": pl.Int32},
            ),
        )
