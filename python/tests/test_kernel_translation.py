"""Tests for `to_kernel`: classification (does each shape lower to a kernel
`Predicate`?) and end-to-end correctness (do filter results match?).

Classification asserts via `TableState._classify_predicate`, which counts
kernel-translatable conjuncts in the ``"kernel"`` bucket.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from deltalake import write_deltalake

from polars_deltalake import TableScan, TableState, scan_delta


@pytest.fixture
def rich_table(tmp_path: Path) -> str:
    """Three rows (`id ∈ {1, 2, 3}`) covering every column type the
    translator can produce a kernel literal for. End-to-end tests assert
    surviving ids per shape."""
    df = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.Int64),
            "small": pl.Series([1, 2, 3], dtype=pl.Int32),
            "u": pl.Series([1, 2, 3], dtype=pl.UInt32),
            "f": pl.Series([0.5, 1.5, 2.5], dtype=pl.Float64),
            "s": pl.Series(["a", "b", "c"], dtype=pl.String),
            "b": pl.Series([True, False, True], dtype=pl.Boolean),
            "d": pl.Series(
                [date(2024, 1, 1), date(2024, 6, 1), date(2024, 12, 31)],
                dtype=pl.Date,
            ),
            "t": pl.Series(
                [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 6, 1, tzinfo=timezone.utc),
                    datetime(2024, 12, 31, tzinfo=timezone.utc),
                ],
                dtype=pl.Datetime("us", time_zone="UTC"),
            ),
            "t_ntz": pl.Series(
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 6, 1),
                    datetime(2024, 12, 31),
                ],
                dtype=pl.Datetime("us"),
            ),
            "dec": pl.Series(
                [Decimal("1.23"), Decimal("4.56"), Decimal("7.89")],
                dtype=pl.Decimal(precision=10, scale=2),
            ),
        }
    )
    path = tmp_path / "rich"
    write_deltalake(str(path), df.to_arrow())
    return str(path)


def _kernel_count(table: str, predicate: pl.Expr) -> int:
    """Conjuncts (after top-level AND split) that reach the kernel bucket."""
    return TableState(table)._classify_predicate(predicate)["kernel"]  # type: ignore[attr-defined]


class TestComparisons:
    """All six comparison operators against typed literals."""

    @pytest.mark.parametrize(
        ("col", "value"),
        [
            ("id", 1),
            ("small", 1),
            ("u", 1),
            ("f", 1.5),
            ("s", "a"),
            ("b", True),
            ("d", date(2024, 1, 1)),
            ("t", datetime(2024, 1, 1, tzinfo=timezone.utc)),
            ("t_ntz", datetime(2024, 1, 1)),
            ("dec", Decimal("1.23")),
        ],
    )
    @pytest.mark.parametrize(
        "op",
        ["eq", "ne", "lt", "le", "gt", "ge"],
    )
    def test_binary_comparison(self, rich_table, col, value, op):
        expr = getattr(pl.col(col), op)(value)
        assert _kernel_count(rich_table, expr) == 1, f"{col} {op} {value!r}"

    def test_null_aware_equality(self, rich_table):
        """`eq_missing` / `ne_missing` lower to kernel's null-aware
        `Distinct`."""
        assert _kernel_count(rich_table, pl.col("id").eq_missing(1)) == 1
        assert _kernel_count(rich_table, pl.col("id").ne_missing(1)) == 1


class TestNullChecks:
    def test_is_null(self, rich_table):
        assert _kernel_count(rich_table, pl.col("id").is_null()) == 1

    def test_is_not_null(self, rich_table):
        assert _kernel_count(rich_table, pl.col("id").is_not_null()) == 1


class TestIsIn:
    """`col.is_in([…])` → flattened OR-chain of equalities."""

    def test_int_list(self, rich_table):
        assert _kernel_count(rich_table, pl.col("id").is_in([1, 2, 3])) == 1

    def test_string_list(self, rich_table):
        assert _kernel_count(rich_table, pl.col("s").is_in(["a", "b"])) == 1

    def test_float_list(self, rich_table):
        assert _kernel_count(rich_table, pl.col("f").is_in([0.5, 1.5])) == 1

    def test_single_value(self, rich_table):
        # Optimizer doesn't fold `is_in([x])` → `==`, so we must handle it.
        assert _kernel_count(rich_table, pl.col("id").is_in([1])) == 1

    def test_empty_list(self, rich_table):
        # `is_in([])` → `lit(false)` so kernel prunes the whole table.
        assert _kernel_count(rich_table, pl.col("id").is_in([])) == 1

    def test_below_cap_translates(self, rich_table):
        assert _kernel_count(rich_table, pl.col("id").is_in(list(range(512)))) == 1

    def test_above_cap_falls_back(self, rich_table):
        # Above the expansion cap polars-io handles the predicate row-wise.
        assert _kernel_count(rich_table, pl.col("id").is_in(list(range(513)))) == 0

    def test_cast_around_set_declines(self, rich_table):
        """A cast on the set literal changes its elements; unwrapping it
        would push the pre-cast values as the OR-chain."""
        expr = pl.col("f").is_in(pl.lit(pl.Series([2.5, 2.9])).cast(pl.Int64))
        assert _kernel_count(rich_table, expr) == 0


class TestIsBetween:
    """Every `closed=` variant decomposes into a strict-or-inclusive pair."""

    @pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
    def test_closed_variants(self, rich_table, closed):
        expr = pl.col("id").is_between(1, 5, closed=closed)
        assert _kernel_count(rich_table, expr) == 1


class TestNot:
    """The optimizer folds `~(cmp)` and `~is_null()` before we see them;
    only `~col_b` arrives as a raw `Function::Not`."""

    def test_not_over_is_null(self, rich_table):
        assert _kernel_count(rich_table, ~pl.col("id").is_null()) == 1

    def test_not_over_eq_folds_to_ne(self, rich_table):
        assert _kernel_count(rich_table, ~(pl.col("id") == 1)) == 1

    def test_not_boolean_column(self, rich_table):
        assert _kernel_count(rich_table, ~pl.col("b")) == 1


class TestJunctions:
    """Top-level AND splits per-conjunct; OR stays a single Junction."""

    def test_and(self, rich_table):
        expr = (pl.col("id") == 1) & (pl.col("f") < 2.0)
        assert _kernel_count(rich_table, expr) == 2

    def test_or(self, rich_table):
        expr = (pl.col("id") == 1) | (pl.col("f") < 2.0)
        assert _kernel_count(rich_table, expr) == 1

    def test_or_chain(self, rich_table):
        expr = (pl.col("id") == 1) | (pl.col("id") == 2) | (pl.col("id") == 3)
        assert _kernel_count(rich_table, expr) == 1

    def test_and_chain_three_way(self, rich_table):
        expr = (pl.col("id") >= 1) & (pl.col("id") <= 10) & (pl.col("s") == "a")
        assert _kernel_count(rich_table, expr) == 3


class TestBooleanColumn:
    def test_bare_boolean_column(self, rich_table):
        assert _kernel_count(rich_table, pl.col("b")) == 1

    def test_boolean_column_and_comparison(self, rich_table):
        expr = pl.col("b") & (pl.col("id") == 1)
        assert _kernel_count(rich_table, expr) == 2


class TestBooleanLiteral:
    """Optimizer drops `expr & True` and `expr | False`, but keeps
    `expr & False` and `expr | True` — both legs must lower."""

    def test_lit_true_alone(self, rich_table):
        assert _kernel_count(rich_table, pl.lit(True)) == 1

    def test_lit_false_alone(self, rich_table):
        assert _kernel_count(rich_table, pl.lit(False)) == 1

    def test_and_false_kept_by_optimizer(self, rich_table):
        expr = (pl.col("id") == 1) & pl.lit(False)
        assert _kernel_count(rich_table, expr) == 2

    def test_or_true_kept_by_optimizer(self, rich_table):
        expr = (pl.col("id") == 1) | pl.lit(True)
        assert _kernel_count(rich_table, expr) == 1


class TestPredicateEqualsBool:
    """`<pred> == lit(bool)` / `!=` — folded by the translator."""

    def test_predicate_eq_true(self, rich_table):
        expr = (pl.col("id") > 0) == True  # noqa: E712
        assert _kernel_count(rich_table, expr) == 1

    def test_predicate_eq_false(self, rich_table):
        expr = (pl.col("id") > 0) == False  # noqa: E712
        assert _kernel_count(rich_table, expr) == 1

    def test_predicate_ne_true(self, rich_table):
        expr = (pl.col("id") > 0) != True  # noqa: E712
        assert _kernel_count(rich_table, expr) == 1

    def test_predicate_ne_false(self, rich_table):
        expr = (pl.col("id") > 0) != False  # noqa: E712
        assert _kernel_count(rich_table, expr) == 1


class TestCastUnwrap:
    def test_widening_cast_around_column(self, rich_table):
        """A widening integer cast cannot change a value, so it unwraps."""
        assert _kernel_count(rich_table, pl.col("small").cast(pl.Int64) == 1) == 1

    def test_redundant_cast_around_column(self, rich_table):
        assert _kernel_count(rich_table, pl.col("id").cast(pl.Int64) == 1) == 1

    @pytest.mark.parametrize(
        "expr",
        [
            pl.col("id").cast(pl.Int32) == 1,
            pl.col("f").cast(pl.Int32) == 1,
            pl.col("id").cast(pl.String) == "1",
        ],
        ids=["narrowing", "float-truncating", "stringify"],
    )
    def test_value_changing_cast_declines(self, rich_table, expr):
        """Kernel skips a file whose stats falsify the pushed predicate, so a
        cast that changes values must not be dropped on the way down."""
        assert _kernel_count(rich_table, expr) == 0

    def test_alias_around_predicate(self, rich_table):
        expr = (pl.col("id") == 1).alias("masked")
        assert _kernel_count(rich_table, expr) == 1

    def test_literal_cast_declines(self, rich_table):
        """A cast over a literal changes the compared value; dropping it
        would push the pre-cast literal into the skipping predicate."""
        assert _kernel_count(rich_table, pl.col("f") == pl.lit(2.9).cast(pl.Int64)) == 0


class TestCastPruningSoundness:
    def test_truncating_cast_keeps_matching_rows(self, tmp_path: Path):
        """`f.cast(Int32) == 1` matches 1.4; pushing the cast-stripped
        `f == 1` down would skip the file whose stats are [1.4, 1.6]."""
        table = str(tmp_path / "casts")
        write_deltalake(
            table,
            pl.DataFrame({"id": [1, 2], "f": [1.4, 1.6]}).to_arrow(),
        )
        write_deltalake(
            table,
            pl.DataFrame({"id": [3, 4], "f": [10.0, 11.0]}).to_arrow(),
            mode="append",
        )
        got = (
            scan_delta(table)
            .filter(pl.col("f").cast(pl.Int32) == 1)
            .collect()
            .sort("id")["id"]
            .to_list()
        )
        assert got == [1, 2]

    def test_literal_cast_via_raw_configure(self, tmp_path: Path):
        """polars folds literal casts before the IO plugin sees them, but the
        exported ``TableScan.configure`` receives raw exprs. ``f == 2.9`` must
        not reach kernel: file A's stats [2.0, 2.0] falsify it, pruning the
        file that holds the rows the evaluated predicate (f == 2.0) matches."""
        table = str(tmp_path / "casts2")
        write_deltalake(table, pl.DataFrame({"id": [1, 2], "f": [2.0, 2.0]}).to_arrow())
        write_deltalake(
            table,
            pl.DataFrame({"id": [3], "f": [10.0]}).to_arrow(),
            mode="append",
        )
        scan = TableScan(TableState(table))
        scan.configure(None, None, pl.col("f") == pl.lit(2.9).cast(pl.Int64))
        frames = []
        while (df := scan.next()) is not None:
            frames.append(df)
        got = sorted(pl.concat(frames)["id"].to_list()) if frames else []
        assert got == [1, 2]

    def test_set_cast_via_raw_configure(self, tmp_path: Path):
        """The evaluated set is {2} (2.5 and 2.9 truncate); pushing the
        pre-cast elements as ``f == 2.5 OR f == 2.9`` prunes the file whose
        stats [1.0, 2.0] falsify both, losing the matching row."""
        table = str(tmp_path / "casts3")
        write_deltalake(table, pl.DataFrame({"id": [1, 2], "f": [1.0, 2.0]}).to_arrow())
        write_deltalake(
            table,
            pl.DataFrame({"id": [3], "f": [50.0]}).to_arrow(),
            mode="append",
        )
        scan = TableScan(TableState(table))
        expr = pl.col("f").is_in(pl.lit(pl.Series([2.5, 2.9])).cast(pl.Int64))
        scan.configure(None, None, expr)
        frames = []
        while (df := scan.next()) is not None:
            frames.append(df)
        got = sorted(pl.concat(frames)["id"].to_list()) if frames else []
        assert got == [2]


class TestFloat32Pushdown:
    """polars type-coercion wraps a Float32 column in ``cast(Float64)`` and
    compares against Float64 literals. Pushdown must unwrap the widening cast
    and narrow the literal exactly to the column type — kernel compares stats
    strictly same-type, so an unnarrowed Double literal never skips."""

    @pytest.fixture
    def f32_table(self, tmp_path: Path) -> str:
        path = str(tmp_path / "f32")
        write_deltalake(
            path,
            pl.DataFrame(
                {
                    "id": pl.Series([1, 2], dtype=pl.Int64),
                    "g": pl.Series([0.5, 1.5], dtype=pl.Float32),
                }
            ).to_arrow(),
        )
        write_deltalake(
            path,
            pl.DataFrame(
                {
                    "id": pl.Series([3, 4], dtype=pl.Int64),
                    "g": pl.Series([100.0, 200.0], dtype=pl.Float32),
                }
            ).to_arrow(),
            mode="append",
        )
        return path

    def test_widening_float_cast_translates(self, f32_table):
        expr = pl.col("g").cast(pl.Float64) > pl.lit(60.0, dtype=pl.Float64)
        assert _kernel_count(f32_table, expr) == 1

    def test_widening_cast_around_is_in_translates(self, f32_table):
        expr = pl.col("g").cast(pl.Float64).is_in([0.5, 1.5])
        assert _kernel_count(f32_table, expr) == 1

    def test_float32_predicate_skips_a_whole_file(self, f32_table):
        """Delete the low file: the query only succeeds if kernel skipped it."""
        adds = [
            json.loads(line)["add"]
            for log in sorted(Path(f32_table, "_delta_log").glob("*.json"))
            for line in log.read_text().splitlines()
            if "add" in json.loads(line)
        ]
        low = next(
            a["path"] for a in adds if json.loads(a["stats"])["maxValues"]["g"] < 50
        )
        Path(f32_table, low).unlink()

        # A typed Float64 literal is kept by polars (a dyn one would shrink
        # to Float32), so the plugin delivers cast(g, Float64) > Double.
        out = scan_delta(f32_table).filter(pl.col("g") > pl.lit(60.0, dtype=pl.Float64))
        assert out.collect().sort("id")["id"].to_list() == [3, 4]

    def test_inexact_literal_declines(self, f32_table):
        """2.9 has no exact Float32 form; narrowing it would push a
        satisfiable predicate for a comparison that is false on every row."""
        expr = pl.col("g").cast(pl.Float64) == pl.lit(2.9, dtype=pl.Float64)
        assert _kernel_count(f32_table, expr) == 0
        got = scan_delta(f32_table).filter(pl.col("g") == 2.9).collect()
        assert got.height == 0


class TestUntranslatable:
    """Negative coverage — shapes that must stay out of the kernel bucket."""

    def test_string_starts_with(self, rich_table):
        assert _kernel_count(rich_table, pl.col("s").str.starts_with("a")) == 0

    def test_string_contains(self, rich_table):
        assert _kernel_count(rich_table, pl.col("s").str.contains("a")) == 0

    def test_arithmetic_in_comparison(self, rich_table):
        # Kernel doesn't use arithmetic expressions for stats pruning.
        assert _kernel_count(rich_table, pl.col("id") + 1 < 5) == 0

    def test_temporal_function(self, rich_table):
        assert _kernel_count(rich_table, pl.col("d").dt.year() == 2024) == 0

    def test_abs(self, rich_table):
        assert _kernel_count(rich_table, pl.col("id").abs() >= 3) == 0


# End-to-end correctness: catches lowerings that classify as kernel-
# translatable but rewrite semantics incorrectly (is_in OR-chain,
# <pred> == lit(bool) fold, is_between bound directions, …).

_E2E_CASES = [
    # Comparisons across dtypes
    pytest.param(pl.col("id") == 1, [1], id="id-eq-1"),
    pytest.param(pl.col("id") != 1, [2, 3], id="id-ne-1"),
    pytest.param(pl.col("id") < 2, [1], id="id-lt-2"),
    pytest.param(pl.col("id") <= 2, [1, 2], id="id-le-2"),
    pytest.param(pl.col("id") > 2, [3], id="id-gt-2"),
    pytest.param(pl.col("id") >= 2, [2, 3], id="id-ge-2"),
    pytest.param(pl.col("small") == 2, [2], id="int32-eq"),
    pytest.param(pl.col("u") >= 2, [2, 3], id="uint32-ge"),
    pytest.param(pl.col("f") >= 1.5, [2, 3], id="float-ge"),
    pytest.param(pl.col("s") == "a", [1], id="str-eq"),
    pytest.param(pl.col("d") > date(2024, 6, 1), [3], id="date-gt"),
    pytest.param(
        pl.col("t") > datetime(2024, 6, 1, tzinfo=timezone.utc),
        [3],
        id="tz-datetime-gt",
    ),
    pytest.param(
        pl.col("t_ntz") > datetime(2024, 6, 1),
        [3],
        id="naive-datetime-gt",
    ),
    pytest.param(pl.col("dec") > Decimal("3.00"), [2, 3], id="decimal-gt"),
    # Null-aware equality (kernel `Distinct`)
    pytest.param(pl.col("id").eq_missing(1), [1], id="eq-missing"),
    pytest.param(pl.col("id").ne_missing(1), [2, 3], id="ne-missing"),
    # IsNull / IsNotNull
    pytest.param(pl.col("id").is_null(), [], id="is-null"),
    pytest.param(pl.col("id").is_not_null(), [1, 2, 3], id="is-not-null"),
    # IsIn — OR-chain flattening
    pytest.param(pl.col("id").is_in([1, 3]), [1, 3], id="is-in-multi"),
    pytest.param(pl.col("id").is_in([1]), [1], id="is-in-single"),
    pytest.param(pl.col("id").is_in([]), [], id="is-in-empty"),
    pytest.param(pl.col("s").is_in(["a", "c"]), [1, 3], id="is-in-strings"),
    # Float-spelled elements against an integer column: the integral ones
    # narrow, the fractional ones can match nothing and drop.
    pytest.param(pl.col("small").is_in([2.0]), [2], id="is-in-float-literals"),
    pytest.param(
        pl.col("small").is_in([2.0, 3.0]), [2, 3], id="is-in-float-literalss-multi"
    ),
    pytest.param(
        pl.col("id").is_in([1.0, 2.5]), [1], id="is-in-float-literals-fractional"
    ),
    pytest.param(
        pl.col("id").is_in([2.5]), [], id="is-in-float-literals-all-fractional"
    ),
    pytest.param(
        pl.col("d").is_in([date(2024, 1, 1), date(2024, 12, 31)]),
        [1, 3],
        id="is-in-dates",
    ),
    # IsBetween — closed=Both/Left/Right/None
    pytest.param(pl.col("id").is_between(1, 2), [1, 2], id="between-both"),
    pytest.param(
        pl.col("id").is_between(1, 3, closed="left"),
        [1, 2],
        id="between-left",
    ),
    pytest.param(
        pl.col("id").is_between(1, 3, closed="right"),
        [2, 3],
        id="between-right",
    ),
    pytest.param(
        pl.col("id").is_between(1, 3, closed="none"),
        [2],
        id="between-none",
    ),
    # NOT
    pytest.param(~pl.col("b"), [2], id="not-bool-col"),
    pytest.param(~pl.col("id").is_null(), [1, 2, 3], id="not-is-null"),
    pytest.param(~(pl.col("id") == 1), [2, 3], id="not-eq-folded"),
    # Boolean column as predicate
    pytest.param(pl.col("b"), [1, 3], id="bool-col"),
    pytest.param(pl.col("b") & (pl.col("id") <= 2), [1], id="bool-col-and-cmp"),
    # Junctions
    pytest.param(
        (pl.col("id") == 1) & (pl.col("s") == "a"),
        [1],
        id="and-two",
    ),
    pytest.param(
        (pl.col("id") == 1) | (pl.col("id") == 3),
        [1, 3],
        id="or-two",
    ),
    pytest.param(
        (pl.col("id") >= 1) & (pl.col("id") <= 3) & (pl.col("s") != "b"),
        [1, 3],
        id="and-three",
    ),
    # Boolean literals — optimizer keeps `& False` / `| True`; both legs lower.
    pytest.param(pl.lit(True), [1, 2, 3], id="lit-true"),
    pytest.param(pl.lit(False), [], id="lit-false"),
    pytest.param(
        (pl.col("id") == 1) & pl.lit(False),
        [],
        id="and-lit-false",
    ),
    pytest.param(
        (pl.col("id") == 1) | pl.lit(True),
        [1, 2, 3],
        id="or-lit-true",
    ),
    # `<pred> ==/!= lit(bool)` fold
    pytest.param((pl.col("id") > 0) == True, [1, 2, 3], id="pred-eq-true"),  # noqa: E712
    pytest.param((pl.col("id") > 0) == False, [], id="pred-eq-false"),  # noqa: E712
    pytest.param((pl.col("id") > 0) != True, [], id="pred-ne-true"),  # noqa: E712
    pytest.param(
        (pl.col("id") > 0) != False,  # noqa: E712
        [1, 2, 3],
        id="pred-ne-false",
    ),
    # Cast unwrap
    pytest.param(
        pl.col("small").cast(pl.Int64) == 1,
        [1],
        id="cast-eq",
    ),
    pytest.param(
        pl.col("id").cast(pl.Int32) == 1,
        [1],
        id="cast-eq-narrowing",
    ),
]


class TestEndToEndFiltering:
    @pytest.mark.parametrize(("predicate", "expected_ids"), _E2E_CASES)
    def test_filter_returns_expected_ids(
        self,
        rich_table,
        predicate: pl.Expr,
        expected_ids: list[int],
    ):
        out = scan_delta(rich_table).filter(predicate).collect().sort("id")
        assert out["id"].to_list() == expected_ids


@pytest.fixture
def nested_table(tmp_path: Path) -> str:
    """Two files with disjoint `person.age` ranges, plus non-primitive columns.

    `deltalake` writes nested `minValues` / `maxValues`, so kernel can skip a
    whole file on `person.age` alone.
    """

    def rows(ids: list[int], ages: list[int]):
        return pl.DataFrame(
            {
                "id": ids,
                "person": [{"name": f"n{a}", "age": a} for a in ages],
                "tags": [["t"] for _ in ids],
            }
        ).to_arrow()

    path = tmp_path / "nested"
    write_deltalake(str(path), rows([1, 2], [10, 20]))
    write_deltalake(str(path), rows([3, 4], [90, 95]), mode="append")
    return str(path)


class TestNestedColumns:
    """`struct.field(...)` chains lower to multi-segment kernel column names."""

    def test_nested_comparison(self, nested_table):
        assert (
            _kernel_count(nested_table, pl.col("person").struct.field("age") > 50) == 1
        )

    def test_nested_is_null(self, nested_table):
        assert (
            _kernel_count(nested_table, pl.col("person").struct.field("name").is_null())
            == 1
        )

    def test_nested_is_in(self, nested_table):
        assert (
            _kernel_count(
                nested_table, pl.col("person").struct.field("age").is_in([10])
            )
            == 1
        )

    def test_nested_filter_returns_expected_ids(self, nested_table):
        out = scan_delta(nested_table).filter(pl.col("person").struct.field("age") > 50)
        assert out.collect().sort("id")["id"].to_list() == [3, 4]

    def test_nested_stats_skip_a_whole_file(self, nested_table):
        """Delete the low-age file: the query only succeeds if kernel skipped it."""
        adds = [
            json.loads(line)["add"]
            for log in sorted(Path(nested_table, "_delta_log").glob("*.json"))
            for line in log.read_text().splitlines()
            if "add" in json.loads(line)
        ]
        low = next(
            a["path"]
            for a in adds
            if json.loads(a["stats"])["maxValues"]["person"]["age"] < 50
        )
        Path(nested_table, low).unlink()

        out = scan_delta(nested_table).filter(pl.col("person").struct.field("age") > 50)
        assert out.collect().sort("id")["id"].to_list() == [3, 4]


class TestNonPrimitiveReferences:
    """Kernel fails the whole scan on a predicate column it cannot resolve, and
    it resolves only primitive leaves — `person` and `tags` are containers, so
    they would abort the query. The translator must decline them and let polars
    filter."""

    @pytest.mark.parametrize(
        ("predicate", "expected_ids"),
        [
            pytest.param(pl.col("person").is_null(), [], id="struct-is-null"),
            pytest.param(
                pl.col("person").is_not_null(), [1, 2, 3, 4], id="struct-is-not-null"
            ),
            pytest.param(pl.col("tags").is_null(), [], id="list-is-null"),
            pytest.param(
                pl.col("tags").is_not_null(), [1, 2, 3, 4], id="list-is-not-null"
            ),
        ],
    )
    def test_declines_and_still_scans(
        self, nested_table, predicate: pl.Expr, expected_ids: list[int]
    ):
        assert _kernel_count(nested_table, predicate) == 0
        out = scan_delta(nested_table).filter(predicate).collect().sort("id")
        assert out["id"].to_list() == expected_ids

    def test_struct_path_declines(self, nested_table):
        """`person` alone is a struct — a path stopping there has no stats."""
        assert _kernel_count(nested_table, pl.col("person") == pl.col("person")) == 0

    def test_unknown_column_declines(self, nested_table):
        assert _kernel_count(nested_table, pl.col("nope") == 1) == 0
