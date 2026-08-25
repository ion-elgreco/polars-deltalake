"""A corrupted commit whose add action lacks the non-nullable `path` must
error loudly instead of null-filling — a silently dropped add (or a dropped
remove resurrecting its file) is wrong data. ScanJson null-fills only
nullable fields per the kernel plan contract."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from deltalake import write_deltalake

from polars_deltalake import scan_delta


def _strip_add_path(table_path: Path, version: int = 0) -> None:
    log = table_path / "_delta_log" / f"{version:020d}.json"
    lines = []
    for line in log.read_text().splitlines():
        action = json.loads(line)
        if "add" in action:
            del action["add"]["path"]
        lines.append(json.dumps(action))
    log.write_text("\n".join(lines) + "\n")


def test_add_without_path_errors(tmp_path: Path) -> None:
    table = tmp_path / "table"
    write_deltalake(table, pl.DataFrame({"x": [1, 2]}).to_arrow())
    _strip_add_path(table)

    with pytest.raises(Exception, match="non-nullable"):
        scan_delta(str(table)).collect()


def _rewrite_add_stats(table_path: Path, stats: str, version: int = 0) -> None:
    log = table_path / "_delta_log" / f"{version:020d}.json"
    lines = []
    for line in log.read_text().splitlines():
        action = json.loads(line)
        if "add" in action:
            action["add"]["stats"] = stats
        lines.append(json.dumps(action))
    log.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "stats",
    ["", "   ", "\t\n", "not json", '{"numRecords":', "42", "null"],
    ids=[
        "empty",
        "spaces",
        "whitespace",
        "garbage",
        "truncated",
        "scalar",
        "json-null",
    ],
)
def test_unparsable_stats_reads_every_row(tmp_path: Path, stats: str) -> None:
    """`ParseJson` decodes unparsable input to NULL — data skipping reads a
    null stats struct as "keep the file". polars' `json_decode` instead drops
    the *row* for a blank value (losing the add action and every row of the
    file it names) and fails the batch for malformed JSON. Kernel only builds
    a `ParseJson` node when the scan carries a predicate, so the unfiltered
    read stays correct either way."""
    table = tmp_path / "stats"
    write_deltalake(table, pl.DataFrame({"x": [1, 2]}).to_arrow())
    write_deltalake(table, pl.DataFrame({"x": [3, 4]}).to_arrow(), mode="append")
    _rewrite_add_stats(table, stats, version=0)

    got = scan_delta(str(table)).filter(pl.col("x") >= 0).collect()
    assert sorted(got["x"].to_list()) == [1, 2, 3, 4]
