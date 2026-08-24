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
