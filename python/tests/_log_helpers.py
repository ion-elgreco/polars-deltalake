"""Read, write and rewrite `_delta_log` commit entries.

Every corrupt-log / foreign-writer test mutates a `deltalake`-written
commit the same way: parse each NDJSON action, edit it, write the file
back. The scaffold lives here once; each test supplies its mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _commit_file(table_path: str | Path, version: int) -> Path:
    return Path(table_path) / "_delta_log" / f"{version:020d}.json"


def read_log_actions(table_path: str | Path, version: int) -> list[dict[str, Any]]:
    """The actions of commit `version`, in file order."""
    lines = _commit_file(table_path, version).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def write_log_actions(
    table_path: str | Path, version: int, actions: list[dict[str, Any]]
) -> None:
    """Write commit `version` holding exactly `actions`."""
    _commit_file(table_path, version).write_text(
        "".join(json.dumps(a) + "\n" for a in actions), encoding="utf-8"
    )


def rewrite_log_actions(
    table_path: str | Path,
    mutate: Callable[[dict[str, Any]], None],
    version: int = 0,
) -> None:
    """Apply `mutate` (in place) to every action of commit `version`."""
    actions = read_log_actions(table_path, version)
    for action in actions:
        mutate(action)
    write_log_actions(table_path, version, actions)
