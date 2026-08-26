"""Rewrite a committed `_delta_log` entry in place.

Every corrupt-log / foreign-writer test mutates a `deltalake`-written
commit the same way: parse each NDJSON action, edit it, write the file
back. The scaffold lives here once; each test supplies its mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def rewrite_log_actions(
    table_path: str | Path,
    mutate: Callable[[dict[str, Any]], None],
    version: int = 0,
) -> None:
    """Apply `mutate` (in place) to every action of commit `version`."""
    log = Path(table_path) / "_delta_log" / f"{version:020d}.json"
    lines = []
    for line in log.read_text(encoding="utf-8").splitlines():
        action = json.loads(line)
        mutate(action)
        lines.append(json.dumps(action))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
