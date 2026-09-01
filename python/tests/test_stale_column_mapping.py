"""Column-mapping annotations left behind on a table whose mode is not set.

Cloning and converting writers can leave `delta.columnMapping.physicalName` on
schema fields without enabling `delta.columnMapping.mode`. Kernel tolerates
that: in `None` mode it resolves each field by its *logical* name
(`StaleAnnotationPolicy::Ignore`), so the physical schema and the parquet files
both use logical names.

No conforming writer produces this state, so the annotations are injected into
a `deltalake`-written log the same way `test_partition_values` injects
partition values.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from _log_helpers import rewrite_log_actions
from deltalake import write_deltalake
from polars.testing import assert_frame_equal

from polars_deltalake import scan_delta


def _annotate_schema(table_path: Path, version: int = 0) -> None:
    """Add `delta.columnMapping.*` metadata to every field, leaving mode unset."""

    def annotate(action: dict) -> None:
        meta = action.get("metaData")
        if meta is None:
            return
        schema = json.loads(meta["schemaString"])
        for column_id, field in enumerate(schema["fields"], start=1):
            field.setdefault("metadata", {}).update(
                {
                    "delta.columnMapping.id": column_id,
                    "delta.columnMapping.physicalName": f"col-{column_id}",
                }
            )
        meta["schemaString"] = json.dumps(schema)

    rewrite_log_actions(table_path, annotate, version)


@pytest.fixture
def stale_annotations(tmp_path) -> str:
    """Unmapped table carrying `physicalName` metadata it no longer honours."""
    path = tmp_path / "stale"
    write_deltalake(str(path), pl.DataFrame({"id": [1, 2], "v": ["a", "b"]}).to_arrow())
    _annotate_schema(path)
    return str(path)


class TestStaleAnnotations:
    def test_mode_is_not_set(self, stale_annotations):
        """Guard: the fixture must leave column mapping disabled."""
        meta = next(
            json.loads(line)["metaData"]
            for line in Path(stale_annotations, "_delta_log", f"{0:020d}.json")
            .read_text()
            .splitlines()
            if "metaData" in json.loads(line)
        )
        assert "delta.columnMapping.mode" not in meta["configuration"]
        assert "physicalName" in meta["schemaString"]

    def test_scan_uses_logical_names(self, stale_annotations):
        out = scan_delta(stale_annotations).collect().sort("id")
        assert_frame_equal(out, pl.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
