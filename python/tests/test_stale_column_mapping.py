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
from deltalake import write_deltalake

from polars_deltalake import scan_delta


def _annotate_schema(table_path: Path, version: int = 0) -> None:
    """Add `delta.columnMapping.*` metadata to every field, leaving mode unset."""
    log = table_path / "_delta_log" / f"{version:020d}.json"
    lines = []
    for line in log.read_text().splitlines():
        action = json.loads(line)
        meta = action.get("metaData")
        if meta is not None:
            schema = json.loads(meta["schemaString"])
            for column_id, field in enumerate(schema["fields"], start=1):
                field.setdefault("metadata", {}).update(
                    {
                        "delta.columnMapping.id": column_id,
                        "delta.columnMapping.physicalName": f"col-{column_id}",
                    }
                )
            meta["schemaString"] = json.dumps(schema)
        lines.append(json.dumps(action))
    log.write_text("\n".join(lines) + "\n")


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
        assert out.columns == ["id", "v"]
        assert out["id"].to_list() == [1, 2]
        assert out["v"].to_list() == ["a", "b"]
