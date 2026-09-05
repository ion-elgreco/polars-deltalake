"""Hand-authored deletion-vector tables.

`deltalake` cannot write a deletion vector and Spark is heavy, so these
helpers reuse the DAT `deletion_vectors` fixture: one 5-row parquet file
whose DV drops the four `letter == 'a'` rows. A new table is assembled
from its protocol, metadata, data file and DV file, plus any number of
plain data files written by polars and committed by hand. The DV commit
can go first or last so the DV file lands on either side of the plain
files in scan order.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import polars as pl
from _dat_helper import reader_cases

# The physical rows of the DAT file; the DV keeps only `b, 228`.
DAT_PHYSICAL = pl.DataFrame(
    {
        "letter": ["a", "b", "a", "a", "a"],
        "int": [25, 228, 692, 604, 95],
    }
)
DAT_SURVIVOR = DAT_PHYSICAL.filter(pl.col("letter") == "b")


def dat_dv_table() -> Path:
    case = next(c for c in reader_cases() if c.name == "deletion_vectors")
    return case / "delta"


def _dat_actions() -> tuple[dict, dict, dict]:
    """(protocol, metaData, add-with-DV) actions from the DAT log."""
    log = dat_dv_table() / "_delta_log"
    protocol = metadata = dv_add = None
    for commit in sorted(log.glob("*.json")):
        for line in commit.read_text().splitlines():
            action = json.loads(line)
            if "protocol" in action:
                protocol = action
            elif "metaData" in action:
                metadata = action
            elif "add" in action and action["add"].get("deletionVector"):
                dv_add = action
    assert protocol and metadata and dv_add
    return protocol, metadata, dv_add


def _write_commit(table: Path, version: int, actions: list[dict]) -> None:
    commit = table / "_delta_log" / f"{version:020d}.json"
    commit.write_text("".join(json.dumps(a) + "\n" for a in actions))


def _plain_add(
    table: Path, df: pl.DataFrame, name: str, row_group_size: int, num_records: bool
) -> dict:
    file = table / name
    df.write_parquet(file, row_group_size=row_group_size, statistics=True)
    add = {
        "path": name,
        "partitionValues": {},
        "size": file.stat().st_size,
        "modificationTime": int(time.time() * 1000),
        "dataChange": True,
    }
    if num_records:
        add["stats"] = json.dumps({"numRecords": df.height})
    return {"add": add}


def build_dv_table(
    dst: Path,
    plain: list[pl.DataFrame],
    *,
    dv_commit: str = "last",
    row_group_size: int = 1000,
    num_records: bool = True,
) -> Path:
    """Assemble a DV table at `dst`.

    `plain` frames become one parquet file each (schema: letter, int, date),
    committed one per version. `dv_commit` places the DAT DV file in the
    oldest ("first") or newest ("last") commit. `num_records=False` commits
    the plain files without any stats, the shape a stats-less writer
    produces, which forces the reader to fetch row counts from the footers.
    """
    assert dv_commit in ("first", "last")
    protocol, metadata, dv_add = _dat_actions()
    src = dat_dv_table()
    dst.mkdir(parents=True)
    (dst / "_delta_log").mkdir()
    shutil.copy(src / dv_add["add"]["path"], dst / dv_add["add"]["path"])
    for dv_file in src.glob("deletion_vector_*.bin"):
        shutil.copy(dv_file, dst / dv_file.name)

    _write_commit(dst, 0, [protocol, metadata])
    version = 1
    if dv_commit == "first":
        _write_commit(dst, version, [dv_add])
        version += 1
    for i, df in enumerate(plain):
        add = _plain_add(
            dst, df, f"part-plain-{i}.parquet", row_group_size, num_records
        )
        _write_commit(dst, version, [add])
        version += 1
    if dv_commit == "last":
        _write_commit(dst, version, [dv_add])
    return dst


def plain_frame(start: int, n: int, letter: str) -> pl.DataFrame:
    """`n` rows with `int` running from `start`, sorted, so row-group
    statistics can skip on `int` predicates."""
    return pl.DataFrame(
        {
            "letter": [letter] * n,
            "int": pl.int_range(start, start + n, eager=True).cast(pl.Int64),
            "date": [None] * n,
        },
        schema={"letter": pl.String, "int": pl.Int64, "date": pl.Date},
    )
