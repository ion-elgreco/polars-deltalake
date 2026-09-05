"""Deletion-vector tables assembled from the DAT fixture.

`deltalake` cannot write a deletion vector and Spark is heavy, so a DV
table starts from the DAT `deletion_vectors` fixture: one 5-row parquet
file whose DV drops the four `letter == 'a'` rows. Its protocol, metadata,
data file and DV file are copied over, `deltalake` appends the plain data
files, and the DV commit goes first or last so the DV file lands on either
side of the plain files in scan order.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
from _dat_helper import reader_cases
from _log_helpers import read_log_actions, rewrite_log_actions, write_log_actions

# The one logical row of the DAT file after its DV.
DAT_SURVIVOR = pl.DataFrame(
    {"letter": ["b"], "int": [228], "date": [date(1978, 12, 1)]},
    schema={"letter": pl.String, "int": pl.Int64, "date": pl.Date},
)


def dat_dv_table() -> Path:
    case = next(c for c in reader_cases() if c.name == "deletion_vectors")
    return case / "delta"


def _versions(table: Path) -> list[int]:
    return sorted(int(p.stem) for p in (table / "_delta_log").glob("*.json"))


def build_dv_table(
    dst: Path,
    plain: list[pl.DataFrame],
    *,
    dv_commit: str,
    row_group_size: int,
    num_records: bool,
) -> Path:
    """Assemble a DV table at `dst`.

    Each frame in `plain` becomes one appended parquet file (schema:
    letter, int, date). `dv_commit` puts the DAT DV file in the oldest
    ("first") or newest ("last") commit. `num_records=False` strips the
    stats off the plain files' add actions, the shape a stats-less writer
    produces, which forces the reader to fetch row counts from the footers.
    """
    from deltalake import WriterProperties, write_deltalake

    assert dv_commit in ("first", "last")
    src = dat_dv_table()
    actions = [a for v in _versions(src) for a in read_log_actions(src, v)]
    protocol = next(a for a in actions if "protocol" in a)
    metadata = next(a for a in actions if "metaData" in a)
    dv_add = next(a for a in actions if a.get("add", {}).get("deletionVector"))

    dst.mkdir(parents=True)
    (dst / "_delta_log").mkdir()
    shutil.copy(src / dv_add["add"]["path"], dst)
    for dv_file in src.glob("deletion_vector_*.bin"):
        shutil.copy(dv_file, dst)

    first = [protocol, metadata] + ([dv_add] if dv_commit == "first" else [])
    write_log_actions(dst, 0, first)
    for df in plain:
        write_deltalake(
            dst,
            df.to_arrow(),
            mode="append",
            writer_properties=WriterProperties(max_row_group_size=row_group_size),
        )
        if not num_records:
            rewrite_log_actions(
                dst, lambda a: a.get("add", {}).pop("stats", None), _versions(dst)[-1]
            )
    if dv_commit == "last":
        write_log_actions(dst, _versions(dst)[-1] + 1, [dv_add])
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
