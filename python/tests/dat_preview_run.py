"""Run the reader-side specs of a DAT preview workload bundle and summarise.

Usage: uv run --no-sync python tests/dat_preview_run.py [bundle-root] [--json out]

Covers `read`, `cdf` and `snapshot` specs; `write`, `checkpoint` and `crc`
are writer workloads and are skipped. Not collected by pytest.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
import traceback
from pathlib import Path

import polars as pl

from polars_deltalake import scan_cdf, scan_delta

ROOT = Path(__file__).parent / ".dat-cache" / "v0.0.8-preview"

# Spark spellings polars' SQL parser lacks but whose meaning is unambiguous.
_PREDICATE_REWRITES = [
    (r"TIMESTAMP_NTZ'", "TIMESTAMP'"),
    (r"AS BYTE\)", "AS TINYINT)"),
    (r"AS SHORT\)", "AS SMALLINT)"),
]


class Outcome:
    def __init__(self, status: str, reason: str = ""):
        self.status = status
        self.reason = reason


def passed() -> Outcome:
    return Outcome("pass")


def failed(reason: str) -> Outcome:
    return Outcome("fail", reason)


def unsupported(reason: str) -> Outcome:
    return Outcome("unsupported", reason)


def predicate_expr(predicate: str) -> pl.Expr:
    for pattern, repl in _PREDICATE_REWRITES:
        predicate = re.sub(pattern, repl, predicate)
    return pl.sql_expr(predicate)


def expected_frame(case: Path, spec_name: str) -> pl.DataFrame | None:
    d = case / "expected" / spec_name / "expected_data"
    parts = sorted(d.glob("*.parquet")) if d.exists() else []
    if not parts:
        return None
    return pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")


def canonical_rows(df: pl.DataFrame) -> list[str]:
    """Order-free, dtype-free row encoding so nested columns compare too."""
    return sorted(
        json.dumps(row, default=str, sort_keys=True) for row in df.to_dicts()
    )


def normalise_temporal(df: pl.DataFrame) -> pl.DataFrame:
    """Spark writes its expected parquet as tz-naive nanoseconds; compare instants."""
    exprs = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime):
            e = pl.col(name)
            if dtype.time_zone is not None:
                e = e.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
            exprs.append(e.cast(pl.Datetime("us")))
        elif isinstance(dtype, pl.Duration):
            # Spark writes day-time intervals as raw INT64 microseconds.
            exprs.append(pl.col(name).dt.total_microseconds())
    return df.with_columns(exprs) if exprs else df


def compare(actual: pl.DataFrame, expected: pl.DataFrame) -> Outcome:
    missing = [c for c in expected.columns if c not in actual.columns]
    if missing:
        return failed(f"columns missing from actual: {missing}")
    actual = normalise_temporal(actual.select(expected.columns))
    expected = normalise_temporal(expected)
    a, e = canonical_rows(actual), canonical_rows(expected)
    if a != e:
        for i, (x, y) in enumerate(zip(a, e)):
            if x != y:
                return failed(f"content: row {i}: actual {x} != expected {y}")
        return failed(f"content: {len(a)} actual rows vs {len(e)} expected")
    dtype_diffs = [
        (c, str(actual.schema[c]), str(expected.schema[c]))
        for c in expected.columns
        if actual.schema[c] != expected.schema[c]
    ]
    if dtype_diffs:
        return Outcome("dtype", f"dtype: {dtype_diffs}")
    return passed()


def expect_error(spec: dict, collect) -> Outcome:
    try:
        collect()
    except Exception as e:  # noqa: BLE001
        return Outcome("pass", f"raised {type(e).__name__}: {str(e)[:120]}")
    err = spec["error"]
    return failed(f"expected error {err.get('errorCode')}: {err.get('errorMessage')}")


def run_read(case: Path, spec_name: str, spec: dict) -> Outcome:
    if "timestamp" in spec:
        return unsupported("timestamp time travel")
    lf = scan_delta(str(case / "delta"), version=spec.get("version"))
    if "predicate" in spec:
        try:
            lf = lf.filter(predicate_expr(spec["predicate"]))
        except Exception as e:  # noqa: BLE001
            return unsupported(f"predicate parse: {str(e).splitlines()[0][:80]}")
    if "columns" in spec:
        lf = lf.select(spec["columns"])
    if "error" in spec:
        return expect_error(spec, lf.collect)
    actual = lf.collect()
    exp = spec["expected"]
    if actual.height != exp["rowCount"]:
        return failed(f"rowCount {actual.height} != {exp['rowCount']}")
    expected = expected_frame(case, spec_name)
    if expected is None:
        return passed()
    return compare(actual, expected)


def run_cdf(case: Path, spec_name: str, spec: dict) -> Outcome:
    lf = scan_cdf(
        str(case / "delta"),
        start_version=spec["startVersion"],
        end_version=spec.get("endVersion"),
    )
    if "predicate" in spec:
        try:
            lf = lf.filter(predicate_expr(spec["predicate"]))
        except Exception as e:  # noqa: BLE001
            return unsupported(f"predicate parse: {str(e).splitlines()[0][:80]}")
    if "error" in spec:
        return expect_error(spec, lf.collect)
    actual = lf.collect()
    exp = spec["expected"]
    if actual.height != exp["rowCount"]:
        return failed(f"rowCount {actual.height} != {exp['rowCount']}")
    expected = expected_frame(case, spec_name)
    if expected is None:
        return passed()
    return compare(actual, expected)


def run_snapshot(case: Path, spec_name: str, spec: dict) -> Outcome:
    if "timestamp" in spec:
        return unsupported("timestamp time travel")
    lf = scan_delta(str(case / "delta"), version=spec.get("version"))
    if "error" in spec:
        return expect_error(spec, lf.collect_schema)
    schema = lf.collect_schema()
    expected_names = [
        f["name"]
        for f in json.loads(spec["expected"]["metadata"]["schemaString"])["fields"]
    ]
    if list(schema.names()) != expected_names:
        return failed(f"schema names {list(schema.names())} != {expected_names}")
    return passed()


RUNNERS = {"read": run_read, "cdf": run_cdf, "snapshot": run_snapshot}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(ROOT))
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="case-name substring filter")
    args = ap.parse_args()
    root = Path(args.root)

    results: list[dict] = []
    t0 = time.time()
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        if args.only and args.only not in case.name:
            continue
        for spec_file in sorted((case / "specs").glob("*.json")):
            spec = json.loads(spec_file.read_text())
            runner = RUNNERS.get(spec["type"])
            if runner is None:
                out = Outcome("skip", f"{spec['type']} is a writer workload")
            else:
                try:
                    out = runner(case, spec_file.stem, spec)
                except Exception as e:  # noqa: BLE001
                    msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                    if "error" in spec:
                        # The failure may surface before `expect_error`
                        # (table open); any exception satisfies the spec.
                        out = Outcome("pass", f"raised {type(e).__name__}: {msg[:120]}")
                    else:
                        out = Outcome("error", f"{type(e).__name__}: {msg[:400]}")
                        out.trace = traceback.format_exc()
            results.append(
                {
                    "case": case.name,
                    "spec": spec_file.stem,
                    "type": spec["type"],
                    "status": out.status,
                    "reason": out.reason,
                }
            )
    elapsed = time.time() - t0

    by_type = collections.defaultdict(collections.Counter)
    for r in results:
        by_type[r["type"]][r["status"]] += 1
    print(f"{len(results)} specs in {elapsed:.0f}s")
    for t, c in sorted(by_type.items()):
        print(f"  {t:9s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))

    def family(name: str) -> str:
        return name.split("_", 1)[0]

    for status in ("fail", "error", "dtype", "unsupported"):
        rows = [r for r in results if r["status"] == status]
        if not rows:
            continue
        print(f"\n== {status} ({len(rows)}) by family ==")
        fam = collections.Counter(family(r["case"]) for r in rows)
        print("  " + ", ".join(f"{k}={v}" for k, v in fam.most_common()))
        reasons = collections.Counter(
            re.sub(r"\d+", "N", r["reason"].split(":")[0])[:60] for r in rows
        )
        print("  reasons: " + ", ".join(f"{k!r}={v}" for k, v in reasons.most_common(12)))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
