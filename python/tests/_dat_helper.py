"""Fetch + cache the Delta Acceptance Tests (DAT) tarball.

`reader_cases()` returns per-case directories under
`out/reader_tests/generated/`, each containing a `delta/` table and an
`expected/latest/table_content/*.parquet` reference. Tarball downloads once
into `python/tests/.dat-cache/` (gitignored); the marker file gates re-runs.
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

_DAT_VERSION = "v0.0.3"
_DAT_URL = (
    "https://github.com/delta-incubator/dat/releases/download/"
    f"{_DAT_VERSION}/deltalake-dat-{_DAT_VERSION}.tar.gz"
)
_CACHE_DIR = Path(__file__).parent / ".dat-cache"


def _dat_root() -> Path:
    """Return the extracted DAT bundle root, fetching + extracting on first call."""
    extracted = _CACHE_DIR / _DAT_VERSION
    marker = extracted / ".extracted"
    if marker.exists():
        return extracted

    _CACHE_DIR.mkdir(exist_ok=True)
    tarball = _CACHE_DIR / f"dat-{_DAT_VERSION}.tar.gz"
    if not tarball.exists():
        urllib.request.urlretrieve(_DAT_URL, tarball)
    # Wipe any partial tree from a crashed prior run before re-extracting.
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir()
    with tarfile.open(tarball) as tar:
        # filter='data' is the tarbomb-safe default added in Python 3.12.
        tar.extractall(extracted, filter="data")
    marker.touch()
    return extracted


def reader_cases() -> list[Path]:
    """Paths to all DAT reader test cases (one per generated table)."""
    root = _dat_root() / "out" / "reader_tests" / "generated"
    return sorted(p for p in root.iterdir() if p.is_dir())
