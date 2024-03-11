from typing import Dict, Optional

import polars as pl

from ._internal import __version__ as __version__
from ._internal import custom_scan_delta as _scan_delta


def scan_delta(
    uri: str,
    version: Optional[int] = None,
    retries: Optional[int] = None,
    storage_options: Optional[Dict[str, str]] = None,
    low_memory: bool = False,
    use_statistics: bool = True,
) -> pl.LazyFrame:
    return _scan_delta(
        uri=uri,
        version=version,
        storage_options=storage_options,
        retries=retries,
        low_memory=low_memory,
        use_statistics=use_statistics,
    )


__all__ = ["scan_delta"]
