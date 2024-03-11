from typing import TYPE_CHECKING, Dict, Optional

__version__: str

if TYPE_CHECKING:
    import polars as pl

def custom_scan_delta(
    uri: str,
    version: Optional[int],
    storage_options: Optional[Dict[str, str]],
    retries: Optional[int],
    low_memory: bool = False,
    use_statistics: bool = True,
) -> pl.LazyFrame: ...
