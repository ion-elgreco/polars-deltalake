from typing import Dict, Optional

__version__: str


def scan_delta(uri: str, version: Optional[int], storage_options: Optional[Dict[str,str]], 
               parallel: bool = True,
    low_memory: bool = False,
    use_statistics: bool = True,): 
    ...