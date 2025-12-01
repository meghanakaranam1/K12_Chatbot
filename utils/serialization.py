"""
Utilities for sanitizing data structures before API responses
"""
from typing import Any, Union, Dict, List


def strip_binary_fields(obj: Any) -> Any:
    """
    Recursively remove bytes-like fields from dict/list structures.
    Also drop very large strings if needed.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (bytes, bytearray, memoryview)):
                print(f"🔍 [DEBUG] serialization: Stripping binary field '{k}'")
                continue  # drop
            out[k] = strip_binary_fields(v)
        return out
    if isinstance(obj, list):
        return [strip_binary_fields(x) for x in obj]
    if isinstance(obj, str) and len(obj) > 10000:  # Drop very large strings
        print(f"🔍 [DEBUG] serialization: Truncating large string (length: {len(obj)})")
        return obj[:1000] + "... [truncated]"
    return obj



