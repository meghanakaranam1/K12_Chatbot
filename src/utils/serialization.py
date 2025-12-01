"""
Utility functions for serialization and data sanitization
"""
import json
from typing import Any, Dict, List, Union


def strip_binary_fields(data: Any) -> Any:
    """
    Recursively remove binary fields and large strings from data structures
    to prevent serialization issues in API responses.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Skip binary fields
            if key in ['docbytes', 'doc_bytes', 'content_bytes', 'file_bytes']:
                continue
            # Skip very large strings (over 10KB)
            if isinstance(value, str) and len(value) > 10000:
                result[key] = f"[Large string truncated: {len(value)} chars]"
            else:
                result[key] = strip_binary_fields(value)
        return result
    elif isinstance(data, list):
        return [strip_binary_fields(item) for item in data]
    elif isinstance(data, str) and len(data) > 10000:
        return f"[Large string truncated: {len(data)} chars]"
    else:
        return data


def clean_text(text: str) -> str:
    """
    Clean text to ensure valid UTF-8 encoding
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove or replace problematic characters
    text = text.replace('\x00', '')  # Remove null bytes
    text = text.replace('\ufffd', '')  # Remove replacement characters
    
    # Ensure valid UTF-8
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        return text.encode('utf-8', errors='ignore').decode('utf-8')






