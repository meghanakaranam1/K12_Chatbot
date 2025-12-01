"""
Session sanitization utilities to ensure JSON-safe data storage
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

def to_plain_activity(row) -> Dict[str, Any]:
    """
    Convert a pandas row (Series/DataFrame) or dict to a clean, JSON-safe dict.
    
    Args:
        row: pandas Series, DataFrame, or dict
        
    Returns:
        dict: Clean dict with only JSON-safe primitives
    """
    if hasattr(row, "to_dict"):  # pandas Series/DataFrame
        row = row.to_dict()
    
    # Ensure only JSON-safe primitives
    def norm(v):
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (np.ndarray,)):
            return [norm(x) for x in v.tolist()]
        if isinstance(v, (list, tuple, set)):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {key: norm(val) for key, val in v.items()}
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v
    
    return {k: norm(v) for k, v in (row or {}).items()}

def sanitize_session_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize session data to remove pandas objects and ensure JSON safety.
    
    Args:
        session_data: Raw session data
        
    Returns:
        dict: Sanitized session data
    """
    sanitized = {}
    
    for key, value in session_data.items():
        if key == "active_focus" and isinstance(value, dict):
            # Special handling for active_focus
            sanitized_focus = {}
            for focus_key, focus_value in value.items():
                if focus_key == "activity" and focus_value:
                    sanitized_focus[focus_key] = to_plain_activity(focus_value)
                else:
                    sanitized_focus[focus_key] = focus_value
            sanitized[key] = sanitized_focus
        elif key == "last_results" and isinstance(value, list):
            # Sanitize each result in last_results
            sanitized[key] = [to_plain_activity(item) if isinstance(item, dict) else item for item in value]
        else:
            sanitized[key] = value
    
    return sanitized






