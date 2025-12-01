# services/router/utils.py

from typing import Optional

def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    if not text:
        return ""
    # Convert to lowercase and remove extra whitespace
    return " ".join(text.lower().split())

def get_embedder():
    """
    Return the global sentence embedder used across the app (same one FAISS uses).
    Safe to call even if no embedder is configured.
    """
    try:
        from data.processor import data_processor  # local import to avoid circulars at module import time
        return getattr(data_processor, "embedder", None)
    except Exception:
        return None






