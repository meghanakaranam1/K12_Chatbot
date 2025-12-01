# src/router/utils.py
import re
import unicodedata

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # unify smart quotes/dashes
    s = (s.replace("’", "'").replace("‘", "'")
           .replace("“", '"').replace("”", '"')
           .replace("—", "-").replace("–", "-"))
    # drop all non word/space/'/-
    s = re.sub(r"[^\w\s'\-]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def soft_contains(needle: str, hay: str) -> bool:
    n = normalize_text(needle)
    h = normalize_text(hay)
    if not n or not h:
        return False
    
    # exact containment
    if n in h or h in n:
        return True
    
    # allow "the X one" pattern
    n_clean = re.sub(r"\b(the|one|activity|lesson|plan)\b", "", n).strip()
    return n_clean and n_clean in h

def get_embedder():
    """Get the global embedder instance"""
    try:
        from data.processor import data_processor
        return data_processor.embedder
    except Exception:
        return None
