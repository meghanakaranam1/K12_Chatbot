import re
from typing import List, Tuple, Optional

_punct_re = re.compile(r"[^\w\s]+")
_timey = re.compile(r"\b(\d+|\d+\s*(min|mins|minutes?)|minute|minutes?|hour|hours?)\b", re.I)

def _normalize(s: str) -> str:
    s = s.lower()
    s = _punct_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_candidate_title_from_text(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    m = re.search(r"^(?:show|open|find|display|view)\s+(?:me\s+)?(.+)$", text, flags=re.I)
    if m:
        cand = m.group(1).strip()
        cand = re.sub(r"\s+for\s+\d{1,3}\s*(?:min|mins|minutes?)\b.*$", "", cand, flags=re.I).strip()
        if _timey.search(cand) or re.fullmatch(r"\d+\s*", cand):
            return None
        return cand

    m = re.search(r"^(.+?)\s+activit(?:y|ies)\b", text, flags=re.I)
    if not m:
        return None

    cand = m.group(1).strip()
    cand = re.sub(r"\s+for\s+\d{1,3}\s*(?:min|mins|minutes?)\b.*$", "", cand, flags=re.I).strip()
    if _timey.search(cand) or re.fullmatch(r"\d+\s*", cand):
        return None
    return cand

def _is_timey_phrase(s: str) -> bool:
    t = (s or "").lower().strip()
    return bool(re.fullmatch(r"(?:[\d\s]+|and|to|under|less|than|mins?|minutes?|hours?|seconds?|time|duration)+", t))

def best_title_match(
    user_input: str,
    all_titles: List[str],
    fuzzy_threshold: int = 90
) -> Tuple[Optional[str], float]:
    cand = extract_candidate_title_from_text(user_input)
    if not cand:
        return None, 0.0
    if re.search(r"\b(\d+|\d+\s*(min|mins|minutes?)|minute|minutes?|hour|hours?)\b", cand, flags=re.I):
        return None, 0.0
    if re.fullmatch(r"\d+\s*", cand):
        return None, 0.0

    norm_cand = _normalize(cand)
    norm_map = {_normalize(t): t for t in (all_titles or [])}
    if norm_cand in norm_map:
        return norm_map[norm_cand], 100.0

    # Fuzzy match if available
    try:
        from rapidfuzz import fuzz, process  # type: ignore
        match = process.extractOne(
            norm_cand, list(norm_map.keys()), scorer=fuzz.token_set_ratio
        )
        if match:
            key, score, _ = match
            if score >= fuzzy_threshold and key in norm_map:
                return norm_map[key], float(score)
            return None, float(score)
        return None, 0.0
    except Exception:
        # No rapidfuzz → no fuzzy; fall back to no match
        return None, 0.0


