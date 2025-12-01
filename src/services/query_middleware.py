from __future__ import annotations
import re
from typing import Dict, Any, Optional, Tuple


def _looks_similar_intent(text: str) -> Optional[str]:
    """
    Return the 'seed' text if the user asked for 'similar to X' / 'like X' etc.
    """
    if not text:
        return None
    t = text.strip()
    m = re.search(
        r"\b(?:similar\s+(?:to|like)|(?:more|other)\s+like|something\s+like)\s+(.+)$",
        t, flags=re.I
    )
    if not m:
        return None
    return m.group(1).strip()


def _is_grade_only(raw_q: Dict[str, Any], user_text: str) -> bool:
    """
    True if the query carries only a grade range and nothing else meaningful.
    """
    has_grade = bool(raw_q.get("grade_range"))
    has_topic = bool((raw_q.get("topic") or "").strip())
    has_syns  = bool(raw_q.get("synonyms"))
    has_time  = (raw_q.get("time_minutes") is not None) or (raw_q.get("time_range") is not None)
    has_other = any([
        raw_q.get("constraints"), raw_q.get("must_have"), raw_q.get("nice_to_have")
    ])
    return has_grade and not any([has_topic, has_syns, has_time, has_other])


def _strip_time_grade_words(text: str) -> str:
    t = text or ""
    t = re.sub(r"\b\d{1,3}\s*(?:min|mins|minutes?|hours?)\b", " ", t, flags=re.I)
    t = re.sub(r"\b(under|less than|<)\s*\d{1,3}\b", " ", t, flags=re.I)
    t = re.sub(r"\bgrade(?:s)?\s*\d{1,2}\b", " ", t, flags=re.I)
    t = re.sub(r"\b(k|kindergarten|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|grader|graders)\b", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_low_specificity(content_only: str, specificity) -> bool:
    """
    Uses your precomputed TF-IDF specificity object if available.
    Low specificity → neutralize topic/synonyms.
    """
    if not content_only:
        return True
    try:
        if specificity is None:
            # conservative: not enough info → treat as low specificity only if very short
            return len(content_only.split()) <= 2
        return not bool(specificity.is_specific(content_only))
    except Exception:
        return True


def encode_similar_constraint(raw_q: Dict[str, Any], user_text: str, session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect 'similar to X' and encode as constraints while clearing topic/synonyms.
    Also stores a seed hint for downstream exclusion logic.
    """
    seed = _looks_similar_intent(user_text)
    if not seed:
        return raw_q

    # Detect A# reference first
    seed_id = None
    m_id = re.search(r"\bA(\d{1,2})\b", seed, flags=re.I)
    if m_id:
        seed_id = f"A{int(m_id.group(1))}"

    constraints = list(dict.fromkeys((raw_q.get("constraints") or [])))

    if seed_id and (session.get("shortlist_map_rev") or {}):
        # map A# to row_index
        constraints.append(f"similar_to_id:{seed_id}")  # will resolve later
        session["_similar_seed"] = {"id": seed_id}
    else:
        # title text fallback
        constraints.append(f"similar_to_title:{seed}")
        session["_similar_seed"] = {"title": seed}

    raw_q["constraints"] = constraints
    raw_q["topic"] = ""
    raw_q["synonyms"] = []
    return raw_q


def neutralize_if_vague(raw_q: Dict[str, Any], user_text: str, specificity) -> Dict[str, Any]:
    """
    Grade-only / time-only / low-specificity → clear topic/synonyms.
    """
    if _is_grade_only(raw_q, user_text):
        raw_q["topic"] = ""
        raw_q["synonyms"] = []
        return raw_q

    content_only = _strip_time_grade_words(user_text)
    if _is_low_specificity(content_only, specificity) and not (raw_q.get("topic") or "").strip():
        raw_q["topic"] = ""
        raw_q["synonyms"] = []
    return raw_q


def merge_time_filters(raw_q: Dict[str, Any], desired_single: Optional[int], desired_range: Optional[Tuple[int,int]], user_text: str) -> Dict[str, Any]:
    """
    Deterministic time merge (your existing logic, wrapped).
    Honors 'under N' overriding single-minute.
    """
    if desired_range:
        raw_q["time_range"] = [int(desired_range[0]), int(desired_range[1])]
        raw_q["time_minutes"] = None
    elif desired_single is not None:
        raw_q["time_minutes"] = int(desired_single)

    # explicit 'under N'
    m = re.search(r"\b(under|less than|<)\s*(\d{1,3})\s*(?:min|mins|minutes?)?\b", (user_text or "").lower())
    if m:
        n = int(m.group(2))
        raw_q["time_range"] = [1, max(1, n - 1)]
        raw_q["time_minutes"] = None
    return raw_q






