# services/router/select.py
from __future__ import annotations
from typing import Dict, List, Optional
import re
import os

from .utils import normalize_text, get_embedder

# Optional: thresholds from env (safe defaults)
TAU_COS = float(os.getenv("SELECTION_MIN_COS", "0.62"))     # embedding similarity
TAU_JAC = float(os.getenv("SELECTION_MIN_JAC", "0.55"))     # token Jaccard


def _tokens(s: str) -> List[str]:
    s = normalize_text(s)
    return [w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 1]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B)
    return (inter / union) if union else 0.0


def _variants(title: str) -> List[str]:
    """Auto-generate soft variants of a title (no hardcoded phrases)."""
    t = (title or "").strip()
    cand = {t}
    # remove punctuation-heavy decorations
    for sep in (":", "–", "—", "-", "—"):
        if sep in t:
            cand.add(t.split(sep, 1)[0].strip())
            cand.add(t.split(sep, 1)[-1].strip())
    # collapse multiple spaces
    cand.update({re.sub(r"\s+", " ", x).strip() for x in list(cand)})
    return [c for c in cand if c]


def try_select_from_last_dynamic(
    user_input: str,
    last_results: List[Dict],
    *,
    embedder,  # expects .encode([text], normalize_embeddings=True) -> np.array
) -> Optional[Dict]:
    """
    Dynamic (non rule-based) selection:
    - score semantic similarity between user text and each last_result title (and a few auto variants)
    - combine with token Jaccard
    - if best score clears threshold, pick it; else return None
    """
    if not last_results:
        return None

    ui_text = normalize_text(user_input)
    if not ui_text:
        return None

    # Build candidate strings per item
    candidates: List[Dict] = []
    for item in last_results:
        title = (item.get("title") or item.get("Strategic Action") or "").strip()
        if not title:
            continue
        vars_ = _variants(title)
        candidates.append({"item": item, "title": title, "variants": vars_})

    # Embed once
    try:
        ui_vec = embedder.encode([user_input], normalize_embeddings=True)[0]
    except Exception:
        ui_vec = None

    best = None
    best_score = -1.0

    utok = _tokens(user_input)

    for cand in candidates:
        t = cand["title"]
        vlist = [t] + [v for v in cand["variants"] if v != t]

        # semantic score: max cosine among variants
        cos = 0.0
        if ui_vec is not None:
            try:
                vvecs = embedder.encode(vlist, normalize_embeddings=True)
                # cosine since normalized
                cos_vals = (vvecs @ ui_vec)
                cos = float(cos_vals.max())
            except Exception:
                cos = 0.0

        # lexical score: Jaccard on tokens
        jmax = 0.0
        for v in vlist:
            j = _jaccard(utok, _tokens(v))
            if j > jmax:
                jmax = j

        # combine (use the best of cos, jaccard, or their mean)
        score = max(cos, jmax, 0.5 * cos + 0.5 * jmax)

        if score > best_score:
            best_score = score
            best = cand

    # Decide
    if best and (best_score >= TAU_COS or best_score >= TAU_JAC):
        item = best["item"]
        return {
            "intent": "open_activity",
            "target": "activity",
            "title": item.get("title") or item.get("Strategic Action"),
            "payload": item,
            "confidence": float(min(0.99, max(best_score, 0.6))),
        }

    return None


# ---- Compatibility wrapper ----
def try_select_from_last(user_input: str, last_results: list, current_idx=None):
    """
    Backward/compat wrapper used by services/router.py
    Delegates to dynamic selector if we have an embedder; otherwise returns None.
    """
    emb = get_embedder()
    if not emb:
        return None
    return try_select_from_last_dynamic(user_input, last_results, embedder=emb)
