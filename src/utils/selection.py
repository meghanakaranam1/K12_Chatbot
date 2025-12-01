"""
Selection utilities for parsing user input into activity indices.
"""

import re

_WORD_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}

# Matches:
#   "the 2nd one", "2nd", "8th", "10th option"
#   "the second one", "eighth", "fourth option"
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(?:(\d+)(?:st|nd|rd|th)?|"
    r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth))"
    r"(?:\s+(?:one|option|item|activity))?\b",
    re.IGNORECASE
)

def parse_selection_index(user_text: str, max_n: int) -> int | None:
    """
    Return a 1-based index if the user referenced an ordinal/number within [1, max_n].
    Supports numeric ordinals ('4th'), plain numbers ('4'), and word ordinals ('fourth').
    
    Args:
        user_text: The user input text to parse
        max_n: Maximum valid index (length of the list)
        
    Returns:
        int: 1-based index if valid selection found, None otherwise
        
    Examples:
        parse_selection_index("show the 4th one", 8) -> 4
        parse_selection_index("2nd one please", 8) -> 2
        parse_selection_index("open 8", 8) -> 8
        parse_selection_index("the eighth activity", 8) -> 8
        parse_selection_index("open 10th", 8) -> None  # out of range
    """
    if not user_text:
        return None
    t = user_text.strip().lower()
    m = _ORDINAL_RE.search(t)
    if not m:
        # Also accept a bare number like "open 5"
        m2 = re.search(r"\b(\d{1,3})\b", t)
        if not m2:
            return None
        n = int(m2.group(1))
        return n if 1 <= n <= max_n else None

    # Group 1 = numeric part (e.g., 4 from '4th'); Group 2 = word ordinal
    if m.group(1):
        try:
            n = int(m.group(1))
            return n if 1 <= n <= max_n else None
        except ValueError:
            return None

    word = (m.group(2) or "").lower()
    n = _WORD_ORDINALS.get(word)
    return n if n and 1 <= n <= max_n else None






