# src/data/time_processor.py
"""
Time parsing & duration indexing (no circular imports)
"""
from __future__ import annotations
import re
from typing import Dict, Optional, Tuple, List
import pandas as pd

TIME_COL_CANDIDATES = [
    "Time",
    "Time to implement",
    "Time to prepare/learn",
    "Duration",
]

MIN_RE = re.compile(r"\b(~|around|about|\≈|≈)?\s*(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)
HR_RE  = re.compile(r"\b(~|around|about|\≈|≈)?\s*(\d{1,2})\s*(hr|hrs|hour|hours|h)\b", re.I)
RANGE_MIN_RE = re.compile(r"\b(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)
RANGE_HR_RE  = re.compile(r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*(hr|hrs|hour|hours|h)\b", re.I)

# Ranges like "10 to 15 min" (minutes and hours)
RANGE_MIN_TO_RE = re.compile(r"\b(\d{1,3})\s*(?:to|–|—|-)\s*(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)
RANGE_HR_TO_RE  = re.compile(r"\b(\d{1,2})\s*(?:to|–|—|-)\s*(\d{1,2})\s*(hr|hrs|hour|hours|h)\b", re.I)

# "between 10 and 15 minutes"
BETWEEN_MIN_RE = re.compile(r"\bbetween\s+(\d{1,3})\s+and\s+(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)
BETWEEN_HR_RE  = re.compile(r"\bbetween\s+(\d{1,2})\s+and\s+(\d{1,2})\s*(hr|hrs|hour|hours|h)\b", re.I)

# "up to 15 minutes" / "at most 15 minutes" / "no more than 15 minutes"
UP_TO_RE    = re.compile(r"\b(up\s*to|at\s*most|no\s*more\s*than)\s+(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)
AT_LEAST_RE = re.compile(r"\b(at\s*least|no\s*less\s*than)\s+(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)

UNDER_RE  = re.compile(r"\b(under|less\s*than|≤)\s*(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)
OVER_RE   = re.compile(r"\b(over|more\s*than|≥)\s*(\d{1,3})\s*(min|mins|minute|minutes|m)\b", re.I)

APPROX_TAG = re.compile(r"\b(around|about|~|approx|approximately)\b", re.I)

class TimeProcessor:
    def __init__(self):
        # Maps row_index -> minutes (int) or None if unknown
        self._duration_cache: Dict[int, Optional[int]] = {}

    # -------------------------------
    # Public: Query-time parsing
    # -------------------------------
    def desired_time_from_query(self, text: str) -> Optional[int]:
        """
        Extract a single target time (in minutes) from free text.
        Rules:
          - Prefer explicit minute mentions (10 min)
          - Handle ranges (10-15 min) -> lower bound
          - Handle hours (1 hour -> 60)
          - Handle 'under 10 min' -> 10
          - If multiple, take the first reasonable one
        Returns None if nothing sensible is found.
        """
        # Early exit if no time tokens detected
        TIME_TOKENS = re.compile(r"\b(min|mins|minutes|hour|hours|hr|hrs)\b", re.I)
        if not TIME_TOKENS.search(text or ""):
            return None  # prevents flex mode without a time
            
        t = text.lower()

        # Ranges first (minutes)
        m = RANGE_MIN_RE.search(t)
        if m:
            low = int(m.group(1))
            return low

        # Ranges in hours
        m = RANGE_HR_RE.search(t)
        if m:
            low_hr = int(m.group(1))
            return low_hr * 60

        # Under / less than
        m = UNDER_RE.search(t)
        if m:
            return int(m.group(2))

        # Over / more than (we still return that number so your code can "nearest >=" if you want later)
        m = OVER_RE.search(t)
        if m:
            return int(m.group(2))

        # Single minutes
        m = MIN_RE.search(t)
        if m:
            return int(m.group(2))

        # Single hours
        m = HR_RE.search(t)
        if m:
            return int(m.group(2)) * 60

        # Loose "number" without units + "minute" word somewhere
        m_num = re.search(r"\b(\d{1,3})\b", t)
        if m_num and ("min" in t or "minute" in t):
            return int(m_num.group(1))

        # Nothing useful
        return None

    def desired_time_range_from_query(self, text: str) -> Optional[Tuple[int, int]]:
        """
        Extract a desired (lo, hi) range in minutes from free text.
        Returns (lo, hi). Either bound may be None if only one-sided.
        Examples:
          "10–15 min" -> (10, 15)
          "10 to 15 minutes" -> (10, 15)
          "between 10 and 15 minutes" -> (10, 15)
          "up to 15 minutes" / "at most 15 minutes" -> (0, 15)
          "at least 10 minutes" -> (10, None)
          "under 10 min" -> (0, 10)
          "over 30 min" -> (30, None)
        """
        # Early exit if no time tokens detected
        TIME_TOKENS = re.compile(r"\b(min|mins|minutes|hour|hours|hr|hrs)\b", re.I)
        if not TIME_TOKENS.search(text or ""):
            return None  # prevents flex mode without a time
            
        t = text.lower()

        # Explicit minute ranges (dash & to)
        m = RANGE_MIN_RE.search(t) or RANGE_MIN_TO_RE.search(t) or BETWEEN_MIN_RE.search(t)
        if m:
            low = int(m.group(1))
            high = int(m.group(2))
            return (min(low, high), max(low, high))

        # Explicit hour ranges (dash & to)
        m = RANGE_HR_RE.search(t) or RANGE_HR_TO_RE.search(t) or BETWEEN_HR_RE.search(t)
        if m:
            low = int(m.group(1)) * 60
            high = int(m.group(2)) * 60
            return (min(low, high), max(low, high))

        # Upper-bounded forms ("up to", "at most", "no more than", "under", "less than", "≤")
        m = UP_TO_RE.search(t) or UNDER_RE.search(t)
        if m:
            # UP_TO_RE uses group(2); UNDER_RE uses group(2) – both minutes
            hi = int(m.group(2))
            return (0, hi)

        # Lower-bounded forms ("at least", "no less than", "over", "more than", "≥")
        m = AT_LEAST_RE.search(t) or OVER_RE.search(t)
        if m:
            lo = int(m.group(2))
            return (lo, None)

        # Fallback: single time -> treat as a narrow band later (engine)
        return None

    # -------------------------------
    # Public: Build & store cache
    # -------------------------------
    def build_duration_cache(self, df: pd.DataFrame) -> Dict[int, Optional[int]]:
        """
        Build a per-row duration cache (in minutes) from likely time columns.
        Strategy:
          - prefer 'Time' / 'Time to implement'
          - parse first parsable value per row
          - return dict[row_index] = minutes or None
        """
        cache: Dict[int, Optional[int]] = {}
        if df is None or df.empty:
            self._duration_cache = cache
            return cache

        # Pre-compile a simple function to parse a cell
        def parse_cell(val: str) -> Optional[int]:
            if not val:
                return None
            s = str(val).strip().lower()
            if not s or "depend" in s or "varies" in s:
                return None

            # Try ranges
            m = RANGE_MIN_RE.search(s)
            if m:
                return int(m.group(1))

            m = RANGE_HR_RE.search(s)
            if m:
                return int(m.group(1)) * 60

            # Under/over
            m = UNDER_RE.search(s)
            if m:
                return int(m.group(2))

            m = OVER_RE.search(s)
            if m:
                return int(m.group(2))

            # Minutes
            m = MIN_RE.search(s)
            if m:
                return int(m.group(2))

            # Hours
            m = HR_RE.search(s)
            if m:
                return int(m.group(2)) * 60

            # Bare numbers with the word minute elsewhere
            if "min" in s or "minute" in s:
                m = re.search(r"\b(\d{1,3})\b", s)
                if m:
                    return int(m.group(1))

            # Hours bare number with hour word elsewhere
            if "hour" in s or "hr" in s:
                m = re.search(r"\b(\d{1,2})\b", s)
                if m:
                    return int(m.group(1)) * 60

            return None

        cols = [c for c in TIME_COL_CANDIDATES if c in df.columns]
        for i in range(len(df)):
            minutes = None
            for c in cols:
                v = str(df.iloc[i][c]) if c in df.columns else ""
                minutes = parse_cell(v)
                if minutes is not None:
                    break
            cache[i] = minutes

        self._duration_cache = cache
        return cache

    def set_duration_cache(self, cache: Dict[int, Optional[int]]) -> None:
        self._duration_cache = cache or {}

    # -------------------------------
    # Public: Querying by time
    # -------------------------------
    def indices_with_time_constraint(
        self,
        target_minutes: int,
        fuzzy: bool = True,
        tol: float = 0.2,
    ) -> set[int]:
        """
        Return a SET of row indices whose durations match:
          - if fuzzy=True: within ±tol of target (default ±20%)
          - if fuzzy=False: exactly equal to target
        """
        if not self._duration_cache:
            return set()

        matches: set[int] = set()
        if fuzzy:
            low = max(0, int(round(target_minutes * (1.0 - tol))))
            high = int(round(target_minutes * (1.0 + tol)))
            for idx, mins in self._duration_cache.items():
                if mins is None:
                    continue
                if low <= mins <= high:
                    matches.add(idx)
        else:
            for idx, mins in self._duration_cache.items():
                if mins is None:
                    continue
                if mins == target_minutes:
                    matches.add(idx)
        return matches

    def nearest_by_time(self, target_minutes: int, k: int = 100) -> List[Tuple[int, int]]:
        """
        Return up to k (index, abs_diff) pairs sorted by proximity to target_minutes.
        Rows with unknown times are skipped.
        """
        if not self._duration_cache:
            return []

        diffs: List[Tuple[int, int]] = []
        for idx, mins in self._duration_cache.items():
            if mins is None:
                continue
            diffs.append((idx, abs(mins - target_minutes)))

        diffs.sort(key=lambda x: x[1])
        return diffs[:k]

    def exact_then_nearest(self, target_minutes: int, k: int = 8) -> Tuple[List[int], List[int]]:
        """
        Convenience helper: return (exact_list, nearest_list) of indices.
        exact_list: rows exactly matching target
        nearest_list: remaining nearest (excluding exact), up to k - len(exact_list)
        """
        exact = [i for i, mins in self._duration_cache.items() if mins == target_minutes]
        if len(exact) >= k:
            return exact[:k], []

        remaining = k - len(exact)
        nearest = [i for (i, _) in self.nearest_by_time(target_minutes, k=100) if i not in exact][:remaining]
        return exact, nearest

    def indices_in_range(self, lo: Optional[int], hi: Optional[int]) -> set[int]:
        """
        Return indices whose cached minutes fall within [lo, hi].
        If a bound is None, treat it as open-ended on that side.
        """
        if not self._duration_cache:
            return set()

        matches: set[int] = set()
        for idx, mins in self._duration_cache.items():
            if mins is None:
                continue
            if lo is not None and mins < lo:
                continue
            if hi is not None and mins > hi:
                continue
            matches.add(idx)
        return matches

    def band_distance(self, minutes: Optional[int], lo: Optional[int], hi: Optional[int]) -> int:
        """
        Distance to a [lo, hi] band (0 if inside). Unknown minutes -> large penalty.
        """
        if minutes is None:
            return 10**9
        if lo is not None and minutes < lo:
            return lo - minutes
        if hi is not None and minutes > hi:
            return minutes - hi
        return 0


# Global instance
time_processor = TimeProcessor()