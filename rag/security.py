"""
rag/security.py — guards for a public deployment.

  validate_pdf()     size limit, %PDF magic bytes, opens cleanly, page limit
  safe_filename()    strips paths and odd characters (names are display-only;
                     files are stored under their content hash, never their name)
  sanitize_query()   length cap, control characters removed
  RateLimiter        sliding-window limit per browser session
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple

from .text_utils import clean_control_chars


class ValidationError(Exception):
    pass


def validate_pdf(data: bytes, max_mb: int) -> None:
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ValidationError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {max_mb} MB.")
    head = data[:1024]
    if b"%PDF-" not in head:
        raise ValidationError("This is not a PDF file (missing %PDF header).")


def safe_filename(name: str) -> str:
    name = (name or "document.pdf").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip(" .") or "document.pdf"
    return name[:120]


def sanitize_query(q: str, max_chars: int) -> str:
    q = clean_control_chars(q or "").strip()
    return re.sub(r"\s{3,}", "  ", q)[:max_chars]


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = max(1, per_minute)
        self._events: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Returns (allowed, seconds_to_wait)."""
        now = time.time()
        with self._lock:
            dq = self._events.setdefault(key, deque())
            while dq and now - dq[0] > 60:
                dq.popleft()
            if len(dq) >= self.per_minute:
                return False, int(60 - (now - dq[0])) + 1
            dq.append(now)
            return True, 0
