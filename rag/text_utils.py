"""
rag/text_utils.py — small, dependency-free text helpers used everywhere.

Contains the v3 `extract_metadata()` (kept, with broader unit coverage) and a
tokenizer that understands datasheet tokens such as "3.3V", "I2C", "GPIO12",
"VCC", "µA" — ordinary word tokenizers destroy these and hurt keyword search.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Set

AVG_CHARS_PER_TOK = 4

STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or", "for",
    "on", "at", "by", "with", "this", "that", "it", "be", "as", "what", "how", "does",
    "do", "which", "who", "when", "where", "why", "can", "could", "should", "would",
    "i", "me", "my", "you", "your", "we", "our", "there", "their", "its", "from", "into",
    "about", "tell", "give", "show", "list", "please", "any", "all", "some", "if", "than",
    "then", "these", "those", "has", "have", "had", "will", "shall", "may", "might",
    "between", "much", "many", "datasheet", "document", "pdf",
}


def tok(text: str) -> int:
    """Fast token-count approximation (same rule as v3)."""
    return max(1, len(text or "") // AVG_CHARS_PER_TOK)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return (
        text.replace("μ", "µ").replace("Ω", "Ω").replace("–", "-").replace("—", "-")
        .replace("−", "-").replace("\u00a0", " ")
    )


_TOKEN_RE = re.compile(r"[a-z0-9µ°±%]+(?:[.,][0-9]+)?[a-zµΩ°%]*", re.IGNORECASE)


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """
    Lower-cased technical tokens. "3.3V" → ["3.3v", "3.3", "v"],
    "GPIO12" → ["gpio12", "gpio", "12"]. The composite token preserves exact
    matches; the parts give partial recall.
    """
    out: List[str] = []
    for m in _TOKEN_RE.finditer(normalize(text).lower()):
        t = m.group(0).replace(",", ".")
        if not keep_stopwords and t in STOPWORDS:
            continue
        out.append(t)
        parts = re.findall(r"[0-9]+(?:\.[0-9]+)?|[a-zµΩ°%]+", t)
        if len(parts) > 1:
            out.extend(p for p in parts if keep_stopwords or p not in STOPWORDS)
    return out


def content_words(text: str) -> Set[str]:
    return {t for t in tokenize(text) if len(t) > 1}


# ---------------------------------------------------------------------------
# Metadata extraction (v3 feature, kept and extended)
# ---------------------------------------------------------------------------
_META_PATTERNS = {
    "voltage": r"\bV[A-Z0-9_()]{0,8}\s*(?:=|≤|≥|<|>)\s*[±\d.\-+]+\s*[mk]?V\b",
    "temperature": r"\bT[A-Z0-9_]{0,4}\s*(?:=|≤|≥|<|>)\s*[±\d.\-+]+\s*°?C\b",
    "current": r"\bI[A-Z0-9_()]{0,8}\s*(?:=|≤|≥|<|>)\s*[±\d.\-+]+\s*(?:mA|µA|uA|nA|A)\b",
    "frequency": r"\bf[A-Z0-9_]{0,6}\s*(?:=|≤|≥|<|>)\s*[\d.]+\s*(?:GHz|MHz|kHz|Hz)\b",
    "power": r"\bP[A-Z0-9_]{0,4}\s*(?:=|≤|≥|<|>)\s*[\d.]+\s*(?:mW|µW|uW|W)\b",
    "resistance": r"\bR[A-Z0-9_]{0,4}\s*(?:=|≤|≥|<|>)\s*[\d.]+\s*(?:kΩ|MΩ|mΩ|Ω|ohms?)\b",
    "capacitance": r"\bC[A-Z0-9_]{0,4}\s*(?:=|≤|≥|<|>)\s*[\d.]+\s*(?:pF|nF|µF|uF|mF|F)\b",
}
_VALUE_RE = re.compile(
    r"[±]?\d+(?:\.\d+)?\s*(?:GHz|MHz|kHz|Hz|mV|kV|V|mA|µA|uA|nA|A|mW|µW|W|kΩ|MΩ|mΩ|Ω|pF|nF|µF|uF|°C|ns|µs|us|ms|s|%|dB|ppm)\b"
)
PART_NUMBER_RE = re.compile(r"\b[A-Z]{1,5}[0-9]{2,8}[A-Z0-9\-]*\b")


def extract_metadata(text: str) -> Dict[str, Any]:
    text = normalize(text)
    meta: Dict[str, Any] = {}
    for key, pat in _META_PATTERNS.items():
        found = re.findall(pat, text, re.IGNORECASE)
        if found:
            meta[key] = list(dict.fromkeys(f.strip() for f in found))[:8]
    vals = _VALUE_RE.findall(text)
    if vals:
        meta["values"] = list(dict.fromkeys(v.replace(" ", "") for v in vals))[:12]
    pn = PART_NUMBER_RE.findall(text)
    if pn:
        meta["part_numbers"] = list(dict.fromkeys(pn))[:10]
    return meta


def numbers_with_units(text: str) -> List[str]:
    """Normalised value strings ("3.3v", "500ma") for grounding checks."""
    return [v.replace(" ", "").lower() for v in _VALUE_RE.findall(normalize(text))]


def bare_numbers(text: str) -> List[str]:
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", normalize(text))


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(•\-*])|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def clean_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
