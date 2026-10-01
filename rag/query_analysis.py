"""
rag/query_analysis.py — query preprocessing (cheap, rule-based, no LLM call).

Produces:
  • modality weights  — "pin configuration" prefers PIN/TABLE evidence,
                        "equation" prefers EQUATION, "graph" prefers FIGURE …
  • key terms         — content tokens used for lexical coverage gating
  • symbols           — part numbers / electrical symbols that must match exactly
  • retrieval query   — follow-ups ("and its max?") are expanded with the
                        previous question so retrieval has a subject
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .text_utils import PART_NUMBER_RE, content_words, tokenize

_INTENTS = [
    # (name, regex, weights)
    ("pin", r"\b(pin(?:out|s)?|pad|ball|gpio|terminal|signal names?|package pins?|pin ?map|footprint)\b",
     {"pin": 1.6, "row": 1.3, "table": 1.25, "figure": 1.05}),
    ("spec", r"\b(max(?:imum)?|min(?:imum)?|typ(?:ical)?|rating|rated|absolute|limit|range|tolerance|"
             r"current|voltage|power|frequency|temperature|resistance|capacitance|dissipation|"
             r"supply|threshold|leakage|quiescent|efficiency|speed|bandwidth|accuracy|value)\b",
     {"row": 1.35, "table": 1.3, "pin": 1.05, "text": 1.0}),
    ("equation", r"\b(equation|formula|calculat\w*|compute|derive|expression|relationship between)\b",
     {"equation": 1.6, "text": 1.1}),
    ("figure", r"\b(figure|fig\.?|graph|plot|curve|chart|waveform|diagram|characteristic(?:s)? curve|"
               r"timing diagram|block diagram|vs\.?|versus|image|picture|drawing)\b",
     {"figure": 1.7, "text": 1.0}),
    ("compare", r"\b(compare|comparison|difference|differ|versus|vs\.?|which (?:one|datasheet|part|device)|"
                r"highest|lowest|better|best|all (?:the )?(?:parts|devices|datasheets))\b",
     {}),
]

_FOLLOWUP_RE = re.compile(r"^(and|what about|how about|also|its?|their|they|that|this|those|these|same)\b", re.I)
_PRONOUN_RE = re.compile(r"\b(it|its|they|their|them|that|this|those|these|the same|above)\b", re.I)
_SYMBOL_RE = re.compile(r"\b[VITPRCFt][A-Z]{1,6}[0-9]*\b|\b[A-Z]{2,}[0-9]+[A-Z0-9]*\b")


@dataclass
class QueryAnalysis:
    original: str
    retrieval_query: str
    intents: List[str] = field(default_factory=list)
    modality_weights: Dict[str, float] = field(default_factory=dict)
    key_terms: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    is_comparison: bool = False

    def weight(self, modality: str) -> float:
        return self.modality_weights.get(modality, 1.0)

    def preferred_modalities(self) -> List[str]:
        return [m for m, w in sorted(self.modality_weights.items(), key=lambda x: -x[1]) if w >= 1.25]


def analyze_query(query: str, history: Optional[List[str]] = None) -> QueryAnalysis:
    q = query.strip()
    retrieval_query = q
    if history:
        prev = history[-1]
        if len(tokenize(q)) <= 4 or _FOLLOWUP_RE.match(q) or _PRONOUN_RE.search(q):
            retrieval_query = f"{prev} {q}"

    intents: List[str] = []
    weights: Dict[str, float] = {}
    for name, pat, w in _INTENTS:
        if re.search(pat, retrieval_query, re.I):
            intents.append(name)
            for m, v in w.items():
                weights[m] = max(weights.get(m, 1.0), v)

    symbols = list(dict.fromkeys(PART_NUMBER_RE.findall(retrieval_query) + _SYMBOL_RE.findall(retrieval_query)))
    key_terms = sorted(content_words(q) if q else set(), key=len, reverse=True)[:12]
    return QueryAnalysis(
        original=q,
        retrieval_query=retrieval_query,
        intents=intents,
        modality_weights=weights,
        key_terms=key_terms,
        symbols=symbols[:8],
        is_comparison="compare" in intents,
    )
