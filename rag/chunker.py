"""
rag/chunker.py — structural recursive chunking with parent-context injection.

Same idea as v3 (split at the coarsest structural boundary first: headings →
paragraphs → lines → sentences → characters; prefix every chunk with its heading
breadcrumb; overlap adjacent chunks). Three bugs from v3 are fixed:

1. v3 never MERGED small pieces, so a section without blank lines was split at
   "newline" level into one chunk per line (dozens of 5-word chunks). v4 packs
   adjacent pieces greedily up to MAX_CHUNK_TOKENS, like a standard recursive
   splitter.
2. v3 added overlap at every recursion level, so overlap compounded (a chunk
   could contain the tail of its parent AND of its sibling). v4 adds overlap
   exactly once, between final neighbouring chunks.
3. v3 read chunk sizes from module globals mutated by the sidebar. v4 takes
   explicit parameters, so the chunker signature can be stored with each doc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .text_utils import AVG_CHARS_PER_TOK, tok

# Ordered from coarsest to finest boundary (kept from v3).
SPLIT_PATTERNS: List[Tuple[str, str]] = [
    ("heading_caps", r"\n(?=[A-Z][A-Z0-9 \-/.]{4,}\n)"),
    ("heading_title", r"\n(?=[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,6}\n)"),
    ("heading_num", r"\n(?=\d+[.)]\s+[A-Z])"),
    ("paragraph", r"\n{2,}"),
    ("newline", r"\n"),
    ("sentence", r"(?<=[.!?;])\s+"),
]


@dataclass
class TextPiece:
    text: str
    depth: int          # split level at which this piece became small enough
    chunk_idx: int = 0  # sibling index inside its section


def _split(text: str, level: int, max_tokens: int) -> List[str]:
    if level >= len(SPLIT_PATTERNS):
        size = max_tokens * AVG_CHARS_PER_TOK
        return [text[i:i + size] for i in range(0, len(text), size)]
    parts = re.split(SPLIT_PATTERNS[level][1], text)
    return [p for p in parts if p and p.strip()]


def _pack(parts: List[str], max_tokens: int, joiner: str) -> List[str]:
    """Greedily merge adjacent parts while the result stays ≤ max_tokens."""
    packed: List[str] = []
    buf = ""
    for p in parts:
        candidate = f"{buf}{joiner}{p}" if buf else p
        if tok(candidate) <= max_tokens:
            buf = candidate
        else:
            if buf:
                packed.append(buf)
            buf = p
    if buf:
        packed.append(buf)
    return packed


def _recursive(text: str, level: int, max_tokens: int) -> List[TextPiece]:
    if tok(text) <= max_tokens:
        return [TextPiece(text.strip(), level)]
    if level > len(SPLIT_PATTERNS):
        return [TextPiece(text.strip(), level)]
    parts = _split(text, level, max_tokens)
    if len(parts) <= 1:
        return _recursive(text, level + 1, max_tokens)
    joiner = "\n\n" if level <= 3 else ("\n" if level == 4 else " ")
    out: List[TextPiece] = []
    for group in _pack(parts, max_tokens, joiner):
        if tok(group) <= max_tokens:
            out.append(TextPiece(group.strip(), level))
        else:
            out.extend(_recursive(group, level + 1, max_tokens))
    return out


def recursive_chunk(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
    min_tokens: int = 40,
) -> List[TextPiece]:
    """
    Split one section's text into ≤ max_tokens pieces.
    Returns pieces with depth (structural level) and sibling index; overlap is
    a word-aligned tail of the previous piece, added once.
    """
    text = (text or "").strip()
    if not text:
        return []
    pieces = [p for p in _recursive(text, 0, max_tokens) if p.text]

    # Merge a tiny trailing/leading fragment into its neighbour (avoids 1-line chunks).
    merged: List[TextPiece] = []
    for p in pieces:
        if merged and tok(p.text) < min_tokens and tok(merged[-1].text) + tok(p.text) <= max_tokens * 1.15:
            merged[-1] = TextPiece(merged[-1].text + "\n" + p.text, min(merged[-1].depth, p.depth))
        else:
            merged.append(p)

    overlap_chars = max(0, overlap_tokens) * AVG_CHARS_PER_TOK
    result: List[TextPiece] = []
    for i, p in enumerate(merged):
        body = p.text
        if i > 0 and overlap_chars:
            prev = merged[i - 1].text
            tail = prev[-overlap_chars:]
            cut = tail.find(" ")
            if 0 <= cut < len(tail) - 1:
                tail = tail[cut + 1:]          # start the overlap on a word boundary
            body = f"…{tail}\n{body}"
        result.append(TextPiece(body, p.depth, i))
    return result


def with_context_prefix(text: str, breadcrumb: str) -> str:
    """Parent-context injection (v3): the embedding sees where the text lives."""
    return f"[{breadcrumb}]\n{text}" if breadcrumb else text
