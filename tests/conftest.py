"""
Shared test fixtures: synthetic datasheet PDFs and an offline pipeline.

All tests run WITHOUT Ollama, without model downloads and without network:
APP_MODE=test uses the hash embedder, the lexical reranker and extractive
answers. A tiny fake LLM is provided to test the generation path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, List

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _grid_table(page, x0, y0, col_w, row_h, rows):
    n_cols = len(rows[0])
    x1 = x0 + sum(col_w)
    y1 = y0 + row_h * len(rows)
    for r in range(len(rows) + 1):
        page.draw_line((x0, y0 + r * row_h), (x1, y0 + r * row_h), width=0.8)
    x = x0
    for c in range(n_cols + 1):
        page.draw_line((x, y0), (x, y1), width=0.8)
        if c < n_cols:
            x += col_w[c]
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            font = "hebo" if r == 0 else "helv"
            page.insert_text((x + 4, y0 + r * row_h + row_h - 6), cell, fontsize=9, fontname=font)
            x += col_w[c]
    return y1


def make_datasheet(part: str, vmax: str, imax: str, tmax: str, pins: List[List[str]], dropout: str) -> bytes:
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    p.insert_text((50, 60), f"{part} Adjustable Voltage Regulator", fontsize=18, fontname="hebo")
    p.insert_text((50, 95), "1. Features", fontsize=13, fontname="hebo")
    y = 115
    for line in [
        f"The {part} is an adjustable three-terminal positive voltage regulator.",
        f"It supplies more than {imax} A over an output-voltage range of 1.25 V to 37 V.",
        "Line regulation is typically 0.01 %/V and load regulation 0.1 %.",
        "Internal current limiting and thermal overload protection are included.",
    ]:
        p.insert_text((50, y), line, fontsize=10, fontname="helv")
        y += 14
    p.insert_text((50, y + 20), "2. Absolute Maximum Ratings", fontsize=13, fontname="hebo")
    rows = [
        ["Parameter", "Symbol", "Min", "Max", "Unit"],
        ["Input-output voltage differential", "VI-VO", "-0.3", vmax, "V"],
        ["Output current", "IOUT", "-", imax, "A"],
        ["Operating junction temperature", "TJ", "-40", tmax, "C"],
        ["Storage temperature", "TSTG", "-65", "150", "C"],
    ]
    yt = _grid_table(p, 50, y + 35, [210, 70, 50, 50, 50], 20, rows)
    p.insert_text((50, yt + 30), "3. Output Voltage Setting", fontsize=13, fontname="hebo")
    p.insert_text((50, yt + 50), "The output voltage is set by two external resistors:", fontsize=10, fontname="helv")
    p.insert_text((70, yt + 70), "VOUT = VREF * (1 + R2 / R1) + IADJ * R2      (1)", fontsize=10, fontname="helv")
    p.insert_text((50, yt + 90), "where VREF is 1.25 V and IADJ is typically 50 uA.", fontsize=10, fontname="helv")

    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 60), "4. Pin Configuration", fontsize=13, fontname="hebo")
    _grid_table(p2, 50, 80, [60, 90, 220], 20, [["Pin", "Name", "Function"]] + pins)
    p2.insert_text((50, 230), "5. Typical Performance Characteristics", fontsize=13, fontname="hebo")
    # a vector plot (axes + curve) — most datasheet graphs are vector drawings
    ox, oy, w, h = 90, 470, 300, 200
    p2.draw_line((ox, oy), (ox + w, oy), width=1)
    p2.draw_line((ox, oy), (ox, oy - h), width=1)
    pts = [(ox + i * 30, oy - 20 - i * i * 1.6) for i in range(11)]
    for a, b in zip(pts, pts[1:]):
        p2.draw_line(a, b, width=1.5)
    for i in range(6):
        p2.draw_line((ox + i * 60, oy), (ox + i * 60, oy + 5), width=0.5)
    p2.insert_text((ox + 80, oy + 20), "Output current (A)", fontsize=8, fontname="helv")
    p2.insert_text((50, oy + 45), f"Figure 1. Dropout voltage vs output current (dropout {dropout} V at full load)",
                   fontsize=9, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="session")
def pdf_a() -> bytes:
    return make_datasheet("LM317", "40", "1.5", "125",
                          [["1", "ADJ", "Adjustment input"], ["2", "VOUT", "Regulated output"], ["3", "VIN", "Unregulated input"]],
                          "2.5")


@pytest.fixture(scope="session")
def pdf_b() -> bytes:
    return make_datasheet("LT1085", "30", "3.0", "150",
                          [["1", "ADJ", "Adjust pin"], ["2", "OUT", "Output"], ["3", "IN", "Input supply"]],
                          "1.3")


@pytest.fixture()
def settings(tmp_path):
    os.environ["APP_MODE"] = "test"
    from config import load_settings

    return load_settings("test").with_overrides(
        data_dir=str(tmp_path / "data"), cache_dir=str(tmp_path / "cache"), storage_backend="local",
    )


@pytest.fixture()
def pipeline(settings):
    from rag.pipeline import RAGPipeline, build_engines

    return RAGPipeline(settings, build_engines(settings))


class FakeLLM:
    """Deterministic stand-in for an LLM: cites REF-1 and copies its first value."""
    provider = "fake"
    model = "fake-1"
    name = "fake:fake-1"

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = 0

    def stream(self, messages) -> Iterator[str]:
        self.calls += 1
        if self.reply is not None:
            yield self.reply
            return
        user = messages[-1]["content"]
        block = user.split("[REF-1]", 1)[1].split(">>>", 1)[0]
        line = [l for l in block.splitlines() if any(ch.isdigit() for ch in l)][-1].strip()
        for word in f"According to the datasheet: {line} [REF-1]".split(" "):
            yield word + " "

    def health(self) -> str:
        return ""
