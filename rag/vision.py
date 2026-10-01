"""
rag/vision.py — on-demand figure understanding.

v3 behaviour kept: LLaVA (or any vision model) describes a figure only when
the question looks visual or nothing else matched.

v4 improvements:
  • the figure is chosen by the RETRIEVER (caption + nearby text + section),
    not by fuzzy-matching the first caption on the page
  • the description is query-independent, so it is cached permanently with the
    chunk (keyed by vision model) and re-embedded → later questions can find
    the figure through its described content (axis names, curves, values)
  • at most MAX_VISION_FIGURES figures per question
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import httpx

VISION_PROMPT = (
    "This image is a figure from an electronic component datasheet. Describe it precisely for an engineer: "
    "figure type (graph, timing diagram, block diagram, package drawing, schematic), title, axis names and "
    "units with ranges, each curve/trace and its condition label (e.g. temperature, supply voltage), notable "
    "numeric values or crossing points, pin or block labels. Report only what is visible. "
    "Write 'unreadable' for anything you cannot read. Do not guess values."
)


class VisionEngine(ABC):
    provider = "base"

    def __init__(self, model: str, timeout: int = 180):
        self.model = model
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def describe(self, image_path: Path, caption: str = "") -> str: ...


class OllamaVisionEngine(VisionEngine):
    provider = "ollama"

    def __init__(self, model: str, host: str, auth_header: str = "", **kw):
        super().__init__(model, **kw)
        import ollama

        kwargs = {"timeout": self.timeout}
        if auth_header:
            kwargs["headers"] = {"Authorization": auth_header}
        self._client = ollama.Client(host=host, **kwargs)

    def describe(self, image_path: Path, caption: str = "") -> str:
        prompt = VISION_PROMPT + (f"\nCaption printed under the figure: {caption}" if caption else "")
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
            options={"temperature": 0.0},
        )
        msg = resp["message"] if isinstance(resp, dict) else resp.message
        return (msg["content"] if isinstance(msg, dict) else msg.content or "").strip()


class OpenAICompatVisionEngine(VisionEngine):
    provider = "openai_compat"

    def __init__(self, model: str, base_url: str, api_key: str = "", **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def describe(self, image_path: Path, caption: str = "") -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        prompt = VISION_PROMPT + (f"\nCaption printed under the figure: {caption}" if caption else "")
        payload = {
            "model": self.model, "temperature": 0.0, "max_tokens": 600,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def create_vision_engine(settings) -> Optional[VisionEngine]:
    if not settings.enable_vision or settings.vision_provider in ("", "none") or not settings.vision_model:
        return None
    if settings.vision_provider == "ollama":
        return OllamaVisionEngine(settings.vision_model, host=settings.ollama_host,
                                  auth_header=settings.ollama_auth_header, timeout=settings.llm_timeout_s)
    if settings.vision_provider == "openai_compat":
        return OpenAICompatVisionEngine(settings.vision_model, settings.llm_base_url, settings.llm_api_key,
                                        timeout=settings.llm_timeout_s)
    return None
