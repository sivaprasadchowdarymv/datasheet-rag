"""
rag/llm.py — LLMEngine interface + implementations.

    LLMEngine
    ├── OllamaLLMEngine        LOCAL (localhost) or REMOTE (your Oracle VM) Ollama
    ├── OpenAICompatLLMEngine  any /v1/chat/completions server: llama.cpp `llama-server`,
    │                          vLLM, LM Studio, Ollama's /v1, or an optional free-tier
    │                          host of open-weight models
    └── (extractive)           no LLM at all — implemented in generator.py

Every engine yields text deltas (streaming). <think>…</think> blocks emitted by
reasoning models (deepseek-r1, qwen3) are filtered out before the user sees them.
A process-wide semaphore caps concurrent generations so one CPU box is not
overwhelmed by many simultaneous users.
"""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional

import httpx


class LLMError(Exception):
    """Human-readable generation failure."""


class _ThinkFilter:
    """Streams text while hiding <think>…</think> sections."""

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def feed(self, s: str) -> str:
        self.buf += s
        out = []
        while True:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    self.buf = self.buf[-8:]
                    break
                self.buf = self.buf[end + 8:]
                self.inside = False
            else:
                start = self.buf.find("<think>")
                if start < 0:
                    safe = len(self.buf) - 7          # keep a possible partial "<think" tag
                    if safe > 0:
                        out.append(self.buf[:safe])
                        self.buf = self.buf[safe:]
                    break
                out.append(self.buf[:start])
                self.buf = self.buf[start + 7:]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


class LLMEngine(ABC):
    provider = "base"
    _semaphores: Dict[int, threading.BoundedSemaphore] = {}

    def __init__(self, model: str, temperature: float, max_tokens: int, timeout: int, max_concurrent: int = 2):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._sem = LLMEngine._semaphores.setdefault(max_concurrent, threading.BoundedSemaphore(max_concurrent))

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def _stream(self, messages: List[Dict[str, str]]) -> Iterator[str]: ...

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        if not self._sem.acquire(timeout=self.timeout):
            raise LLMError("The server is busy answering other questions. Please try again in a minute.")
        try:
            filt = _ThinkFilter()
            for delta in self._stream(messages):
                out = filt.feed(delta)
                if out:
                    yield out
            tail = filt.flush()
            if tail:
                yield tail
        finally:
            self._sem.release()

    def complete(self, messages: List[Dict[str, str]]) -> str:
        return "".join(self.stream(messages)).strip()

    @abstractmethod
    def health(self) -> str:
        """'' when healthy, otherwise a human-readable problem."""


class OllamaLLMEngine(LLMEngine):
    provider = "ollama"

    def __init__(self, model: str, host: str, auth_header: str = "", num_ctx: int = 4096, **kw):
        super().__init__(model, **kw)
        try:
            import ollama
        except ImportError as e:
            raise LLMError("The 'ollama' package is not installed (pip install ollama).") from e
        kwargs = {"timeout": self.timeout}
        if auth_header:
            kwargs["headers"] = {"Authorization": auth_header}
        self._client = ollama.Client(host=host, **kwargs)
        self.host = host
        self.num_ctx = num_ctx

    def _stream(self, messages):
        try:
            for part in self._client.chat(
                model=self.model, messages=messages, stream=True,
                options={"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
            ):
                msg = part["message"] if isinstance(part, dict) else part.message
                content = msg["content"] if isinstance(msg, dict) else (msg.content or "")
                if content:
                    yield content
        except Exception as e:  # noqa: BLE001
            text = str(e).lower()
            if "not found" in text:
                raise LLMError(f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}") from e
            if "connect" in text or "refused" in text:
                raise LLMError(f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`.") from e
            if "timed out" in text or "timeout" in text:
                raise LLMError("The language model timed out. Try a smaller model or a shorter question.") from e
            raise LLMError(f"Ollama error: {e}") from e

    def health(self) -> str:
        try:
            resp = self._client.list()
            models = resp["models"] if isinstance(resp, dict) else resp.models
            names = set()
            for m in models:
                n = m.get("model") if isinstance(m, dict) else getattr(m, "model", "")
                names.update({n, n.split(":")[0]})
            if self.model not in names and self.model.split(":")[0] not in names:
                return f"Model '{self.model}' not pulled on {self.host}. Run: ollama pull {self.model}"
            return ""
        except Exception as e:  # noqa: BLE001
            return f"Ollama not reachable at {self.host} ({type(e).__name__})."


class OpenAICompatLLMEngine(LLMEngine):
    provider = "openai_compat"

    def __init__(self, model: str, base_url: str, api_key: str = "", **kw):
        super().__init__(model, **kw)
        if not base_url:
            raise LLMError("LLM_BASE_URL is empty. Set it to your server's …/v1 URL.")
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _stream(self, messages):
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=15)) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as r:
                    if r.status_code == 429:
                        raise LLMError("The model server is rate-limiting requests (HTTP 429). Try again shortly.")
                    if r.status_code >= 400:
                        body = r.read().decode("utf-8", "ignore")[:300]
                        raise LLMError(f"Model server returned HTTP {r.status_code}: {body}")
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if choices:
                            delta = (choices[0].get("delta") or {}).get("content")
                            if delta:
                                yield delta
        except LLMError:
            raise
        except httpx.TimeoutException as e:
            raise LLMError("The model server timed out.") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Cannot reach the model server at {self.base_url} ({type(e).__name__}).") from e

    def health(self) -> str:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=10)
            return "" if r.status_code < 400 else f"Model server answered HTTP {r.status_code}."
        except Exception as e:  # noqa: BLE001
            return f"Model server not reachable at {self.base_url} ({type(e).__name__})."


def create_llm_engine(settings) -> Optional[LLMEngine]:
    """Returns None for extractive mode (no LLM)."""
    p = settings.resolved_llm_provider
    common = dict(
        temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s, max_concurrent=settings.max_concurrent_generations,
    )
    if p == "extractive":
        return None
    if p == "ollama":
        return OllamaLLMEngine(
            settings.llm_model, host=settings.ollama_host, auth_header=settings.ollama_auth_header,
            num_ctx=settings.llm_num_ctx, **common,
        )
    if p == "openai_compat":
        return OpenAICompatLLMEngine(settings.llm_model, settings.llm_base_url, settings.llm_api_key, **common)
    raise LLMError(f"Unknown LLM_PROVIDER '{p}'. Use ollama | openai_compat | extractive | auto.")
