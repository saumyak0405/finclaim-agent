"""LLM clients.

Groq and Ollama both expose an OpenAI-compatible /chat/completions endpoint,
so one stdlib-only client covers both. `FallbackLLM` tries providers in
order (e.g. Groq free tier -> local Ollama) and `call_json` turns free-text
model output into validated JSON with bounded repair retries.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1200) -> str: ...


@dataclass
class OpenAICompatClient:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    name: str = "openai-compat"

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1200) -> str:
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:  # 429 / 5xx are retryable
                last = e
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise LLMError(f"{self.name}: HTTP {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
            time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"{self.name}: failed after {self.max_retries} attempts: {last}")


@dataclass
class FallbackLLM:
    providers: list[LLM]
    name: str = "fallback"
    used: list[str] = field(default_factory=list)

    def complete(self, messages: list[Message], **kw: Any) -> str:
        errors = []
        for p in self.providers:
            try:
                out = p.complete(messages, **kw)
                self.used.append(p.name)
                return out
            except LLMError as e:
                errors.append(str(e))
        raise LLMError("all providers failed: " + " | ".join(errors))


@dataclass
class ScriptedLLM:
    """Deterministic LLM for tests: returns queued responses or calls a function."""
    responses: list[str] | None = None
    fn: Callable[[list[Message]], str] | None = None
    name: str = "scripted"
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: list[Message], **kw: Any) -> str:
        self.calls.append(messages)
        if self.fn is not None:
            return self.fn(messages)
        if not self.responses:
            raise LLMError("scripted LLM exhausted")
        return self.responses.pop(0)


def groq(model: str = "llama-3.3-70b-versatile") -> OpenAICompatClient:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    return OpenAICompatClient("https://api.groq.com/openai/v1", model, key, name=f"groq:{model}")


def ollama(model: str = "llama3.1:8b") -> OpenAICompatClient:
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
    return OpenAICompatClient(base, model, None, timeout=180, name=f"ollama:{model}")


def build_llm(provider: str) -> LLM:
    if provider == "groq":
        return groq(os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    if provider == "ollama":
        return ollama(os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    if provider == "auto":
        chain: list[LLM] = []
        if os.environ.get("GROQ_API_KEY"):
            chain.append(groq(os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")))
        chain.append(ollama(os.environ.get("OLLAMA_MODEL", "llama3.1:8b")))
        return FallbackLLM(chain)
    if provider == "mock":
        from .mock_llm import MockLLM
        return MockLLM()
    raise ValueError(f"unknown provider {provider!r}")


# ---------------------------------------------------------------- JSON calls
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Pull the first JSON object out of model text (handles ``` fences and chatter)."""
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start:i + 1])
    raise ValueError("unbalanced JSON object")


def call_json(llm: LLM, messages: list[Message], validate: Callable[[Any], str | None],
              retries: int = 2, on_call: Callable[[], None] | None = None) -> Any:
    """Call the model and return validated JSON.

    `validate(obj)` returns None when valid or an error string, which is fed
    back to the model so it can repair its own output (bounded by `retries`).
    """
    convo = list(messages)
    last_err = ""
    for _ in range(retries + 1):
        if on_call:
            on_call()
        raw = llm.complete(convo)
        try:
            obj = extract_json(raw)
            err = validate(obj)
        except (ValueError, json.JSONDecodeError) as e:
            obj, err = None, f"invalid JSON: {e}"
        if err is None:
            return obj
        last_err = err
        convo = convo + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Your output was rejected: {err}. Reply with ONLY a corrected JSON object."},
        ]
    raise LLMError(f"model never produced valid JSON: {last_err}")
