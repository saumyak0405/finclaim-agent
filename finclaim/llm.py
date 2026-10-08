"""LLM clients.

Groq and Ollama both expose an OpenAI-compatible /chat/completions endpoint,
so one stdlib-only client covers both. `FallbackLLM` tries providers in
order (e.g. Groq free tier -> local Ollama) and `call_json` turns free-text
model output into validated JSON with bounded repair retries.
"""
from __future__ import annotations

import json
import os
import random
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

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 4096,
                 tools: list[dict[str, Any]] | None = None) -> str: ...


@dataclass
class OpenAICompatClient:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 6
    max_backoff_s: float = 60.0
    name: str = "openai-compat"

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 4096,
                 tools: list[dict[str, Any]] | None = None) -> str:
        """Return the model's reply as text.

        With `tools`, the request uses native function calling with tool_choice=required and the
        chosen call is returned as a JSON action string, so callers never see the wire format.
        Reasoning models (e.g. gpt-oss) sometimes emit a native tool call even when no tools were
        declared; Groq rejects that with 400 tool_use_failed but includes the attempted call, which
        is recovered here instead of failing the run.
        """
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        body = json.dumps(payload).encode()
        # explicit UA: Groq's Cloudflare front door rejects the default Python-urllib agent (error 1010)
        headers = {"Content-Type": "application/json", "User-Agent": "finclaim-agent/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                if msg.get("tool_calls"):
                    return tool_call_to_action(msg["tool_calls"][0].get("function", {}), msg.get("content"))
                return msg.get("content") or ""
            except urllib.error.HTTPError as e:  # 429 / 5xx are retryable
                last = e
                detail = e.read()
                if e.code == 400:
                    recovered = recover_failed_generation(detail)
                    if recovered is not None:
                        return recovered
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise LLMError(f"{self.name}: HTTP {e.code}: {detail[:300]!r}") from e
                wait = retry_after_seconds(e.headers)
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                last, wait = e, None
            if attempt < self.max_retries - 1:
                # per-minute rate limits need real waits: honour the server's hint, else exponential backoff
                backoff = min(2 ** attempt, self.max_backoff_s)
                time.sleep(min(max(wait or 0.0, backoff), self.max_backoff_s) + random.uniform(0, 0.5))
        raise LLMError(f"{self.name}: failed after {self.max_retries} attempts: {last}")


def retry_after_seconds(headers: Any) -> float | None:
    """Seconds to wait from Retry-After or Groq's x-ratelimit-reset-* headers ('7.66s', '1m2.5s', '450ms')."""
    if headers is None:
        return None
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        val = headers.get(key)
        if not val:
            continue
        try:
            return float(val)
        except ValueError:
            pass
        total, found = 0.0, False
        for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", val):
            total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
            found = True
        if found:
            return total
    return None


def tool_call_to_action(fn: dict[str, Any], content: str | None = None) -> str:
    """Native tool call -> the agent's JSON action text. Namespaces like 'functions.' are stripped."""
    name = str(fn.get("name", "")).split(".")[-1]
    raw = fn.get("arguments", {})
    if isinstance(raw, str):
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": raw}
    else:
        args = raw
    return json.dumps({"thought": (content or "").strip()[:500], "action": "call_tool", "tool": name, "args": args})


def recover_failed_generation(body: bytes) -> str | None:
    try:
        err = json.loads(body).get("error", {})
    except (json.JSONDecodeError, AttributeError):
        return None
    if err.get("code") != "tool_use_failed" or not err.get("failed_generation"):
        return None
    gen = err["failed_generation"]
    try:
        obj = json.loads(gen)
    except json.JSONDecodeError:
        return gen
    if isinstance(obj, dict) and "name" in obj:
        return tool_call_to_action({"name": obj["name"], "arguments": obj.get("arguments", obj.get("parameters", {}))})
    return gen


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


def groq(model: str = "openai/gpt-oss-120b") -> OpenAICompatClient:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    return OpenAICompatClient("https://api.groq.com/openai/v1", model, key, name=f"groq:{model}")


def ollama(model: str = "llama3.1:8b") -> OpenAICompatClient:
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
    return OpenAICompatClient(base, model, None, timeout=180, name=f"ollama:{model}")


def build_llm(provider: str) -> LLM:
    if provider == "groq":
        return groq(os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    if provider == "ollama":
        return ollama(os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    if provider == "auto":
        chain: list[LLM] = []
        if os.environ.get("GROQ_API_KEY"):
            chain.append(groq(os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")))
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
              retries: int = 2, on_call: Callable[[], None] | None = None, **kw: Any) -> Any:
    """Call the model and return validated JSON.

    `validate(obj)` returns None when valid or an error string, which is fed
    back to the model so it can repair its own output (bounded by `retries`).
    """
    convo = list(messages)
    last_err = ""
    for _ in range(retries + 1):
        if on_call:
            on_call()
        raw = llm.complete(convo, **kw)
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
