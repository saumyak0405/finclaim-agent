"""The OpenAI-compatible client against a local fake server reproducing real Groq behaviours."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from finclaim.llm import LLMError, OpenAICompatClient


class FakeGroq(BaseHTTPRequestHandler):
    replies: list = []
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGroq.seen.append({"body": body, "ua": self.headers.get("User-Agent", "")})
        status, payload, *extra = FakeGroq.replies.pop(0)
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        for k, v in (extra[0] if extra else {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    FakeGroq.replies, FakeGroq.seen = [], []
    srv = HTTPServer(("127.0.0.1", 0), FakeGroq)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def client(url):
    return OpenAICompatClient(url, "openai/gpt-oss-120b", "gsk_test", timeout=5, max_retries=2)


def test_native_tool_call_becomes_json_action(server):
    FakeGroq.replies = [(200, {"choices": [{"message": {"content": "look first", "tool_calls": [
        {"type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/a.py\"}"}}]}}]})]
    out = json.loads(client(server).complete([{"role": "user", "content": "x"}],
                                             tools=[{"type": "function", "function": {"name": "read_file"}}]))
    assert out["tool"] == "read_file" and out["args"] == {"path": "src/a.py"} and out["thought"] == "look first"
    sent = FakeGroq.seen[0]
    assert sent["body"]["tool_choice"] == "required" and sent["body"]["tools"]
    assert "urllib" not in sent["ua"].lower()  # Groq's Cloudflare blocks Python-urllib (error 1010)


def test_tool_use_failed_400_is_recovered_not_fatal(server):
    err = {"error": {"message": "Tool choice is none, but model called a tool", "code": "tool_use_failed",
                     "failed_generation": "{\"name\": \"functions.search\", \"arguments\": {\"pattern\": \"def f\"}}"}}
    FakeGroq.replies = [(400, err)]
    out = json.loads(client(server).complete([{"role": "user", "content": "x"}]))
    assert out["tool"] == "search" and out["args"] == {"pattern": "def f"}


def test_non_retryable_errors_fail_fast_with_detail(server):
    FakeGroq.replies = [(403, b"error code: 1010\n")]
    with pytest.raises(LLMError, match="1010"):
        client(server).complete([{"role": "user", "content": "x"}])
    assert len(FakeGroq.seen) == 1


def test_rate_limit_is_retried(server, monkeypatch):
    monkeypatch.setattr("finclaim.llm.time.sleep", lambda s: None)
    FakeGroq.replies = [(429, {"error": {"message": "slow down"}}),
                        (200, {"choices": [{"message": {"content": "{\"tool\": \"submit\"}"}}]})]
    assert "submit" in client(server).complete([{"role": "user", "content": "x"}])
    assert len(FakeGroq.seen) == 2


def test_rate_limit_waits_as_long_as_the_server_says(server, monkeypatch):
    waits = []
    monkeypatch.setattr("finclaim.llm.time.sleep", waits.append)
    FakeGroq.replies = [(429, {"error": {"message": "tokens per minute"}}, {"x-ratelimit-reset-tokens": "7.5s"}),
                        (429, {"error": {"message": "rate"}}, {"retry-after": "2"}),
                        (200, {"choices": [{"message": {"content": "ok"}}]})]
    c = client(server)
    c.max_retries = 4
    assert c.complete([{"role": "user", "content": "x"}]) == "ok"
    assert 7.5 <= waits[0] <= 8.0 and 2.0 <= waits[1] <= 2.5


def test_retry_after_parsing():
    from finclaim.llm import retry_after_seconds
    assert retry_after_seconds({"retry-after": "3"}) == 3.0
    assert retry_after_seconds({"x-ratelimit-reset-tokens": "1m2.5s"}) == 62.5
    assert retry_after_seconds({"x-ratelimit-reset-requests": "450ms"}) == 0.45
    assert retry_after_seconds({}) is None
