"""A rule-based stand-in for an LLM, used for offline smoke runs and CI.

It is NOT a model and its eval numbers say nothing about real model quality.
It exists so the full loop (plan -> tools -> write -> verify -> repair ->
checkpoint -> trace) can be exercised deterministically without an API key.
By default its writer injects one uncited, made-up number per answer — the
most common real LLM failure — so the verifier path is always exercised.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

TICKER = re.compile(r"\b[A-Z]{2,5}\b")
STOP = {"PE", "EPS", "FY", "USD", "INR", "CEO", "AI", "IPO", "YOY", "QOQ"}


@dataclass
class MockLLM:
    hallucinate: bool = True
    name: str = "mock"
    calls: int = 0
    _log: list[str] = field(default_factory=list)

    def complete(self, messages, **kw) -> str:
        self.calls += 1
        role = messages[0]["content"].split("\n", 1)[0].replace("ROLE:", "").strip()
        user = messages[-1]["content"]
        return getattr(self, f"_{role}")(user, messages)

    # ---------------------------------------------------------------- roles
    def _planner(self, user: str, _m) -> str:
        q = user.split("\n", 1)[0].lower()
        tickers = [t for t in TICKER.findall(user.split("\n", 1)[0]) if t not in STOP]
        if re.search(r"\b(predict|will .* (rise|fall|go)|should i (buy|sell)|target price)\b", q) or not tickers:
            return json.dumps({"answerable": False, "reason": "needs prediction/advice or no ticker", "steps": []})
        steps = []
        for t in tickers[:2]:
            if re.search(r"price|stock|share|perform|return|move", q):
                steps.append({"goal": f"Get 1y price history for {t}", "tool": "price_history"})
            if re.search(r"revenue|income|profit|margin|fundamental|valuation|p/e|earnings", q):
                steps.append({"goal": f"Get fundamentals for {t}", "tool": "fundamentals"})
            if re.search(r"news|why|sentiment|latest|recent", q):
                steps.append({"goal": f"Get recent news for {t}", "tool": "news"})
        return json.dumps({"answerable": True, "reason": "", "steps": steps[:6]})

    def _actor(self, user: str, _m) -> str:
        m = re.search(r"Current step: \d+\. (.*?) \(suggested tool: (\w+)\)", user)
        goal, tool = (m.group(1), m.group(2)) if m else ("", "None")
        tick = (TICKER.findall(goal) or ["?"])[-1]
        ev_block = user.split("Evidence so far:", 1)[1].split("Working memory:", 1)[0]
        mem = user.split("Working memory:", 1)[1]
        if f"{tool}(" in ev_block and tick.lower() in ev_block.lower():
            eid = re.findall(r"\[(E\d+)\] " + tool, ev_block)[-1]
            return json.dumps({"thought": "evidence collected", "action": "step_done", "note": f"see {eid}"})
        if mem.count(f"{tool} failed") >= 2:
            return json.dumps({"thought": "tool keeps failing", "action": "step_failed", "note": f"{tool} unavailable"})
        args = {"ticker": tick}
        if tool == "price_history":
            args["period"] = "1y"
        elif tool == "news":
            args = {"query": tick, "limit": 5}
        elif tool == "None":
            return json.dumps({"thought": "no tool", "action": "step_failed", "note": "no tool for step"})
        return json.dumps({"thought": f"call {tool}", "action": "call_tool", "tool": tool, "args": args})

    def _writer(self, user: str, _m) -> str:
        blocks = re.findall(r'<evidence id="(E\d+)"[^>]*>\n(.*?)\n</evidence>', user, re.S)
        sents = []
        for eid, text in blocks:
            first = re.split(r"\n|; ", text.strip())
            for part in first[:3]:
                part = part.strip().rstrip(".")
                if part:
                    sents.append(f"{part[0].upper() + part[1:]} [{eid}].")
        if not sents:
            return "Insufficient evidence: the tools returned no usable data for this question."
        if self.hallucinate:
            sents.append("Analysts broadly expect revenue to grow 37% next year.")
        return " ".join(sents)

    def _judge(self, user: str, _m) -> str:
        n = len(re.findall(r"^\d+\. ", user.split("Evidence:", 1)[0], re.M))
        return json.dumps({"verdicts": [{"i": i, "label": "SUPPORTED", "reason": "mock"} for i in range(n)]})

    def _repair(self, user: str, _m) -> str:
        draft = user.split("Current answer:\n", 1)[1].split("\n\nRejected claims:", 1)[0]
        rejected = re.findall(r'^- "(.*)" -> ', user, re.M)
        for r in rejected:
            draft = draft.replace(r, "Insufficient evidence: no source supports a forward revenue estimate.")
        return draft
