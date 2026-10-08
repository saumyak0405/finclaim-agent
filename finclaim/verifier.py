"""Claim-level verification.

The draft answer is split into atomic claims (one sentence each). Every claim
goes through deterministic checks first, then an optional LLM judge:

1. citation present      -> otherwise MISSING_CITATION
2. citation resolves     -> otherwise BAD_CITATION (hallucinated evidence id)
3. evidence not tainted  -> otherwise TAINTED_CITATION (prompt-injected source)
4. every number in the claim appears in the cited evidence (unit-normalised,
   0.5% relative tolerance)  -> otherwise UNSUPPORTED_NUMBER
5. LLM entailment judge  -> CONTRADICTED / NOT_ENOUGH_INFO

Deterministic checks run first because they are cheap, exact and cannot be
talked out of a verdict. The judge only sees claims that already passed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from . import prompts
from .llm import LLM, LLMError, call_json
from .state import RunState

SUPPORTED = "supported"
EXEMPT = "exempt"
FAIL_LABELS = {"missing_citation", "bad_citation", "tainted_citation", "unsupported_number",
               "contradicted", "not_enough_info"}

_CITE = re.compile(r"\[(E\d+)\]")
_EXEMPT_PREFIX = re.compile(
    r"^(insufficient evidence|not enough|i could not|i couldn't|no (data|evidence)|note:|disclaimer|"
    r"this is not (investment|financial) advice|removed \d+ claim)", re.I)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9,
          "billion": 1e9, "t": 1e12, "tn": 1e12, "trillion": 1e12, "cr": 1e7, "crore": 1e7,
          "l": 1e5, "lakh": 1e5}
_NUM = re.compile(
    r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|k|thousand|m|mn|million|b|bn|billion|t|tn|trillion|cr|crore|l|lakh)?(?![A-Za-z0-9])",
    re.I)


@dataclass
class ClaimResult:
    i: int
    text: str
    citations: list[str]
    status: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (SUPPORTED, EXEMPT)


def split_claims(answer: str) -> list[str]:
    text = re.sub(r"^\s*[-*•]\s*", "", answer, flags=re.M)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])|\n+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 2]


def parse_numbers(text: str) -> list[float]:
    """Numbers with scale words expanded: '1.52 billion' -> 1.52e9, '12%' -> 12."""
    out = []
    for m in _NUM.finditer(text):
        tok = m.group(0).strip()
        unit = re.sub(r"[-+\d,.\s%]", "", tok).lower()
        num = re.match(r"[-+]?\d[\d,]*(?:\.\d+)?", tok)
        if not num:
            continue
        try:
            val = float(num.group(0).replace(",", ""))
        except ValueError:
            continue
        out.append(val * _SCALE.get(unit, 1.0))
    return out


def _claim_numbers(claim: str) -> list[float]:
    body = _CITE.sub("", claim)
    # ignore bare years and fiscal-year labels: they identify, they don't assert
    body = re.sub(r"\b(?:FY|Q[1-4]\s*)?(19|20)\d{2}\b", "", body)
    body = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", body)
    return parse_numbers(body)


def _matches(value: float, pool: list[float], rel: float = 0.005) -> bool:
    for p in pool:
        for scale in (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100.0, 0.01):  # unit + ratio<->percent slack
            q = p * scale
            if abs(q - value) <= max(abs(q) * rel, 0.006):
                return True
    return False


def deterministic_check(claim: str, i: int, state: RunState) -> ClaimResult:
    cites = _CITE.findall(claim)
    stripped = _CITE.sub("", claim).strip()
    if _EXEMPT_PREFIX.match(stripped):
        return ClaimResult(i, claim, cites, EXEMPT, "abstention/meta sentence")
    if not cites:
        return ClaimResult(i, claim, cites, "missing_citation", "factual sentence without citation")
    missing = [c for c in cites if c not in state.evidence]
    if missing:
        return ClaimResult(i, claim, cites, "bad_citation", f"cites non-existent evidence {missing}")
    tainted = [c for c in cites if state.evidence[c].tainted]
    if tainted:
        return ClaimResult(i, claim, cites, "tainted_citation", f"cites injection-flagged evidence {tainted}")
    pool: list[float] = []
    for c in cites:
        pool += parse_numbers(state.evidence[c].text)
    bad = [n for n in _claim_numbers(claim) if not _matches(n, pool)]
    if bad:
        return ClaimResult(i, claim, cites, "unsupported_number",
                           f"numbers {bad} not found in cited evidence {cites}")
    return ClaimResult(i, claim, cites, SUPPORTED, "deterministic checks passed")


def judge_claims(results: list[ClaimResult], state: RunState, llm: LLM, on_call=None) -> None:
    pending = [r for r in results if r.status == SUPPORTED]
    if not pending:
        return
    ids = sorted({c for r in pending for c in r.citations})
    claims = "\n".join(f"{n}. {r.text}" for n, r in enumerate(pending))

    def valid(obj: Any) -> str | None:
        v = obj.get("verdicts") if isinstance(obj, dict) else None
        if not isinstance(v, list):
            return "missing 'verdicts' list"
        for item in v:
            if not isinstance(item, dict) or item.get("label") not in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"):
                return "each verdict needs i and a valid label"
        return None

    msgs = [{"role": "system", "content": prompts.JUDGE},
            {"role": "user", "content": prompts.JUDGE_USER.format(claims=claims, evidence=state.evidence_full(ids))}]
    try:
        out = call_json(llm, msgs, valid, on_call=on_call)
    except LLMError:
        return  # judge unavailable: keep deterministic verdicts, never block the run
    for v in out["verdicts"]:
        idx = v.get("i")
        if isinstance(idx, int) and 0 <= idx < len(pending) and v["label"] != "SUPPORTED":
            pending[idx].status = v["label"].lower()
            pending[idx].reason = "judge: " + str(v.get("reason", ""))[:200]


def verify(answer: str, state: RunState, llm: LLM | None = None, use_judge: bool = True,
           on_call=None) -> list[ClaimResult]:
    results = [deterministic_check(c, i, state) for i, c in enumerate(split_claims(answer))]
    if use_judge and llm is not None:
        judge_claims(results, state, llm, on_call=on_call)
    return results


def assemble_final(results: list[ClaimResult], state: RunState) -> str:
    kept = [r.text for r in results if r.ok]
    dropped = sum(1 for r in results if not r.ok)
    body = " ".join(kept) if kept else "Insufficient evidence: no claim could be verified against the gathered sources."
    if dropped:
        body += f"\n\nNote: removed {dropped} claim(s) that could not be verified against the sources."
    cited = sorted({c for r in results if r.ok for c in r.citations}, key=lambda x: int(x[1:]))
    if cited:
        body += "\n\nSources:\n" + "\n".join(
            f"[{c}] {state.evidence[c].source} via {state.evidence[c].tool}" for c in cited)
    return body


def as_dicts(results: list[ClaimResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
