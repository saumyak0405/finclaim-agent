"""Run state, evidence store, checkpointing and context compaction.

Design:
- The *state* is the single source of truth for a run. It is plain data
  (dataclasses -> JSON), so it can be checkpointed after every step and a
  crashed run can be resumed exactly where it stopped.
- *Evidence* is stored once, keyed by a stable id (E1, E2, ...). Prompts
  reference evidence by id instead of re-inlining raw tool output, which
  keeps context small on long runs and makes every claim traceable.
- *Working memory* (the step log) is compacted: the most recent entries are
  kept verbatim, older ones collapse to one-line gists.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Evidence:
    id: str
    tool: str
    args: dict[str, Any]
    source: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    tainted: bool = False  # set when the content looks like a prompt injection

    def gist(self, limit: int = 160) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= limit else t[: limit - 3] + "..."


@dataclass
class Step:
    id: int
    goal: str
    tool_hint: str | None = None
    status: str = "pending"  # pending | done | failed | skipped
    note: str = ""
    attempts: int = 0


@dataclass
class LogEntry:
    step_id: int
    kind: str  # action | observation | error | note
    text: str


@dataclass
class RunState:
    run_id: str
    question: str
    phase: str = "plan"  # plan | act | synthesize | verify | done | failed
    plan: list[Step] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    log: list[LogEntry] = field(default_factory=list)
    call_index: dict[str, str] = field(default_factory=dict)  # call signature -> evidence id
    draft: str = ""
    final: str = ""
    verification: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=lambda: {
        "llm_calls": 0, "tool_calls": 0, "tool_errors": 0, "duplicate_calls": 0,
        "replans": 0, "repairs": 0, "act_turns": 0,
    })
    started_at: float = field(default_factory=time.time)

    # ---- construction -------------------------------------------------
    @classmethod
    def new(cls, question: str, run_id: str | None = None) -> "RunState":
        return cls(run_id=run_id or uuid.uuid4().hex[:10], question=question)

    # ---- evidence -----------------------------------------------------
    def add_evidence(self, ev: Evidence) -> Evidence:
        self.evidence[ev.id] = ev
        return ev

    def next_evidence_id(self) -> str:
        return f"E{len(self.evidence) + 1}"

    def usable_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence.values() if not e.tainted]

    # ---- plan ---------------------------------------------------------
    def current_step(self) -> Step | None:
        return next((s for s in self.plan if s.status == "pending"), None)

    # ---- memory / compaction -----------------------------------------
    def working_memory(self, keep_recent: int = 6, max_chars: int = 4000) -> str:
        """Recent log verbatim, older log as one-line gists, hard char cap."""
        older, recent = self.log[:-keep_recent], self.log[-keep_recent:]
        recent_txt = "\n".join(f"step {e.step_id} {e.kind}: {e.text[:600]}" for e in recent)
        if not older:
            return recent_txt[-max_chars:]
        header = f"[compacted {len(older)} earlier entries]"
        budget = max_chars - len(recent_txt) - len(header) - 2
        gists: list[str] = []
        done = [s for s in self.plan if s.status in ("done", "failed", "skipped")]
        for s in reversed(done):  # newest outcomes first; oldest are dropped when over budget
            line = f"  step {s.id} ({s.status}): {s.goal} -> {s.note[:120]}"
            if budget - len(line) - 1 < 0:
                gists.append(f"  ({len(done) - len(gists)} older step summaries omitted)")
                break
            budget -= len(line) + 1
            gists.append(line)
        return "\n".join([header, *reversed(gists), recent_txt])

    def evidence_index(self, max_items: int = 40) -> str:
        items = list(self.evidence.values())[-max_items:]
        rows = []
        for e in items:
            flag = " [TAINTED - do not cite]" if e.tainted else ""
            rows.append(f"[{e.id}] {e.tool}({_fmt_args(e.args)}) from {e.source}{flag}: {e.gist()}")
        return "\n".join(rows) or "(no evidence yet)"

    def evidence_full(self, ids: list[str] | None = None, max_chars: int = 12000) -> str:
        chosen = [self.evidence[i] for i in ids if i in self.evidence] if ids else self.usable_evidence()
        blocks, total = [], 0
        for e in chosen:
            block = f"<evidence id=\"{e.id}\" source=\"{e.source}\">\n{e.text}\n</evidence>"
            if total + len(block) > max_chars:
                blocks.append(f"<!-- {len(chosen) - len(blocks)} more evidence items omitted -->")
                break
            blocks.append(block)
            total += len(block)
        return "\n".join(blocks)

    # ---- serialization -----------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, raw: str) -> "RunState":
        d = json.loads(raw)
        d["plan"] = [Step(**s) for s in d["plan"]]
        d["evidence"] = {k: Evidence(**v) for k, v in d["evidence"].items()}
        d["log"] = [LogEntry(**e) for e in d["log"]]
        return cls(**d)


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))


def call_signature(tool: str, args: dict[str, Any]) -> str:
    return tool + json.dumps(args, sort_keys=True, default=str).lower()


class Checkpointer:
    """Append-only SQLite checkpoints. One row per step; latest row wins."""

    def __init__(self, path: Path | str = ".finclaim/checkpoints.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints ("
                " run_id TEXT, seq INTEGER, phase TEXT, ts REAL, state TEXT,"
                " PRIMARY KEY (run_id, seq))"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, state: RunState) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(MAX(seq), 0) FROM checkpoints WHERE run_id=?", (state.run_id,)).fetchone()
            seq = row[0] + 1
            c.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)",
                      (state.run_id, seq, state.phase, time.time(), state.to_json()))
        return seq

    def load(self, run_id: str) -> RunState | None:
        with self._conn() as c:
            row = c.execute("SELECT state FROM checkpoints WHERE run_id=? ORDER BY seq DESC LIMIT 1",
                            (run_id,)).fetchone()
        return RunState.from_json(row[0]) if row else None

    def runs(self) -> list[tuple[str, str, int]]:
        with self._conn() as c:
            return c.execute(
                "SELECT run_id, phase, MAX(seq) FROM checkpoints c1 WHERE seq = "
                "(SELECT MAX(seq) FROM checkpoints c2 WHERE c2.run_id = c1.run_id) GROUP BY run_id"
            ).fetchall()
