"""Structured JSONL tracing.

Every decision the agent makes is emitted as one event. The trace is the
debugging surface, the audit record, and the input to the eval harness.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Tracer:
    path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        event = {"ts": round(time.time(), 3), "kind": kind, **payload}
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        return event

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e["kind"] == kind)

    @staticmethod
    def load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
