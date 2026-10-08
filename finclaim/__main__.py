"""CLI.

  python -m finclaim ask "How has INFY stock performed over the last year?" --provider auto
  python -m finclaim ask "..." --fixtures evals/fixtures/nova.json --provider mock
  python -m finclaim resume <run_id>
  python -m finclaim runs
  python -m finclaim trace <run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent, AgentConfig
from .llm import build_llm
from .state import Checkpointer
from .tools import FixtureBackend, LiveBackend, ToolRegistry
from .trace import Tracer

HOME = Path(".finclaim")


def make_agent(provider: str, fixtures: str | None, no_verify: bool = False) -> Agent:
    backend = FixtureBackend.load(fixtures) if fixtures else LiveBackend()
    return Agent(llm=build_llm(provider), tools=ToolRegistry(backend),
                 config=AgentConfig(verify=not no_verify),
                 checkpointer=Checkpointer(HOME / "checkpoints.sqlite"), trace_dir=HOME / "traces")


def print_result(state) -> None:
    print(state.final or state.draft)
    bad = [v for v in state.verification if v["status"] not in ("supported", "exempt")]
    print(f"\n-- run {state.run_id} | phase={state.phase} | {state.counters} | "
          f"claims={len(state.verification)} rejected_in_last_check={len(bad)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="finclaim")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--provider", default="auto", choices=["auto", "groq", "ollama", "mock"])
    a.add_argument("--fixtures")
    a.add_argument("--no-verify", action="store_true")
    r = sub.add_parser("resume")
    r.add_argument("run_id")
    r.add_argument("--provider", default="auto", choices=["auto", "groq", "ollama", "mock"])
    r.add_argument("--fixtures")
    sub.add_parser("runs")
    t = sub.add_parser("trace")
    t.add_argument("run_id")
    args = p.parse_args(argv)

    if args.cmd == "ask":
        print_result(make_agent(args.provider, args.fixtures, args.no_verify).run(args.question))
    elif args.cmd == "resume":
        print_result(make_agent(args.provider, args.fixtures).resume(args.run_id))
    elif args.cmd == "runs":
        for run_id, phase, seq in Checkpointer(HOME / "checkpoints.sqlite").runs():
            print(f"{run_id}  {phase:<10} checkpoints={seq}")
    elif args.cmd == "trace":
        for ev in Tracer.load(HOME / "traces" / f"{args.run_id}.jsonl"):
            print(json.dumps(ev, default=str)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
