"""The agent loop: plan -> act (tool loop) -> synthesize -> verify -> repair.

The loop is an explicit state machine over `RunState.phase`. Each phase
handler does one unit of work and returns; the state is checkpointed after
every unit, so a crash at any point resumes from the last completed unit
without re-running tool calls that already produced evidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts, verifier
from .llm import LLM, LLMError, call_json
from .state import Checkpointer, Evidence, LogEntry, RunState, Step, call_signature
from .tools import ToolRegistry
from .trace import Tracer


@dataclass
class AgentConfig:
    max_plan_steps: int = 6
    max_act_turns: int = 14        # global tool-loop budget (prevents runaway loops)
    max_attempts_per_step: int = 3  # tool errors before a step is marked failed
    max_duplicates_per_step: int = 2
    max_replans: int = 1
    max_repairs: int = 2
    verify: bool = True             # ablation switch for evals
    use_judge: bool = True
    wall_clock_s: float = 300.0


@dataclass
class Agent:
    llm: LLM
    tools: ToolRegistry
    config: AgentConfig = field(default_factory=AgentConfig)
    checkpointer: Checkpointer | None = None
    trace_dir: Path | None = None

    # ------------------------------------------------------------ public
    def run(self, question: str, run_id: str | None = None) -> RunState:
        state = RunState.new(question, run_id)
        return self._drive(state, self._tracer(state.run_id))

    def resume(self, run_id: str) -> RunState:
        if self.checkpointer is None:
            raise RuntimeError("resume needs a checkpointer")
        state = self.checkpointer.load(run_id)
        if state is None:
            raise KeyError(f"no checkpoint for run {run_id}")
        tracer = self._tracer(run_id)
        tracer.emit("resume", run_id=run_id, phase=state.phase, evidence=len(state.evidence))
        return self._drive(state, tracer)

    # ------------------------------------------------------------ driver
    def _tracer(self, run_id: str) -> Tracer:
        return Tracer(self.trace_dir / f"{run_id}.jsonl" if self.trace_dir else None)

    def _drive(self, state: RunState, tracer: Tracer) -> RunState:
        self.tracer = tracer
        handlers = {"plan": self._plan, "act": self._act, "synthesize": self._synthesize, "verify": self._verify}
        deadline = time.time() + self.config.wall_clock_s
        tracer.emit("run_start", run_id=state.run_id, question=state.question, phase=state.phase)
        while state.phase in handlers:
            if time.time() > deadline and state.phase in ("plan", "act"):
                self._skip_remaining(state, "wall-clock budget exhausted")
                state.phase = "synthesize"
            try:
                handlers[state.phase](state)
            except LLMError as e:
                tracer.emit("error", phase=state.phase, error=str(e))
                state.final = f"Run failed in phase '{state.phase}': {e}"
                state.phase = "failed"
            self._checkpoint(state)
        tracer.emit("run_end", run_id=state.run_id, phase=state.phase, counters=state.counters,
                    elapsed_s=round(time.time() - state.started_at, 2))
        state.trace = tracer.events  # type: ignore[attr-defined]
        return state

    def _checkpoint(self, state: RunState) -> None:
        if self.checkpointer is not None:
            seq = self.checkpointer.save(state)
            self.tracer.emit("checkpoint", seq=seq, phase=state.phase)

    def _llm_call(self, state: RunState) -> None:
        state.counters["llm_calls"] += 1

    # ------------------------------------------------------------ phases
    def _plan(self, state: RunState) -> None:
        def valid(obj: Any) -> str | None:
            if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
                return "need an object with a 'steps' list"
            if len(obj["steps"]) > self.config.max_plan_steps:
                return f"at most {self.config.max_plan_steps} steps"
            for s in obj["steps"]:
                if not isinstance(s, dict) or not s.get("goal"):
                    return "each step needs a 'goal'"
                if s.get("tool") not in (None, "null", *self.tools.specs):
                    return f"unknown tool {s.get('tool')!r}"
            return None

        user = f"Question: {state.question}"
        if state.plan:  # replanning
            problems = "\n".join(f"step {s.id}: {s.note}" for s in state.plan if s.status in ("failed",)) or \
                       "\n".join(e.text for e in state.log[-3:])
            done = "\n".join(f"step {s.id}: {s.goal} -> {s.note}" for s in state.plan if s.status == "done") or "(none)"
            user += "\n\n" + prompts.REPLANNER_NOTE.format(problems=problems, done=done) + \
                    "\n\nEvidence:\n" + state.evidence_index()
        out = call_json(self.llm, [{"role": "system", "content": prompts.PLANNER.format(tools=self.tools.catalog())},
                                   {"role": "user", "content": user}], valid, on_call=lambda: self._llm_call(state))
        kept = [s for s in state.plan if s.status != "pending"]
        start = max((s.id for s in kept), default=0)
        new = [Step(id=start + n + 1, goal=s["goal"], tool_hint=None if s.get("tool") in (None, "null") else s["tool"])
               for n, s in enumerate(out["steps"])]
        state.plan = kept + new
        self.tracer.emit("plan", answerable=out.get("answerable", True), reason=out.get("reason", ""),
                         steps=[(s.id, s.goal, s.tool_hint) for s in new], replan=bool(kept))
        state.log.append(LogEntry(0, "note", f"plan: {[s.goal for s in new]}"))
        state.phase = "act" if new else "synthesize"

    def _act(self, state: RunState) -> None:
        step = state.current_step()
        if step is None:
            state.phase = "synthesize"
            return
        if state.counters["act_turns"] >= self.config.max_act_turns:
            self._skip_remaining(state, "tool-loop budget exhausted")
            state.phase = "synthesize"
            return
        state.counters["act_turns"] += 1

        def valid(obj: Any) -> str | None:
            if not isinstance(obj, dict):
                return "need a JSON object"
            act = obj.get("action")
            if act == "call_tool":
                return self.tools.validate(obj.get("tool", ""), obj.get("args", {}))
            if act in ("step_done", "step_failed", "replan"):
                return None
            return "action must be call_tool | step_done | step_failed | replan"

        plan_txt = "\n".join(f"{s.id}. [{s.status}] {s.goal}" for s in state.plan)
        msgs = [{"role": "system", "content": prompts.ACTOR.format(tools=self.tools.catalog())},
                {"role": "user", "content": prompts.ACTOR_USER.format(
                    question=state.question, plan=plan_txt, step_id=step.id, goal=step.goal,
                    tool_hint=step.tool_hint, evidence=state.evidence_index(), memory=state.working_memory())}]
        out = call_json(self.llm, msgs, valid, on_call=lambda: self._llm_call(state))
        act = out["action"]
        self.tracer.emit("decision", step=step.id, action=act, thought=str(out.get("thought", ""))[:300],
                         tool=out.get("tool"), args=out.get("args"), note=out.get("note"))

        if act == "call_tool":
            self._run_tool(state, step, out["tool"], out.get("args", {}))
        elif act == "step_done":
            step.status, step.note = "done", str(out.get("note", ""))[:400]
            state.log.append(LogEntry(step.id, "note", f"done: {step.note}"))
        elif act == "step_failed":
            step.status, step.note = "failed", str(out.get("note", ""))[:400]
            state.log.append(LogEntry(step.id, "note", f"failed: {step.note}"))
        elif act == "replan":
            if state.counters["replans"] < self.config.max_replans:
                state.counters["replans"] += 1
                step.status, step.note = "failed", "replan: " + str(out.get("note", ""))[:300]
                state.phase = "plan"
            else:
                step.status, step.note = "skipped", "replan budget exhausted"
            state.log.append(LogEntry(step.id, "note", step.note))

    def _run_tool(self, state: RunState, step: Step, tool: str, args: dict[str, Any]) -> None:
        sig = call_signature(tool, args)
        if sig in state.call_index:  # loop guard: never pay for the same call twice
            state.counters["duplicate_calls"] += 1
            eid = state.call_index[sig]
            dups = sum(1 for e in state.log if e.step_id == step.id and e.text.startswith("duplicate"))
            state.log.append(LogEntry(step.id, "observation", f"duplicate call; result already in {eid}"))
            self.tracer.emit("duplicate_call", step=step.id, tool=tool, evidence=eid)
            if dups + 1 >= self.config.max_duplicates_per_step:
                step.status, step.note = "done", f"forced complete after repeated calls; see {eid}"
            return

        state.counters["tool_calls"] += 1
        t0 = time.time()
        res = self.tools.execute(tool, args)
        latency = round(time.time() - t0, 3)
        if not res.ok:
            state.counters["tool_errors"] += 1
            step.attempts += 1
            state.log.append(LogEntry(step.id, "error", f"{tool} failed: {res.error}"))
            self.tracer.emit("tool_error", step=step.id, tool=tool, args=args, error=res.error, latency_s=latency)
            if step.attempts >= self.config.max_attempts_per_step:
                step.status, step.note = "failed", f"{tool} failed {step.attempts}x: {res.error}"
            return

        ev = Evidence(id=state.next_evidence_id(), tool=tool, args=args, source=res.result.get("source", tool),
                      text=res.result["text"], data=res.result.get("data", {}), tainted=res.injected)
        state.add_evidence(ev)
        state.call_index[sig] = ev.id
        obs = (f"{ev.id} FLAGGED as possible prompt injection; treated as untrusted and uncitable"
               if ev.tainted else f"{ev.id}: {ev.gist()}")
        state.log.append(LogEntry(step.id, "observation", obs))
        self.tracer.emit("tool_result", step=step.id, tool=tool, args=args, evidence=ev.id,
                         tainted=ev.tainted, latency_s=latency, chars=len(ev.text))

    def _synthesize(self, state: RunState) -> None:
        notes = "\n".join(f"step {s.id} [{s.status}] {s.goal}: {s.note}" for s in state.plan) or "(no plan)"
        msgs = [{"role": "system", "content": prompts.WRITER},
                {"role": "user", "content": prompts.WRITER_USER.format(
                    question=state.question, evidence=state.evidence_full() or "(none)", notes=notes)}]
        self._llm_call(state)
        state.draft = self.llm.complete(msgs).strip()
        self.tracer.emit("draft", text=state.draft)
        state.phase = "verify"

    def _verify(self, state: RunState) -> None:
        if not self.config.verify:  # ablation: ship the draft as-is
            state.verification = verifier.as_dicts(verifier.verify(state.draft, state, None, use_judge=False))
            state.final = state.draft
            state.phase = "done"
            return
        results = verifier.verify(state.draft, state, self.llm, self.config.use_judge,
                                  on_call=lambda: self._llm_call(state))
        state.verification = verifier.as_dicts(results)
        failed = [r for r in results if not r.ok]
        self.tracer.emit("verify", round=state.counters["repairs"], total=len(results),
                         failed=[(r.i, r.status, r.reason) for r in failed])
        if failed and state.counters["repairs"] < self.config.max_repairs:
            state.counters["repairs"] += 1
            rejected = "\n".join(f"- \"{r.text}\" -> {r.status}: {r.reason}" for r in failed)
            msgs = [{"role": "system", "content": prompts.REPAIR},
                    {"role": "user", "content": prompts.REPAIR_USER.format(
                        question=state.question, draft=state.draft, rejected=rejected,
                        evidence=state.evidence_full() or "(none)")}]
            self._llm_call(state)
            state.draft = self.llm.complete(msgs).strip()
            self.tracer.emit("repair", round=state.counters["repairs"], text=state.draft)
            return  # stay in verify phase; next iteration re-checks the repaired draft
        state.final = verifier.assemble_final(results, state)
        state.phase = "done"
        self.tracer.emit("final", text=state.final)

    # ------------------------------------------------------------ helpers
    def _skip_remaining(self, state: RunState, why: str) -> None:
        for s in state.plan:
            if s.status == "pending":
                s.status, s.note = "skipped", why
        self.tracer.emit("budget_stop", reason=why)
