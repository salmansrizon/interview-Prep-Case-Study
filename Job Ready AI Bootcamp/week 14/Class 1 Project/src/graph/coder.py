"""
Self-Correcting LLM Coder as a LangGraph StateGraph.

    START → generate → static_check ──bad──┐
                          │ ok              │
                          ▼                 │
                     run_visible ──fail──► (attempts < max?) ──yes──► generate
                          │ pass                 │ no
                          ▼                      ▼
                     run_hidden ──fail──► (attempts < max?) ──yes──► generate (generic hint only)
                          │ pass                 │ no
                          ▼                      ▼
                     approval (interrupt)   hidden_fail / give_up (best attempt kept)
                          │
                          ▼
                     done | rejected

HIGHLIGHTS:
* State = graph-এর shared memory। ``history``-তে reducer (``operator.add``)
  — node ``{"history": [...]}`` দিলে list-এ জোড়া হয়; বাকি key overwrite।
* Loop-এর থামার শর্ত state-এ (``attempts``), আর ``recursion_limit`` config-এ
  explicitly — LangGraph 1.x-এর default আর ২৫ না, ~১০ হাজার।
* Hidden test কখনো model-কে দেখানো হয় না। Visible pass কিন্তু hidden fail
  হলে model শুধু একটা generic hint পায় ("আরও case-এ ভুল, spec আবার পড়ো") —
  কোন test, কী input, কিছুই না। এটা দুই রকম ভুল ধরে: example মুখস্থ করা
  (reward hacking) আর spec-এর অর্ধেক বোঝা। Budget শেষ হলে ``hidden_fail``।
* ``interrupt()``-এর node resume-এ **শুরু থেকে আবার চলে** — তাই approval
  node-এ interrupt-এর আগে কোনো side effect নেই।
* ``llm`` parameter হিসেবে আসে (closure) — test-এ fake LLM।
"""
from __future__ import annotations

import operator
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from config import get_config
from src.sandbox.runner import run_tests
from src.sandbox.static_check import extract_code, static_check


class CoderState(TypedDict, total=False):
    # input
    task_id: str
    func_name: str
    prompt: str
    visible: list
    hidden: list
    # working memory
    code: str
    attempts: int
    feedback: str          # what the model sees on the next attempt
    stage_ok: bool         # did the last check pass?
    visible_passed: int
    visible_total: int
    hidden_passed: int
    hidden_total: int
    best_code: str         # attempt with the most visible tests passed
    best_passed: int
    first_attempt_passed: bool
    approved: bool
    status: str            # done | rejected | gave_up | hidden_fail
    history: Annotated[list[str], operator.add]


def build_prompt(state: CoderState) -> str:
    p = f"Task: {state['prompt']}\nThe function must be named `{state['func_name']}`.\n"
    examples = "\n".join(f"  {c}  ->  {e}" for c, e in state["visible"])
    p += f"Examples that must work:\n{examples}\n"
    if state.get("feedback"):
        p += (f"\nYour previous attempt:\n```python\n{state['code']}\n```\n"
              f"It FAILED:\n{state['feedback']}\n"
              "Explain the bug to yourself in one sentence, then write the corrected function. "
              "Do not special-case the example inputs; solve the general task.\n")
    return p


def build_graph(llm: Callable[[str], str], *, max_attempts: int | None = None,
                require_approval: bool | None = None, checkpointer=None):
    cfg = get_config()
    max_attempts = cfg.agent.max_attempts if max_attempts is None else max_attempts
    require_approval = cfg.agent.require_approval if require_approval is None else require_approval
    sb = cfg.sandbox

    def generate(state: CoderState) -> dict:
        n = state.get("attempts", 0) + 1
        code = extract_code(llm(build_prompt(state)))
        return {"code": code, "attempts": n, "history": [f"attempt {n}: code generated ({len(code.splitlines())} lines)"]}

    def check(state: CoderState) -> dict:
        problems = static_check(state["code"], state["func_name"], sb.banned_imports, sb.banned_calls)
        if problems:
            return {"stage_ok": False, "feedback": "Static check failed:\n- " + "\n- ".join(problems),
                    "history": [f"attempt {state['attempts']}: static check FAILED ({problems[0]})"]}
        return {"stage_ok": True}

    def run_visible(state: CoderState) -> dict:
        rep = run_tests(state["code"], state["visible"], sb.timeout_seconds, sb.cpu_seconds, sb.memory_mb)
        update = {"stage_ok": rep.all_passed, "visible_passed": rep.passed, "visible_total": len(state["visible"]),
                  "feedback": "" if rep.all_passed else rep.feedback(sb.max_output_chars),
                  "history": [f"attempt {state['attempts']}: visible tests {rep.passed}/{len(state['visible'])}"
                              + ("" if not rep.crashed else f" ({rep.crashed[:60]})")]}
        if state["attempts"] == 1 and not rep.all_passed:
            update["first_attempt_passed"] = False
        if rep.passed >= state.get("best_passed", -1):
            update.update(best_code=state["code"], best_passed=rep.passed)
        return update

    def run_hidden(state: CoderState) -> dict:
        if not state.get("hidden"):
            return {"stage_ok": True, "hidden_passed": 0, "hidden_total": 0,
                    **({"first_attempt_passed": True} if state["attempts"] == 1 else {})}
        rep = run_tests(state["code"], state["hidden"], sb.timeout_seconds, sb.cpu_seconds, sb.memory_mb)
        update = {"stage_ok": rep.all_passed, "hidden_passed": rep.passed, "hidden_total": len(state["hidden"]),
                  "history": [f"attempt {state['attempts']}: hidden tests {rep.passed}/{len(state['hidden'])}"
                              + ("" if rep.all_passed else " — passes the examples but not the general task")]}
        if not rep.all_passed:
            # Generic on purpose: revealing hidden inputs/outputs would let the model memorise them.
            update["feedback"] = (f"Your code passes the {len(state['visible'])} examples but fails "
                                  f"{len(state['hidden']) - rep.passed} other (hidden) test case(s). Re-read the "
                                  "specification carefully, think about edge cases it mentions, and fix the "
                                  "general logic. Do not special-case the examples.")
        if state["attempts"] == 1:
            update["first_attempt_passed"] = rep.all_passed
        return update

    def approval(state: CoderState) -> dict:
        # Nothing with side effects BEFORE interrupt(): on resume this node runs again from the top.
        decision = interrupt({"question": "Approve this code?", "code": state["code"],
                              "visible": f"{state['visible_passed']}/{state['visible_total']}",
                              "hidden": f"{state.get('hidden_passed', 0)}/{state.get('hidden_total', 0)}"})
        return {"approved": bool(decision), "history": [f"human approved={bool(decision)}"]}

    def give_up(state: CoderState) -> dict:
        return {"status": "gave_up", "code": state.get("best_code", state.get("code", "")),
                "history": [f"gave up after {state['attempts']} attempts; kept best ({state.get('best_passed', 0)} tests)"]}

    # ── routing ───────────────────────────────────────────────────────
    def retry_or_give_up(state: CoderState) -> str:
        return "generate" if state["attempts"] < max_attempts else "give_up"

    def after_check(state: CoderState) -> str:
        return "run_visible" if state["stage_ok"] else retry_or_give_up(state)

    def after_visible(state: CoderState) -> str:
        return "run_hidden" if state["stage_ok"] else retry_or_give_up(state)

    def after_hidden(state: CoderState) -> str:
        if not state["stage_ok"]:
            return "generate" if state["attempts"] < max_attempts else "hidden_fail"
        return "approval" if require_approval else "done"

    g = StateGraph(CoderState)
    g.add_node("generate", generate)
    g.add_node("static_check", check)
    g.add_node("run_visible", run_visible)
    g.add_node("run_hidden", run_hidden)
    g.add_node("approval", approval)
    g.add_node("give_up", give_up)
    g.add_node("hidden_fail", lambda s: {"status": "hidden_fail"})
    g.add_node("done", lambda s: {"status": "done"})
    g.add_node("rejected", lambda s: {"status": "rejected"})
    g.add_edge(START, "generate")
    g.add_edge("generate", "static_check")
    g.add_conditional_edges("static_check", after_check, ["run_visible", "generate", "give_up"])
    g.add_conditional_edges("run_visible", after_visible, ["run_hidden", "generate", "give_up"])
    g.add_conditional_edges("run_hidden", after_hidden, ["approval", "done", "generate", "hidden_fail"])
    g.add_conditional_edges("approval", lambda s: "done" if s["approved"] else "rejected", ["done", "rejected"])
    for n in ("give_up", "hidden_fail", "done", "rejected"):
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def initial_state(task) -> CoderState:
    """CodingTask -> starting state."""
    return {"task_id": task.id, "func_name": task.func_name, "prompt": task.prompt,
            "visible": [list(t) for t in task.visible], "hidden": [list(t) for t in task.hidden],
            "attempts": 0, "best_passed": -1, "history": []}


def run_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": get_config().agent.recursion_limit}


def sqlite_checkpointer(path=None):
    """Durable checkpointer: runs survive restarts; enables time travel (get_state_history)."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    db = path or get_config().data.checkpoint_db
    db.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(db), check_same_thread=False))
