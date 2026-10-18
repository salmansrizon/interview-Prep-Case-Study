"""Self-Correcting Writer: LangGraph StateGraph।

    START -> write_code -> run_tests --pass--> approval? -> done | rejected
                 ^             |
                 +---fail------+  (attempts < max_attempts, নইলে give_up)
"""

from __future__ import annotations

import operator
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.sandbox import extract_code, run_tests


class WriterState(TypedDict, total=False):
    """HIGHLIGHTS: State = graph-এর shared memory। `history`-তে reducer
    (`operator.add`) আছে, তাই node `{"history": ["x"]}` দিলে list-এ জোড়া হয়;
    reducer ছাড়া বাকি key (code, error ...) প্রতিবার overwrite হয়।"""

    task: str
    tests: str
    code: str
    error: str
    attempts: int
    passed: bool
    approved: bool
    status: str  # done | rejected | gave_up
    history: Annotated[list[str], operator.add]


def build_graph(
    llm: Callable[[str], str],
    *,
    max_attempts: int = 3,
    require_approval: bool = True,
    timeout: int = 10,
    checkpointer=None,
):
    """HIGHLIGHTS: `llm` parameter হিসেবে আসে (closure), global না, তাই test-এ
    fake LLM দিয়ে Ollama ছাড়াই পুরো graph চালানো যায়। `interrupt()` কাজ
    করতে checkpointer লাগে; না দিলে InMemorySaver।"""

    def write_code(state: WriterState) -> dict:
        prompt = (
            "Write a Python function for this task. Return only one ```python block.\n"
            f"Task: {state['task']}\n"
        )
        if state.get("error"):
            prompt += (
                f"\nYour previous code:\n```python\n{state['code']}\n```\n"
                f"failed with:\n{state['error']}\nFix it.\n"
            )
        n = state.get("attempts", 0) + 1
        return {
            "code": extract_code(llm(prompt)),
            "attempts": n,
            "history": [f"attempt {n}: code written"],
        }

    def test_code(state: WriterState) -> dict:
        ok, out = run_tests(state["code"], state["tests"], timeout)
        return {
            "passed": ok,
            "error": "" if ok else out,
            "history": [f"attempt {state['attempts']}: {'PASS' if ok else 'FAIL'}"],
        }

    def route_after_tests(state: WriterState) -> str:
        if state["passed"]:
            return "approval" if require_approval else "done"
        return "write_code" if state["attempts"] < max_attempts else "give_up"

    def approval(state: WriterState) -> dict:
        # graph এখানে থেমে যায়; resume value-ই decision (True/False)
        decision = interrupt({"question": "Approve this code?", "code": state["code"]})
        return {"approved": bool(decision), "history": [f"human approved={bool(decision)}"]}

    def route_after_approval(state: WriterState) -> str:
        return "done" if state["approved"] else "rejected"

    g = StateGraph(WriterState)
    g.add_node("write_code", write_code)
    g.add_node("run_tests", test_code)
    g.add_node("approval", approval)
    g.add_node("done", lambda s: {"status": "done"})
    g.add_node("rejected", lambda s: {"status": "rejected"})
    g.add_node("give_up", lambda s: {"status": "gave_up"})
    g.add_edge(START, "write_code")
    g.add_edge("write_code", "run_tests")
    g.add_conditional_edges("run_tests", route_after_tests, ["write_code", "approval", "done", "give_up"])
    g.add_conditional_edges("approval", route_after_approval, ["done", "rejected"])
    for n in ("done", "rejected", "give_up"):
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
