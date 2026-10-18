"""Ollama ছাড়াই graph test: fake LLM দিয়ে routing, retry limit, approval interrupt।"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command

from src.graph import build_graph
from src.sandbox import extract_code, run_tests

TESTS = "assert add(1, 2) == 3\n"
GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"
START = {"task": "add", "tests": TESTS, "attempts": 0, "history": []}


def fake(responses):
    it = iter(responses)
    prompts = []

    def llm(p):
        prompts.append(p)
        return next(it)

    llm.prompts = prompts
    return llm


def cfg(i):
    return {"configurable": {"thread_id": f"t{i}"}}


def test_sandbox_pass_fail_timeout():
    assert run_tests("def add(a,b): return a+b", TESTS)[0]
    ok, out = run_tests("def add(a,b): return 0", TESTS)
    assert not ok and "AssertionError" in out
    assert not run_tests("while True: pass", "", timeout=1)[0]
    assert extract_code(GOOD).startswith("def add")


def test_fix_loop_feeds_error_back():
    llm = fake([BAD, GOOD])
    out = build_graph(llm, require_approval=False).invoke(START, cfg(1))
    assert out["status"] == "done" and out["attempts"] == 2
    assert "AssertionError" in llm.prompts[1]  # error দ্বিতীয় prompt-এ গেছে
    assert out["history"][-1] == "attempt 2: PASS"


def test_retry_limit_gives_up():
    out = build_graph(fake([BAD] * 5), max_attempts=2, require_approval=False).invoke(START, cfg(2))
    assert out["status"] == "gave_up" and out["attempts"] == 2


def test_approval_interrupt_then_approve_or_reject():
    g = build_graph(fake([GOOD]))
    out = g.invoke(START, cfg(3))
    assert "def add" in out["__interrupt__"][0].value["code"]
    assert g.invoke(Command(resume=True), cfg(3))["status"] == "done"

    g2 = build_graph(fake([GOOD]))
    g2.invoke(START, cfg(4))
    assert g2.invoke(Command(resume=False), cfg(4))["status"] == "rejected"
