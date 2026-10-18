"""CLI: python app.py "task" --tests asserts.py [--no-approval]

HIGHLIGHTS: graph চালানো = `invoke`; interrupt হলে result-এ `__interrupt__`
key থাকে, তখন মানুষের উত্তর `Command(resume=...)` দিয়ে একই thread_id-তে পাঠাতে হয়।
"""

from __future__ import annotations

import argparse
import sys
import uuid

from langgraph.types import Command

from config import get_config
from src.graph import build_graph
from src.llm import OllamaUnavailable, ollama_llm

DEMO_TASK = "Write is_palindrome(s) ignoring case and non-alphanumeric characters."
DEMO_TESTS = (
    "assert is_palindrome('A man, a plan, a canal: Panama')\n"
    "assert not is_palindrome('hello')\n"
    "assert is_palindrome('')\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default=DEMO_TASK)
    ap.add_argument("--tests", default=None, help="assert-based test file")
    ap.add_argument("--no-approval", action="store_true")
    a = ap.parse_args()
    cfg = get_config()
    tests = open(a.tests, encoding="utf-8").read() if a.tests else DEMO_TESTS
    graph = build_graph(
        ollama_llm,
        max_attempts=cfg.max_attempts,
        require_approval=cfg.require_approval and not a.no_approval,
        timeout=cfg.test_timeout_sec,
    )
    run = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        out = graph.invoke({"task": a.task, "tests": tests, "attempts": 0, "history": []}, run)
        while "__interrupt__" in out:
            print("\n--- Proposed code ---\n" + out["__interrupt__"][0].value["code"])
            ok = input("Approve? [y/N] ").strip().lower() == "y"
            out = graph.invoke(Command(resume=ok), run)
    except OllamaUnavailable as e:
        print(e)
        return 1
    print("\n".join(out["history"]))
    print(f"status={out['status']}\n{out['code']}")
    return 0 if out["status"] == "done" else 2


if __name__ == "__main__":
    sys.exit(main())
