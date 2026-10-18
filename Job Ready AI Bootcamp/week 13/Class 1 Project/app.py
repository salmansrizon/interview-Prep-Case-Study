"""CLI: python app.py "What is (12+8)*3?"   |   python app.py  (interactive)"""

from __future__ import annotations

import sys

from config import LLM
from src.agent.loop import run_agent
from src.llm.client import OllamaConnectionError


def ask(q: str) -> None:
    try:
        answer, trace = run_agent(q)
    except OllamaConnectionError as exc:
        print(f"[!] {exc}")
        return
    for t in trace:
        print(f"  tool> {t['tool']}({t['args']}) -> {t['result'][:80]!r}")
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print(f"Tool agent ({LLM.model}). Ctrl+C to quit.")
        while True:
            ask(input("you> "))
