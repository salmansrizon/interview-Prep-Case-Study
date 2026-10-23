"""CLI: python cli.py "What is (12+8)*3?"   |   python cli.py   (interactive)

The Streamlit UI is app.py; this is the quick terminal version of the same agent.
"""

from __future__ import annotations

import argparse

from config import LLM
from src.agent.loop import run_agent
from src.llm.client import OllamaConnectionError


def ask(question: str, model: str) -> None:
    try:
        res = run_agent(question, model=model)
    except OllamaConnectionError as exc:
        print(f"[!] {exc}")
        return
    for s in res.steps:
        print(f"  [{s.step}] tool> {s.tool}({s.args}) -> {s.result[:80]!r}")
    print(f"{res.answer}\n  ({res.status}, {res.llm_calls} LLM call(s))")


def main() -> None:
    p = argparse.ArgumentParser(description="Math & sandboxed file inspector agent")
    p.add_argument("question", nargs="*")
    p.add_argument("--model", default=LLM.model)
    args = p.parse_args()
    if args.question:
        ask(" ".join(args.question), args.model)
        return
    print(f"Tool agent ({args.model}). Ctrl+C to quit.")
    while True:
        ask(input("you> "), args.model)


if __name__ == "__main__":
    main()
