"""CLI Travel Agent.

  python app.py "Book the cheapest DAC to CXB flight on 2026-11-05 for Rahim"
  python app.py --compare "..."     # same task on llama3.1:8b, mistral:7b, llama3.2:3b
"""
from __future__ import annotations

import argparse
import functools

from config import get_config
from src.llm_client import ollama_chat
from src.react_agent import run_agent
from src.tools import TravelTools


def run(task: str, model: str):
    cfg = get_config()
    tools = TravelTools(cfg.data.flights_path, cfg.data.bookings_path)
    a = cfg.agent
    print(f"\n=== {model} ===")
    res = run_agent(
        task, tools, functools.partial(ollama_chat, model=model),
        a.max_steps, a.max_llm_retries, a.max_repeat_actions,
        on_step=lambda s: print(f"[{s.n}] Thought: {s.thought}\n    Action: {s.action} {s.action_input}\n"
                                f"    Observation: {s.observation[:200]}"))
    print(f"-> {res.status}: {res.answer}")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--model", default=get_config().llm.model)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    for m in (get_config().llm.compare_models if args.compare else (args.model,)):
        run(args.task, m)


if __name__ == "__main__":
    main()
