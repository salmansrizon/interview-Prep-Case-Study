"""CLI Travel Agent (the Streamlit chat UI is app.py).

  python cli.py "Book the cheapest DAC to CXB flight on 2026-11-05 for Rahim"
  python cli.py --mode text "..."            # classic Thought/Action/Observation text ReAct
  python cli.py --eval -k 3                  # golden-task evaluation, pass^3
"""
from __future__ import annotations

import argparse
import json

from config import get_config
from src.agent.react_agent import Step, run_agent
from src.evaluation.harness import evaluate
from src.llm.client import chat
from src.tools.travel import TravelTools


def _ask_human(details: dict) -> bool:
    return input(f"\n  ⚠ Approve booking: {details['summary']}? [y/N] ").strip().lower() in ("y", "yes")


def _print_step(s: Step) -> None:
    if s.thought:
        print(f"[{s.n}] Thought: {s.thought[:200]}")
    if s.tool:
        print(f"    Action: {s.tool} {s.args}")
    print(f"    Observation: {json.dumps(s.observation, ensure_ascii=False)[:220]}")


def run(task: str, model: str, mode: str, auto_approve: bool) -> None:
    cfg = get_config()
    a = cfg.agent
    tools = TravelTools(cfg.data.flights_path, approve=(lambda s: True) if auto_approve else _ask_human,
                        max_results=a.max_results)
    print(f"\n=== {model} · {mode} ===")
    res = run_agent(task, tools, chat, mode=mode, model=model, max_steps=a.max_steps,
                    max_repeat_calls=a.max_repeat_calls, max_consecutive_errors=a.max_consecutive_errors,
                    deadline_seconds=a.deadline_seconds, max_llm_retries=a.max_llm_retries,
                    nudge_after_tools=a.nudge_after_tools,
                    on_step=_print_step)
    print(f"-> {res.status} ({res.llm_calls} LLM calls): {res.answer}")
    if tools.bookings:
        print("   bookings:", tools.bookings)


def main() -> None:
    cfg = get_config()
    p = argparse.ArgumentParser(description="ReAct travel agent")
    p.add_argument("task", nargs="?")
    p.add_argument("--model", default=cfg.llm.model)
    p.add_argument("--mode", choices=["native", "text"], default=cfg.agent.mode)
    p.add_argument("--yes", action="store_true", help="auto-approve bookings")
    p.add_argument("--eval", action="store_true", help="run the golden-task evaluation")
    p.add_argument("-k", type=int, default=1, help="trials per task for pass^k")
    p.add_argument("--no-nudge", action="store_true", help="disable the llama3.1 tool-visibility nudge")
    args = p.parse_args()
    models = (args.model,)
    if args.eval:
        for m in models:
            tools = TravelTools(cfg.data.flights_path, max_results=cfg.agent.max_results)
            rep = evaluate(chat, tools, k=args.k, mode=args.mode, model=m,
                           max_steps=cfg.agent.max_steps, nudge_after_tools=not args.no_nudge)
            print(f"\n=== eval: {m} · {args.mode} · k={args.k} ===")
            for t in rep.trials:
                print(f"  {'✓' if t.ok else '✗'} {t.task:<18} {t.status:<12} calls={t.llm_calls} {t.reason}")
            print(json.dumps(rep.summary(), indent=2))
        return
    if not args.task:
        p.error("give a task, or use --eval")
    for m in models:
        run(args.task, m, args.mode, args.yes)


if __name__ == "__main__":
    main()
