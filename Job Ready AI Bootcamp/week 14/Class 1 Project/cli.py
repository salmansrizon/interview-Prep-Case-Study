"""CLI for the Self-Correcting LLM Coder (the Streamlit UI is app.py).

  python cli.py                         # list tasks
  python cli.py roman                   # solve one task, ask for approval at the end
  python cli.py roman --no-approval
  python cli.py --bench                 # all tasks: pass@1 vs pass after self-repair
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph.types import Command

from config import get_config
from src.evaluation.bench import run_bench, summarize
from src.graph.coder import build_graph, initial_state, run_config, sqlite_checkpointer
from src.llm.client import OllamaUnavailable, generate
from src.tasks.bank import TASKS


def solve(task_id: str, approval: bool) -> int:
    graph = build_graph(generate, require_approval=approval, checkpointer=sqlite_checkpointer())
    cfg = run_config(f"cli-{task_id}-{uuid.uuid4().hex[:8]}")
    payload = initial_state(TASKS[task_id])
    while True:
        interrupted = None
        # stream_mode="updates": one chunk per finished node — shows the loop live
        for chunk in graph.stream(payload, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupted = update[0].value
                    continue
                for line in (update or {}).get("history", []):
                    print(f"  [{node}] {line}")
        if not interrupted:
            break
        print(f"\n--- code (visible {interrupted['visible']}, hidden {interrupted['hidden']}) ---\n{interrupted['code']}\n")
        payload = Command(resume=input("Approve? [y/N] ").strip().lower() == "y")
    final = graph.get_state(cfg).values
    print(f"\nstatus={final['status']} attempts={final['attempts']}\n{final['code']}")
    return 0 if final["status"] == "done" else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Self-correcting LLM coder (LangGraph + Ollama)")
    p.add_argument("task", nargs="?", choices=list(TASKS))
    p.add_argument("--no-approval", action="store_true")
    p.add_argument("--bench", action="store_true", help="run every task, report pass@1 vs after repair")
    a = p.parse_args()
    try:
        if a.bench:
            rows = run_bench(generate)
            for r in rows:
                print(f"  {r.task:<15} {r.status:<9} attempts={r.attempts} first_ok={r.first_attempt_ok} "
                      f"visible={r.visible} hidden={r.hidden} {r.seconds}s")
            print(json.dumps(summarize(rows), indent=2))
            return 0
        if not a.task:
            for t in TASKS.values():
                print(f"  {t.id:<15} {t.prompt[:90]}")
            return 0
        return solve(a.task, approval=get_config().agent.require_approval and not a.no_approval)
    except OllamaUnavailable as e:
        print(f"[!] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
