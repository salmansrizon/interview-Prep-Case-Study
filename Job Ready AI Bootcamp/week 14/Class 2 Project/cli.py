"""CLI for the multi-agent research team (the Streamlit UI is app.py).

  python cli.py "Why multi-agent systems fail"                 # sequential crew inside the Flow
  python cli.py "Safe tool calling" --mode single               # single-agent baseline
  python cli.py "Safe tool calling" --mode tool                 # researcher calls the search tool itself (experiment)
  python cli.py "Safe tool calling" --mode hierarchical         # manager LLM (experiment)
  python cli.py --search "multi-agent failure modes"            # just the corpus tool
"""
from __future__ import annotations

import argparse
import json
import sys

import config  # noqa: F401 - telemetry env vars first
from config import get_config
from src.crew.team import MODES
from src.flow.research_flow import run_research
from src.llm.factory import ollama_status
from src.report import save_report
from src.tools.corpus_search import _cached_corpus


def main(argv=None) -> int:
    cfg = get_config()
    p = argparse.ArgumentParser(description="Local multi-agent research team (CrewAI + Ollama)")
    p.add_argument("topic", nargs="?")
    p.add_argument("--mode", choices=MODES, default="sequential")
    p.add_argument("--search", help="run only the corpus search tool")
    a = p.parse_args(argv)
    if a.search:
        for passage, score in _cached_corpus(str(cfg.data.corpus_dir)).search(a.search, cfg.crew.search_top_k):
            print(f"[{passage.id}] {score:5.2f}  {passage.title}: {passage.text[:110]}...")
        return 0
    if not a.topic:
        p.error("give a topic, or use --search")
    ok, msg = ollama_status()
    if not ok:
        print(msg)
        return 1
    state = run_research(a.topic, a.mode, on_task=lambda o: print(f"  ✓ task done: {o.agent} ({len(o.raw)} chars)"))
    path = save_report(state.report, a.topic, cfg.data.output_dir)
    print(f"\n{state.report}\n\noutcome={state.outcome} rounds={state.rounds} {state.seconds}s")
    print(json.dumps(state.verification, indent=2))
    print("saved:", path)
    return 0 if state.outcome == "published" else 2


if __name__ == "__main__":
    sys.exit(main())
