"""CLI for the healthcare assistant (in-process; the HTTP API is src/api/main.py, the UI is ui.py).

  python cli.py ingest [--reset]                    # index data/kb into Chroma (--reset after changing HC_EMBEDDING_MODEL)
  python cli.py ask "How do I prepare ORS?"         # one question -> route, answer, citations
  python cli.py eval [--file data/eval/questions.jsonl] [--out output/eval_report.json]
  HC_BACKEND=fake python cli.py eval                # no Ollama, no model download

``eval`` exits with code 2 when any safety metric fails (unsafe_answers or missed_escalations > 0), so it can gate CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.backend import build_assistant
from src.evaluation.runner import load_questions, run_eval
from src.rag.components import LLMUnavailable
from src.settings import PROJECT_DIR, get_settings


def _ready_assistant(s):
    a = build_assistant(s)
    if a.store.count() == 0:
        print(f"(collection empty: indexing {s.kb_dir} first)", file=sys.stderr)
        a.ingest(s.kb_dir)
    return a


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Privacy-first healthcare assistant (local RAG)")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="index the clinic handbook")
    ing.add_argument("--reset", action="store_true", help="drop the collection first")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ev = sub.add_parser("eval", help="route accuracy + retrieval hit-rate on the eval set")
    ev.add_argument("--file", default=str(PROJECT_DIR / "data" / "eval" / "questions.jsonl"))
    ev.add_argument("--out", help="write the full report (summary + rows) as JSON")
    a = p.parse_args(argv)
    s = get_settings()

    if a.cmd == "ingest":
        assistant = build_assistant(s)
        if a.reset:
            assistant.store.reset()
        r = assistant.ingest(s.kb_dir)
        print(f"indexed {r.pages} pages / {r.sections} sections, removed {r.removed} stale "
              f"({'chroma ' + s.chroma_host if s.chroma_host else s.chroma_dir})")
        return 0

    assistant = _ready_assistant(s)
    if a.cmd == "ask":
        try:
            r = assistant.ask(a.question)
        except LLMUnavailable as exc:
            print(f"{exc}. Start it with: ollama serve  (and: ollama pull {s.model})")
            return 1
        print(f"route: {r.route}" + (f"  (rule: {r.rule})" if r.rule else ""))
        print(f"redacted question: {r.redaction.text}  {r.redaction.counts or ''}")
        print(f"\n{r.answer}\n")
        for n, h in r.citations:
            print(f"  [{n}] {h.id}  {h.page} - {h.title}  (distance {h.distance:.3f})")
        print(f"\ntimings_ms: {r.timings_ms}")
        return 0

    report = run_eval(assistant.ask, load_questions(a.file))
    for row in report["rows"]:
        if not row["ok"] or row.get("hit") is False:
            print(f"  MISS {row['id']:4} expected={row['expected']:9} got={row['got']:9} "
                  f"best_distance={row.get('best_distance')} hit={row.get('hit')}")
    print(json.dumps(report["summary"], indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("saved:", a.out)
    sm = report["summary"]
    return 2 if sm["unsafe_answers"] or sm["missed_escalations"] else 0


if __name__ == "__main__":
    sys.exit(main())
