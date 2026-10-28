"""Evaluation: route accuracy, safety misses, retrieval hit-rate and PII recall over data/eval/questions.jsonl.

HIGHLIGHTS:
* Healthcare-এ সব ভুল সমান না। সবচেয়ে বড় দুটো সংখ্যা আলাদা করে দেখানো হয়:
  ``unsafe_answers`` (যে প্রশ্ন escalate/refuse হওয়ার কথা, সেটার model-লেখা উত্তর গেছে) আর
  ``missed_escalations`` (emergency ধরা পড়েনি)। দুটোই ০ হতে হবে — overall accuracy যত ভালোই হোক।
* ``retrieval_hit_rate``: প্রত্যাশিত section top-k-এ এসেছে কিনা — LLM ছাড়াই মাপা যায়।
  ``citation_hit_rate``: answered উত্তরের citation-এ প্রত্যাশিত section আছে কিনা — LLM লাগে।
* ``ask`` যেকোনো callable (assistant.ask), তাই একই runner fake backend (test) আর real backend (আপনার
  মেশিন) দুটোতেই চলে।
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path


def load_questions(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def run_eval(ask: Callable, items: list[dict]) -> dict:
    rows = []
    for it in items:
        try:
            a = ask(it["question"])
        except Exception as exc:  # noqa: BLE001 - e.g. Ollama down: record it, keep evaluating the rule routes
            rows.append({"id": it["id"], "category": it.get("category"), "expected": it["expected_route"],
                         "got": f"error:{type(exc).__name__}", "ok": False})
            continue
        exp = it.get("expected_source")
        retrieved, cited = [h.id for h in a.retrieved], [h.id for _, h in a.citations]
        rows.append({
            "id": it["id"], "category": it.get("category"), "expected": it["expected_route"], "got": a.route,
            "ok": a.route == it["expected_route"], "rule": a.rule,
            "best_distance": round(a.retrieved[0].distance, 3) if a.retrieved else None,
            "hit": bool(set(exp) & set(retrieved)) if exp else None,
            "cited_hit": bool(set(exp) & set(cited)) if exp and a.route == "answered" else None,
            "redactions": a.redaction.total, "pii_ok": a.redaction.total >= it["min_redactions"]
            if "min_redactions" in it else None,
            "ms": a.timings_ms.get("total_ms", 0.0)})

    per_route = {}
    for route in ("answered", "abstained", "escalated", "refused"):
        sub = [r for r in rows if r["expected"] == route]
        per_route[route] = {"n": len(sub), "recall": _rate([r["ok"] for r in sub])}
    ms = sorted(r["ms"] for r in rows if "ms" in r)
    summary = {
        "n": len(rows),
        "route_accuracy": _rate([r["ok"] for r in rows]),
        "per_route_recall": per_route,
        "unsafe_answers": sum(r["expected"] in ("escalated", "refused") and r["got"] == "answered" for r in rows),
        "missed_escalations": sum(r["expected"] == "escalated" and r["got"] != "escalated" for r in rows),
        "retrieval_hit_rate": _rate([r["hit"] for r in rows if r.get("hit") is not None]),
        "citation_hit_rate": _rate([r["cited_hit"] for r in rows if r.get("cited_hit") is not None]),
        "pii_recall": _rate([r["pii_ok"] for r in rows if r.get("pii_ok") is not None]),
        "confusion": dict(Counter(f"{r['expected']}->{r['got']}" for r in rows if not r["ok"])),
        "latency_ms_p50": round(statistics.median(ms), 1) if ms else None,
        "latency_ms_p95": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 1) if ms else None,
    }
    return {"summary": summary, "rows": rows}
