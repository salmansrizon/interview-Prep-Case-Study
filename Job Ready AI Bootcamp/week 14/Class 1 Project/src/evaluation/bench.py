"""
Benchmark: how much does the self-correction loop actually help?

HIGHLIGHTS: "Is Self-Repair a Silver Bullet?" (ICLR 2024) দেখিয়েছে repair-এর
লাভ প্রায়ই ছোট, আর feedback-এর মানই আসল bottleneck। তাই নিজের task-এ মাপি:

* **pass@1** — প্রথম attempt-ই visible + hidden সব test পাস করে
* **pass after repair** — retry budget-এর মধ্যে শেষ পর্যন্ত সব পাস
* **hidden_fail** — budget শেষেও visible পাস কিন্তু hidden fail (example মুখস্থ বা spec অর্ধেক বোঝা)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from src.graph.coder import build_graph, initial_state, run_config
from src.tasks.bank import TASKS


@dataclass
class BenchRow:
    task: str
    status: str
    attempts: int
    first_attempt_ok: bool
    final_ok: bool
    visible: str
    hidden: str
    seconds: float


def run_bench(llm, task_ids=None, max_attempts: int | None = None) -> list[BenchRow]:
    rows = []
    for tid in task_ids or list(TASKS):
        task = TASKS[tid]
        graph = build_graph(llm, max_attempts=max_attempts, require_approval=False)
        t0 = time.perf_counter()
        s = graph.invoke(initial_state(task), run_config(f"bench-{tid}-{uuid.uuid4().hex[:6]}"))
        final_ok = s["status"] == "done"
        rows.append(BenchRow(
            tid, s["status"], s["attempts"],
            first_attempt_ok=bool(s.get("first_attempt_passed")),
            final_ok=final_ok,
            # give_up keeps the BEST attempt, so report its score, not the last attempt's
            visible=f"{s.get('best_passed', 0) if s['status'] == 'gave_up' else s.get('visible_passed', 0)}/{len(task.visible)}",
            hidden=f"{s.get('hidden_passed', 0)}/{len(task.hidden)}" if s["status"] in ("done", "hidden_fail") else "-",
            seconds=round(time.perf_counter() - t0, 1)))
    return rows


def summarize(rows: list[BenchRow]) -> dict:
    n = len(rows) or 1
    return {
        "tasks": len(rows),
        "pass@1": round(sum(r.first_attempt_ok for r in rows) / n, 2),
        "pass_after_repair": round(sum(r.final_ok for r in rows) / n, 2),
        "repaired": sum(r.final_ok and not r.first_attempt_ok for r in rows),
        "hidden_fail": sum(r.status == "hidden_fail" for r in rows),
        "gave_up": sum(r.status == "gave_up" for r in rows),
        "avg_attempts": round(sum(r.attempts for r in rows) / n, 2),
        "avg_seconds": round(sum(r.seconds for r in rows) / n, 1),
    }
