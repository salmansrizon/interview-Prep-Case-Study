"""Monitor = service.observers-এ বসানো একটা observer: metrics হিসাব -> SQLite -> (optional) sinks।

HIGHLIGHTS: SQLite হলো source of truth (Docker/Langfuse ছাড়াও চলে); Langfuse একটা
optional sink। এক observer-এ সব রাখায় quality/cost একবারই হিসাব হয়, দুই জায়গায় একই মান যায়।
"""

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .pricing import equivalent_cost
from .quality import drift_report, grounding_score
from .service import CallRecord

SCHEMA = """CREATE TABLE IF NOT EXISTS calls(
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, request_id TEXT, op TEXT, ok INTEGER, error TEXT,
 latency_s REAL, model TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
 cost_usd REAL, quality REAL, quality_label TEXT, found INTEGER, question TEXT)"""


class Monitor:
    def __init__(self, db_path: Path | str, model: str = "llama3.1:8b", price_key: str = "small-cloud"):
        self.model, self.price_key = model, price_key
        self._lock = threading.Lock()
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(SCHEMA)
        self.sinks: list[Callable[[dict, CallRecord], None]] = []  # e.g. Langfuse

    def metrics(self, rec: CallRecord) -> dict:
        res = rec.result if rec.op == "ask" and rec.ok else None
        usage = (getattr(res, "usage", None) or {}) if res else {}
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        q, label = grounding_score(res.answer, res.context, res.found) if res else (None, None)
        return dict(
            ts=time.time(), request_id=rec.request_id, op=rec.op, ok=int(rec.ok), error=rec.error,
            latency_s=round(rec.latency_s, 4), model=self.model, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=equivalent_cost(self.price_key, pt, ct) if pt is not None else None,
            quality=q, quality_label=label, found=int(res.found) if res else None,
            question=getattr(res, "question", None) if res else None,
        )

    def observer(self, rec: CallRecord) -> None:
        m = self.metrics(rec)
        with self._lock, self._db:
            self._db.execute(
                f"INSERT INTO calls({','.join(m)}) VALUES({','.join('?' * len(m))})", tuple(m.values()))
        for sink in self.sinks:
            try:
                sink(m, rec)
            except Exception:  # একটা sink ভাঙলে request যেন না ভাঙে
                pass

    def rows(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM calls WHERE op='ask' AND ok=1 ORDER BY id")]

    def stats(self, drift_n: int = 20) -> dict:
        with self._lock:
            tot = dict(self._db.execute("SELECT COUNT(*) n, SUM(ok=0) errors FROM calls WHERE op='ask'").fetchone())
        rows = self.rows()
        lat = sorted(r["latency_s"] for r in rows)
        pct = lambda f: lat[min(len(lat) - 1, int(f * len(lat)))] if lat else None  # noqa: E731
        qs = [r["quality"] for r in rows if r["quality"] is not None]
        return {
            "queries": tot["n"], "errors": tot["errors"] or 0,
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
            "equivalent_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 6),
            "cost_note": f"ILLUSTRATIVE equivalent cloud cost ({self.price_key}); local model is free",
            "latency_s": {"avg": sum(lat) / len(lat) if lat else None, "p50": pct(0.5), "p95": pct(0.95)},
            "quality_avg": sum(qs) / len(qs) if qs else None,
            "refusals": sum(1 for r in rows if r["found"] == 0),
            "drift": drift_report(rows, drift_n),
            "recent": rows[-10:][::-1],
        }
