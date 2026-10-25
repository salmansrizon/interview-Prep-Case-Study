"""SQLite request log: one row per ask/stream/ingest — the source of truth for /v1/stats, the dashboard,
the offline judge and trace review.

HIGHLIGHTS:
* **Prometheus বনাম log:** Prometheus aggregate রাখে (সস্তা, alert-এর জন্য), কিন্তু "কোন প্রশ্নে
  ভুল সংখ্যা এল?" বলতে পারে না। সেটার জন্য per-request row লাগে — এখানে SQLite, production-এ
  Postgres/ClickHouse/Langfuse।
* **WAL mode:** API লেখে, একই সময়ে judge CLI (আলাদা process) পড়ে/লেখে।
* **PII:** ``log_text=False`` দিলে প্রশ্ন/উত্তরের বদলে শুধু SHA-256 hash থাকে।
* **SLO burn rate:** bad fraction ÷ error budget। 1.0 = budget ঠিক মাস শেষে শেষ হবে; 14.4 = ২ দিনে শেষ।
  Google SRE Workbook-এর multiwindow rule: 1h আর 5m দুটোই > 14.4 হলে page।
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

COLUMNS = """ts REAL, request_id TEXT, op TEXT, ok INTEGER, error TEXT, outcome TEXT, latency_s REAL, queue_wait_s REAL,
embed_ms REAL, search_ms REAL, llm_ms REAL, ttft_ms REAL, prefill_ms REAL, decode_ms REAL, load_ms REAL,
prompt_tokens INTEGER, completion_tokens INTEGER, tokens_per_s REAL, top_distance REAL,
citations INTEGER, citation_valid INTEGER, numbers_supported INTEGER, unsupported_numbers TEXT,
lexical_support REAL, truncated INTEGER, model TEXT, question TEXT, answer TEXT, chunk_ids TEXT,
judge_supported INTEGER, judge_reason TEXT, feedback INTEGER, feedback_comment TEXT"""
NAMES = [c.split()[0] for c in COLUMNS.replace("\n", " ").split(",")]

# (window seconds, label) for burn rates; alert pairs follow the SRE Workbook (page 14.4x, page 6x)
WINDOWS = {"5m": 300, "30m": 1800, "1h": 3600, "6h": 21600}
ALERTS = [("page_fast", "1h", "5m", 14.4), ("page_slow", "6h", "30m", 6.0)]


def _pct(xs: list[float], p: float):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 3) if xs else None


class RequestLog:
    def __init__(self, path: str | Path, *, log_text: bool = True):
        self.log_text = log_text
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.db:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute(f"CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY AUTOINCREMENT, {COLUMNS})")
            self.db.execute("CREATE INDEX IF NOT EXISTS ix_ts ON requests(ts)")
            self.db.execute("CREATE INDEX IF NOT EXISTS ix_rid ON requests(request_id)")

    def _text(self, s: str | None) -> str | None:
        if s is None or self.log_text:
            return s
        return "sha256:" + hashlib.sha256(s.encode()).hexdigest()[:16]

    # ── writes ────────────────────────────────────────────────────────
    def add(self, row: dict) -> None:
        row = {k: v for k, v in row.items() if k in NAMES}
        row["question"], row["answer"] = self._text(row.get("question")), self._text(row.get("answer"))
        with self.lock, self.db:
            self.db.execute(f"INSERT INTO requests({','.join(row)}) VALUES({','.join('?' * len(row))})",
                            tuple(row.values()))

    def feedback(self, request_id: str, vote: int, comment: str | None) -> bool:
        with self.lock, self.db:
            cur = self.db.execute("UPDATE requests SET feedback=?, feedback_comment=? WHERE request_id=? "
                                  "AND op IN ('ask','stream')", (vote, comment, request_id))
        return cur.rowcount > 0

    def set_judge(self, row_id: int, supported: bool | None, reason: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE requests SET judge_supported=?, judge_reason=? WHERE id=?",
                            (None if supported is None else int(supported), reason, row_id))

    # ── reads ─────────────────────────────────────────────────────────
    def rows(self, since: float = 0.0, ops=("ask", "stream")) -> list[dict]:
        q = f"SELECT * FROM requests WHERE ts >= ? AND op IN ({','.join('?' * len(ops))}) ORDER BY id"
        with self.lock:
            return [dict(r) for r in self.db.execute(q, (since, *ops))]

    def recent(self, n: int = 25) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(
                "SELECT * FROM requests WHERE op IN ('ask','stream') ORDER BY id DESC LIMIT ?", (n,))]

    def unjudged(self, limit: int) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(
                "SELECT * FROM requests WHERE outcome='answered' AND judge_supported IS NULL "
                "AND (judge_reason IS NULL) ORDER BY id DESC LIMIT ?", (limit,))]

    def summary(self, window_s: float = 3600, *, slo_availability=0.99, slo_latency_s=10.0,
                slo_latency_target=0.95, now: float | None = None) -> dict:
        now = now or time.time()
        rows = self.rows(now - max(window_s, max(WINDOWS.values())))
        win = [r for r in rows if r["ts"] >= now - window_s]
        n = len(win)
        by = lambda o: sum(r["outcome"] == o for r in win)          # noqa: E731
        answered = [r for r in win if r["outcome"] == "answered"]
        rate = lambda k: round(sum(1 for r in win if k(r)) / n, 3) if n else None   # noqa: E731
        judged = [r for r in answered if r["judge_supported"] is not None]
        votes = [r["feedback"] for r in win if r["feedback"] is not None]
        return {
            "window_s": window_s, "requests": n,
            "outcomes": {o: by(o) for o in ("answered", "not_found", "cached", "degraded", "error", "disconnected")},
            "rates": {"error": rate(lambda r: not r["ok"] and r["outcome"] != "disconnected"), "not_found": rate(lambda r: r["outcome"] == "not_found"),
                      "degraded": rate(lambda r: r["outcome"] == "degraded"),
                      "cache_hit": rate(lambda r: r["outcome"] == "cached")},
            "latency_s": {"p50": _pct([r["latency_s"] for r in win if r["ok"]], .5),
                          "p95": _pct([r["latency_s"] for r in win if r["ok"]], .95)},
            "ttft_ms": {"p50": _pct([r["ttft_ms"] for r in win], .5), "p95": _pct([r["ttft_ms"] for r in win], .95)},
            "prefill_ms_p50": _pct([r["prefill_ms"] for r in answered], .5),
            "decode_tokens_per_s_p50": _pct([r["tokens_per_s"] for r in answered], .5),
            "tokens": {"prompt": sum(r["prompt_tokens"] or 0 for r in win),
                       "completion": sum(r["completion_tokens"] or 0 for r in win)},
            "gpu_seconds": round(sum((r["llm_ms"] or 0) for r in answered) / 1000, 2),
            "quality": {
                "answered": len(answered),
                "citation_valid_rate": _share(answered, "citation_valid"),
                "numbers_supported_rate": _share(answered, "numbers_supported"),
                "truncated": sum(r["truncated"] or 0 for r in answered),
                "lexical_support_p50": _pct([r["lexical_support"] for r in answered], .5),
                "judge_supported_rate": _share(judged, "judge_supported"), "judged": len(judged),
            },
            "feedback": {"up": votes.count(1), "down": votes.count(-1)},
            "slo": self.slo(rows, now, slo_availability, slo_latency_s, slo_latency_target),
        }

    @staticmethod
    def slo(rows, now, availability, latency_s, latency_target) -> dict:
        """Two SLIs, each with burn rates per window and multiwindow alert states."""
        def bad_avail(r):                       # a client hanging up is not a server failure
            return not r["ok"] and r["outcome"] != "disconnected"

        def bad_latency(r):
            return r["ok"] and r["outcome"] in ("answered", "not_found") and r["latency_s"] > latency_s

        out = {}
        for name, bad, target in (("availability", bad_avail, availability),
                                  ("latency", bad_latency, latency_target)):
            budget = 1 - target
            burn = {}
            for label, secs in WINDOWS.items():
                w = [r for r in rows if r["ts"] >= now - secs]
                burn[label] = round(sum(map(bad, w)) / len(w) / budget, 2) if w else None
            alerts = [a for a, long_w, short_w, thr in ALERTS
                      if (burn[long_w] or 0) > thr and (burn[short_w] or 0) > thr]
            out[name] = {"target": target, "burn_rate": burn, "alerts": alerts}
        out["latency"]["threshold_s"] = latency_s
        return out


def _share(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r[key] is not None]
    return round(sum(vals) / len(vals), 3) if vals else None
