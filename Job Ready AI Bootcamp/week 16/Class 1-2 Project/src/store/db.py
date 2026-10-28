"""
Local data persistence: one SQLite file (``data/market.db``, git-ignored) loaded from the Offline Snapshot.

HIGHLIGHTS:
* stdlib ``sqlite3`` — server লাগে না, একটা file-ই database। Agent আর API শুধু এই local store পড়ে;
  network ছোঁয় শুধু optional ``refresh`` (``src/store/refresh.py``), আর সেটাও এই একই table-এ লেখে।
* পাঁচটা table: ``tickers`` / ``prices`` / ``news`` (snapshot বা refresh থেকে), ``sentiment``
  (FinBERT-এর cache — একই headline দুবার model-এ যায় না), ``memos`` (কোন PDF, কোন Risk Score-এ, কখন)।
* প্রথমবার খুললে store খালি থাকলে snapshot নিজে থেকেই load হয় — তাই ``market.db`` commit করতে হয় না।
* প্রতিটা call নতুন connection খোলে, তাই FastAPI-র threadpool-এ thread-safety নিয়ে ভাবতে হয় না।
  ponytail: connection-per-call; অনেক request/second হলে WAL mode বা connection pool।
"""
from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (ticker TEXT PRIMARY KEY, name TEXT, sector TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS prices  (ticker TEXT, date TEXT, close REAL, PRIMARY KEY (ticker, date));
CREATE TABLE IF NOT EXISTS news    (id TEXT PRIMARY KEY, ticker TEXT, date TEXT, headline TEXT,
                                    summary TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS sentiment (model TEXT, text_hash TEXT, label TEXT, positive REAL, negative REAL,
                                      neutral REAL, PRIMARY KEY (model, text_hash));
CREATE TABLE IF NOT EXISTS memos   (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, created_at TEXT,
                                    path TEXT, risk_score REAL, band TEXT, mode TEXT);
"""


class UnknownTicker(KeyError):
    def __str__(self) -> str:
        return f"Unknown ticker {self.args[0]!r}: not in the local store (see /v1/tickers or `cli.py snapshot`)"


class MarketStore:
    def __init__(self, db_path: Path | str, snapshot_dir: Path | str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            empty = c.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 0
        if empty and snapshot_dir:
            self.load_snapshot(snapshot_dir)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:                       # commit on success, rollback on error
                yield conn
        finally:
            conn.close()

    # ── writes ────────────────────────────────────────────────────────
    def load_snapshot(self, snapshot_dir: Path | str) -> dict:
        """(Re)load the bundled snapshot. Replaces only the snapshot's tickers; refreshed real tickers stay."""
        d = Path(snapshot_dir)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        with open(d / "prices.csv", newline="", encoding="utf-8") as f:
            prices = [(r["ticker"], r["date"], float(r["close"])) for r in csv.DictReader(f)]
        news = json.loads((d / "news.json").read_text(encoding="utf-8"))
        tickers = [(t["ticker"], t["name"], t["sector"], "synthetic") for t in manifest["tickers"]]
        with self._conn() as c:
            marks = ",".join("?" * len(tickers))
            for table in ("tickers", "prices", "news"):
                c.execute(f"DELETE FROM {table} WHERE ticker IN ({marks})", [t[0] for t in tickers])
        self.upsert(tickers=tickers, prices=prices, news=news)
        return {"version": manifest["version"], "tickers": len(tickers), "prices": len(prices), "news": len(news)}

    def upsert(self, tickers=(), prices=(), news=()) -> None:
        """tickers: (ticker, name, sector, source); prices: (ticker, date, close); news: dicts."""
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO tickers VALUES (?,?,?,?)", tickers)
            c.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?)", prices)
            c.executemany("INSERT OR REPLACE INTO news VALUES (:id,:ticker,:date,:headline,:summary,:source)", news)

    def save_sentiment(self, model: str, rows: dict[str, dict]) -> None:
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO sentiment VALUES (?,?,?,?,?,?)",
                          [(model, h, r["label"], r["positive"], r["negative"], r["neutral"]) for h, r in rows.items()])

    def add_memo(self, ticker: str, path: str, risk_score: float, band: str, mode: str) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO memos (ticker, created_at, path, risk_score, band, mode) VALUES (?,?,?,?,?,?)",
                            (ticker, datetime.now().isoformat(timespec="seconds"), path, risk_score, band, mode))
            return cur.lastrowid

    # ── reads ─────────────────────────────────────────────────────────
    def tickers(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT t.ticker, t.name, t.sector, t.source,
                       (SELECT COUNT(*) FROM prices p WHERE p.ticker = t.ticker) AS prices,
                       (SELECT MIN(date) FROM prices p WHERE p.ticker = t.ticker) AS first_date,
                       (SELECT MAX(date) FROM prices p WHERE p.ticker = t.ticker) AS last_date,
                       (SELECT COUNT(*) FROM news n WHERE n.ticker = t.ticker) AS news
                FROM tickers t ORDER BY t.ticker""").fetchall()
        return [dict(r) for r in rows]

    def ticker(self, ticker: str) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tickers WHERE ticker = ?", (ticker,)).fetchone()
        if row is None:
            raise UnknownTicker(ticker)
        return dict(row)

    def closes(self, ticker: str) -> list[tuple[str, float]]:
        with self._conn() as c:
            rows = c.execute("SELECT date, close FROM prices WHERE ticker = ? ORDER BY date", (ticker,)).fetchall()
        if not rows:
            raise UnknownTicker(ticker)
        return [(r["date"], r["close"]) for r in rows]

    def news(self, ticker: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM news WHERE ticker = ? ORDER BY date, id", (ticker,)).fetchall()
        return [dict(r) for r in rows]

    def cached_sentiment(self, model: str, hashes: list[str]) -> dict[str, dict]:
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM sentiment WHERE model = ? AND text_hash IN ({','.join('?' * len(hashes))})",
                             [model, *hashes]).fetchall()
        return {r["text_hash"]: {k: r[k] for k in ("label", "positive", "negative", "neutral")} for r in rows}

    def latest_memo(self, ticker: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memos WHERE ticker = ? ORDER BY id DESC LIMIT 1", (ticker,)).fetchone()
        return dict(row) if row else None


def default_store() -> MarketStore:
    from config import get_config
    d = get_config().data
    return MarketStore(d.db_path, d.snapshot_dir)
