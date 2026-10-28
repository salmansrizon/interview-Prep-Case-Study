"""CLI for the Offline Market Intelligence Team (the API is api/main.py, the UI is ui.py).

  python cli.py snapshot                      # (re)load data/snapshot into data/market.db
  python cli.py risk ACME                     # deterministic Risk Score only (FinBERT, no LLM)
  python cli.py analyze ACME                  # full Flow: agents + verification + PDF memo
  python cli.py analyze ACME --mode tool      # Risk Analyst Agent must call tools itself (failure demo)
  python cli.py refresh AAPL MSFT             # OPTIONAL, the only network step (yfinance + RSS)
"""
from __future__ import annotations

import argparse
import json
import sys

import config  # noqa: F401 - telemetry env vars first
from config import get_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Offline Market Intelligence Team (CrewAI + FinBERT + Ollama)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="(re)load the bundled Offline Snapshot into the local SQLite store")
    r = sub.add_parser("refresh", help="optional: fetch real prices (yfinance) + RSS headlines into the same store")
    r.add_argument("tickers", nargs="+")
    r.add_argument("--period", default="1y")
    r.add_argument("--rss", default=None, help="RSS URL template with {ticker}")
    k = sub.add_parser("risk", help="deterministic Risk Score for one ticker")
    k.add_argument("ticker")
    a = sub.add_parser("analyze", help="run the agents and publish a PDF memo")
    a.add_argument("ticker")
    a.add_argument("--mode", choices=("grounded", "tool"), default="grounded")
    args = p.parse_args(argv)

    from src.store.db import default_store
    store, cfg = default_store(), get_config()

    if args.cmd == "snapshot":
        print(store.load_snapshot(cfg.data.snapshot_dir))
        for t in store.tickers():
            print(f"  {t['ticker']:6} {t['name']:28} {t['source']:10} {t['prices']} prices, {t['news']} news")
        return 0

    if args.cmd == "refresh":
        from src.store.refresh import YAHOO_RSS, refresh
        print(json.dumps(refresh(store, args.tickers, args.period, args.rss or YAHOO_RSS), indent=2))
        return 0

    from src.sentiment.finbert import FinBert
    if args.cmd == "risk":
        from src.risk.metrics import assess
        r, news = assess(store, args.ticker.upper(), FinBert())
        print(f"{r.ticker} as of {r.as_of}: Risk Score {r.score} / 100 ({r.band})")
        for k_, v in r.metrics.items():
            print(f"  {k_:13} raw {v:+.4f}  -> {r.components[k_]:5.1f} risk points x weight {r.weights[k_]}")
        for n in news[-5:]:
            print(f"  [{n['id']}] {n['date']} {n['label']:8} {n['headline']}")
        return 0

    from src.flow.market_flow import run_analysis
    from src.llm.factory import ollama_status
    ok, msg = ollama_status()
    if not ok:
        print(msg)
        return 1
    state = run_analysis(args.ticker.upper(), args.mode, store=store, classifier=FinBert(),
                         on_task=lambda o: print(f"  done: {o.agent} ({len(o.raw)} chars)"))
    print(f"\n{state.brief}\n\n{state.memo}\n\noutcome={state.outcome} rounds={state.rounds} {state.seconds}s")
    print(json.dumps(state.verification, indent=2))
    print("pdf:", state.pdf_path or "(none: needs human review)")
    return 0 if state.outcome == "published" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, RuntimeError, ValueError) as e:   # UnknownTicker, SentimentUnavailable, InsufficientData, ...
        sys.exit(f"error: {e}")
