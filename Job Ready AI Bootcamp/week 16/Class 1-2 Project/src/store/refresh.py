"""
Optional live refresh — the ONLY network step in this project. Nothing downstream needs it.

HIGHLIGHTS:
* ``yfinance`` থেকে real ticker-এর daily close, ``feedparser`` দিয়ে RSS থেকে headline — দুটোই
  **একই** SQLite table-এ লেখা হয়। তারপর Risk Score, agent, PDF — সব আগের মতোই offline চলে।
* দুটো library optional (``requirements.txt``-এ commented)। Import function-এর ভেতরে, না থাকলে
  পরিষ্কার error — বাকি project এগুলো ছাড়াই চলে।
* News id stable থাকে: আগের headline-এর id বদলায় না, নতুনগুলো পরের নম্বর পায় (``news:AAPL:7``)।
  একই headline আবার এলে বাদ।
* ponytail: Yahoo-র per-ticker RSS URL অনানুষ্ঠানিক, যেকোনো দিন বদলাতে পারে — ``--rss`` দিয়ে অন্য feed দিন।
"""
from __future__ import annotations

from datetime import date

from src.store.db import MarketStore

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


def refresh(store: MarketStore, tickers: list[str], period: str = "1y", rss: str = YAHOO_RSS) -> dict:
    try:
        import feedparser
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("`refresh` needs the optional live deps: pip install yfinance feedparser") from e

    report = {}
    for t in (t.upper() for t in tickers):
        closes = yf.Ticker(t).history(period=period, auto_adjust=True)["Close"]
        prices = [(t, ts.date().isoformat(), round(float(c), 4)) for ts, c in closes.items()]
        if not prices:
            report[t] = "no prices returned (wrong symbol?)"
            continue
        old = store.news(t)
        seen = {n["headline"] for n in old}
        seq = max((int(n["id"].rsplit(":", 1)[1]) for n in old), default=0)
        news = []
        for e in feedparser.parse(rss.format(ticker=t)).entries:
            title = (e.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            seq += 1
            p = e.get("published_parsed")
            news.append({"id": f"news:{t}:{seq}", "ticker": t, "date": (date(*p[:3]) if p else date.today()).isoformat(),
                         "headline": title, "summary": (e.get("summary") or "")[:300], "source": e.get("link") or rss})
        store.upsert(tickers=[(t, t, "", "yfinance")], prices=prices, news=news)
        report[t] = f"{len(prices)} prices, {len(news)} new headlines"
    return report
