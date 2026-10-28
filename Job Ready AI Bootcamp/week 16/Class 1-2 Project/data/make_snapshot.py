"""
Build the Offline Snapshot: synthetic daily closes + authored news for 8 FICTIONAL companies.

Run:  python data/make_snapshot.py        # rewrites data/snapshot/*  (same seed -> same bytes)

HIGHLIGHTS:
* সব company, ticker, price আর news **synthetic**। কোনো real company-র নামে news
  বানানো হয়নি; কোনো ticker কোনো real listing-এর symbol-এর সাথে মিলে গেলে সেটা কাকতালীয়।
* Price = geometric Brownian motion (GBM): প্রতিদিন log return ~ Normal((mu - sigma^2/2)*dt, sigma*sqrt(dt)),
  dt = 1/252। প্রতিটা ticker-এর নিজের drift (``mu``) আর volatility (``sigma``)। HELX-এ এক দিনের crash
  (trial failure), VALT-এ কয়েক দিনের slide (regulatory probe) — তাই drawdown আর VaR সত্যিই আলাদা হয়।
* প্রতিটা ticker-এর নিজের ``random.Random(f"{SEED}:{ticker}")`` — global random state ছোঁয় না, আর
  একটা ticker যোগ করলে অন্যগুলোর দাম বদলায় না। Test যাচাই করে যে আবার চালালে byte-by-byte একই file আসে।
* News-এর তারিখ trading-day index দিয়ে লেখা, তাই crash-এর দিনের headline ঠিক সেই দিনেই পড়ে।
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 16
VERSION = "2025.1"
START = date(2025, 1, 2)
TRADING_DAYS = 252
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshot"
DISCLAIMER = ("SYNTHETIC DATA. Every company, ticker, price and headline here is invented for teaching. "
              "Any match with a real listing is coincidental. Not investment advice.")

# ticker, company, sector, start price, drift (mu), volatility (sigma), {trading day: extra one-day return}
COMPANIES = [
    ("ACME", "Acme Robotics", "Industrials", 84.0, 0.10, 0.28, {}),
    ("ZEPH", "Zephyr Wind Power", "Utilities", 41.0, 0.06, 0.22, {}),
    ("KRYO", "Kryo Cold Chain Logistics", "Logistics", 57.0, 0.04, 0.18, {}),
    ("LUMA", "Luma Optics", "Semiconductors", 132.0, 0.18, 0.38, {}),
    ("QBIT", "Qubit Compute", "Technology", 23.0, 0.25, 0.55, {}),
    ("HELX", "Helix Biotherapeutics", "Biotechnology", 68.0, 0.05, 0.45, {180: -0.38}),
    ("MRDN", "Meridian Pantry Foods", "Consumer Staples", 49.0, 0.05, 0.14, {}),
    ("VALT", "Valtara Payments", "Financials", 76.0, 0.08, 0.30, {d: -0.04 for d in range(95, 103)}),
]

# ticker, trading day index, headline, one-sentence summary  (all fictional)
NEWS = [
    ("ACME", 5, "Acme Robotics opens second assembly plant to meet warehouse demand",
     "The new plant doubles annual output of the company's picking arms."),
    ("ACME", 40, "Acme Robotics reports quarterly revenue in line with guidance",
     "Revenue matched the range management gave three months earlier."),
    ("ACME", 88, "Acme Robotics wins multi-year contract from a regional grocery chain",
     "The contract covers automation for twelve distribution centers."),
    ("ACME", 131, "Acme Robotics recalls 1,200 picking arms over a gripper sensor fault",
     "The company expects the recall to cost one quarter of service margin."),
    ("ACME", 168, "Acme Robotics beats third-quarter earnings estimates as recall costs fade",
     "Earnings per share came in above the analyst consensus."),
    ("ACME", 205, "Acme Robotics names new chief financial officer",
     "The incoming CFO previously ran finance at an industrial supplier."),
    ("ACME", 240, "Acme Robotics raises full-year outlook on strong service revenue",
     "Management lifted its revenue forecast for the second time this year."),

    ("ZEPH", 12, "Zephyr Wind Power commissions 300 MW offshore array",
     "The array is the company's largest project to date."),
    ("ZEPH", 55, "Zephyr Wind Power output falls as calm weather lowers generation",
     "Quarterly generation was 14 percent below the long-term average."),
    ("ZEPH", 97, "Zephyr Wind Power keeps quarterly dividend unchanged",
     "The board held the payout at the same level as last year."),
    ("ZEPH", 140, "Regulator approves Zephyr Wind Power grid connection fee increase",
     "The higher fee adds a stable stream of regulated revenue."),
    ("ZEPH", 190, "Zephyr Wind Power delays turbine order after supplier backlog",
     "Two onshore projects will start about six months later than planned."),
    ("ZEPH", 236, "Zephyr Wind Power signs 15-year power purchase agreement with a data center operator",
     "The agreement fixes the price for most of one wind farm's output."),

    ("KRYO", 8, "Kryo Cold Chain adds 40 refrigerated trucks to its fleet",
     "The expansion targets growing demand for temperature-controlled food delivery."),
    ("KRYO", 60, "Kryo Cold Chain reports steady volumes in pharmaceutical shipping",
     "Shipment volumes were flat compared with the previous quarter."),
    ("KRYO", 110, "Kryo Cold Chain fined for temperature logging gaps at one depot",
     "An inspector found missing records for three weeks of shipments."),
    ("KRYO", 150, "Kryo Cold Chain renews contract with its largest vaccine distributor",
     "The renewal runs for four more years."),
    ("KRYO", 200, "Kryo Cold Chain quarterly profit flat as fuel costs offset price increases",
     "Higher diesel prices absorbed the gain from new contract rates."),
    ("KRYO", 245, "Kryo Cold Chain completes acquisition of a regional warehouse operator",
     "The deal adds six cold-storage sites."),

    ("LUMA", 15, "Luma Optics unveils faster photonic interconnect chip",
     "The chip moves data between servers at twice the previous speed."),
    ("LUMA", 48, "Luma Optics shares slide after a key customer delays orders",
     "The customer pushed deliveries into the next quarter."),
    ("LUMA", 92, "Luma Optics revenue jumps 35 percent on data center demand",
     "Sales of interconnect chips drove the increase."),
    ("LUMA", 138, "Luma Optics warns of supply constraints for specialty wafers",
     "A single supplier provides most of the wafers the company uses."),
    ("LUMA", 180, "Luma Optics announces share buyback program",
     "The board approved repurchases worth about 3 percent of shares outstanding."),
    ("LUMA", 222, "Luma Optics cuts guidance as two large customers pause spending",
     "Management lowered its full-year revenue forecast."),
    ("LUMA", 248, "Luma Optics to present at an industry technology conference",
     "The chief technology officer will give a product update."),

    ("QBIT", 3, "Qubit Compute shares fall after dilutive secondary offering",
     "The company sold new shares to fund research, diluting existing holders."),
    ("QBIT", 35, "Qubit Compute demonstrates 128-qubit processor in lab test",
     "The test showed lower error rates than the previous generation."),
    ("QBIT", 77, "Qubit Compute net loss widens as research spending grows",
     "Operating expenses rose faster than revenue."),
    ("QBIT", 120, "Qubit Compute signs pilot agreement with a national laboratory",
     "The pilot gives researchers cloud access to the company's processors."),
    ("QBIT", 165, "Short seller report questions Qubit Compute revenue recognition",
     "The report claims some pilot revenue was booked too early."),
    ("QBIT", 210, "Qubit Compute responds to short seller report and reaffirms its financial statements",
     "The audit committee said its review found no errors."),
    ("QBIT", 243, "Qubit Compute shares rally after cloud partnership announcement",
     "A cloud provider will offer the company's processors to its customers."),

    ("HELX", 10, "Helix Biotherapeutics starts Phase 3 trial of its lead arthritis drug",
     "The trial will enroll about 900 patients."),
    ("HELX", 62, "Helix Biotherapeutics reports positive early safety data",
     "No serious side effects were seen in the first patient group."),
    ("HELX", 115, "Helix Biotherapeutics extends cash runway with a new credit facility",
     "The facility funds operations into the following year."),
    ("HELX", 180, "Helix Biotherapeutics Phase 3 trial misses its primary endpoint",
     "The drug did not reduce joint pain more than placebo."),
    ("HELX", 182, "Analysts downgrade Helix Biotherapeutics after trial failure",
     "Several analysts cut their price targets by more than half."),
    ("HELX", 215, "Helix Biotherapeutics announces 30 percent workforce reduction to cut costs",
     "The company will focus on two early-stage programs."),
    ("HELX", 240, "Helix Biotherapeutics explores strategic alternatives, including a sale",
     "The board hired an adviser to review options."),

    ("MRDN", 20, "Meridian Pantry Foods raises dividend for the 12th straight year",
     "The quarterly dividend increases by 4 percent."),
    ("MRDN", 70, "Meridian Pantry Foods reports stable sales of packaged staples",
     "Volumes were unchanged from the same quarter last year."),
    ("MRDN", 125, "Meridian Pantry Foods passes higher grain costs to customers with modest price increases",
     "Prices on cereal and pasta products rise by about 2 percent."),
    ("MRDN", 170, "Meridian Pantry Foods quarterly earnings meet analyst expectations",
     "Margins held steady despite higher input costs."),
    ("MRDN", 230, "Meridian Pantry Foods expands distribution to two new regions",
     "Its products will reach about 800 more grocery stores."),
    ("MRDN", 250, "Meridian Pantry Foods recalls one batch of canned soup as a precaution",
     "The company said the recall affects less than 0.1 percent of annual sales."),

    ("VALT", 18, "Valtara Payments processes record transaction volume in January",
     "Holiday returns and new merchants lifted payment volume."),
    ("VALT", 65, "Valtara Payments launches instant payouts for small merchants",
     "Merchants can now receive funds within minutes instead of days."),
    ("VALT", 95, "Regulator opens investigation into Valtara Payments anti-money-laundering controls",
     "The investigation covers how the company checks new merchants."),
    ("VALT", 101, "Valtara Payments suspends merchant onboarding during regulatory review",
     "Existing merchants are not affected by the pause."),
    ("VALT", 150, "Valtara Payments reports slower growth as merchant onboarding stays paused",
     "Revenue growth slowed to its lowest rate in three years."),
    ("VALT", 200, "Valtara Payments reaches settlement with regulator and resumes onboarding",
     "The company agreed to pay a fine and hire an independent monitor."),
    ("VALT", 246, "Valtara Payments quarterly revenue returns to growth",
     "New merchant sign-ups recovered after onboarding resumed."),
]


def trading_dates(start: date, n: int) -> list[date]:
    """Weekdays only. ponytail: no exchange holidays — synthetic calendar; real data comes from refresh."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def gbm_closes(rng: random.Random, start: float, mu: float, sigma: float, shocks: dict[int, float],
               n: int = TRADING_DAYS) -> list[float]:
    dt = 1 / 252
    price, out = start, []
    for day in range(n):
        if day:
            price *= math.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rng.gauss(0.0, 1.0))
            price *= 1 + shocks.get(day, 0.0)
        out.append(round(price, 2))
    return out


def build(out_dir: Path = SNAPSHOT_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = trading_dates(START, TRADING_DAYS)

    with open(out_dir / "prices.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "ticker", "close"])
        for t, _, _, p0, mu, sigma, shocks in COMPANIES:
            for d, close in zip(days, gbm_closes(random.Random(f"{SEED}:{t}"), p0, mu, sigma, shocks)):
                w.writerow([d.isoformat(), t, f"{close:.2f}"])

    news, seq = [], {}
    for t, day, headline, summary in sorted(NEWS, key=lambda n: (n[0], n[1])):
        seq[t] = seq.get(t, 0) + 1
        news.append({"id": f"news:{t}:{seq[t]}", "ticker": t, "date": days[day].isoformat(),
                     "headline": headline, "summary": summary, "source": "synthetic"})
    _write_json(out_dir / "news.json", news)

    _write_json(out_dir / "manifest.json", {
        "version": VERSION, "seed": SEED, "synthetic": True, "disclaimer": DISCLAIMER,
        "start": days[0].isoformat(), "end": days[-1].isoformat(), "trading_days": TRADING_DAYS,
        "tickers": [{"ticker": t, "name": name, "sector": sector, "start_price": p0, "drift": mu,
                     "volatility": sigma, "shock_days": sorted(shocks)}
                    for t, name, sector, p0, mu, sigma, shocks in COMPANIES]})
    return out_dir


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    print("snapshot written to", build())
