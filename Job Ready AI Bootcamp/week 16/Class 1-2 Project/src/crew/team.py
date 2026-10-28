"""
The two-agent team: News Scout Agent -> Risk Analyst Agent.

HIGHLIGHTS:
* **News Scout Agent** local snapshot-এর headline (id, তারিখ, FinBERT sentiment সহ) থেকে ঝুঁকির জন্য
  জরুরি news বাছে আর প্রতিটা bullet-এ id cite করে। **Risk Analyst Agent** code-এর দেওয়া Risk Score
  ব্যাখ্যা করে memo-র section লেখে। দুজনের কেউই কোনো সংখ্যা হিসাব বা বদলায় না।
* **Grounded mode (default):** Flow code-এ সব হিসাব করে, সংখ্যা আর news সরাসরি task description-এ বসায়।
  Week 14-এর live finding — llama3.1:8b CrewAI-র ভেতরে tool নির্ভরযোগ্যভাবে ডাকে না, নিজে result
  বানায় — তাই এটাই নিরাপদ পথ। **Tool mode:** Risk Analyst-কে ``get_risk_metrics`` /
  ``get_news_sentiment`` tool দেওয়া হয়, description-এ সংখ্যা থাকে না — failure mode দেখার পরীক্ষা।
* Task ``context=[brief]`` = dependency: News Scout-এর brief Risk Analyst-এর prompt-এ যায়।
* প্রতিটা task-এ **guardrail** — code-level check (id আসল কিনা, section আছে কিনা); fail করলে feedback
  সহ task আবার চলে। Guardrail ``(True, output.raw)`` ফেরত দেয়, আর তার return annotation নেই
  (``from __future__ import annotations`` থাকলে annotation string হয়ে যায়, CrewAI সেটা reject করে)।
* ``allow_delegation=False``, ``max_iter`` ছোট, ``memory=False`` — ছোট model-এ loop আর OpenAI embedder এড়াতে।
"""
from __future__ import annotations

import re
from typing import Callable

import config  # noqa: F401 - telemetry env vars first
from crewai import LLM, Agent, Crew, Process, Task

from config import get_config
from src.crew.tools import risk_tools
from src.verify.memo_check import cited_ids, missing_sections

MODES = ("grounded", "tool")


# ── what the agents see (all computed by code) ────────────────────────
def facts_text(risk: dict) -> str:
    m, c, w = risk["metrics"], risk["components"], risk["weights"]
    return (f"RISK FACTS for {risk['ticker']} as of {risk['as_of']} (computed by code; quote these numbers exactly):\n"
            f"- Risk Score: {risk['score']:.1f} out of 100 (band: {risk['band']})\n"
            f"- Annualized volatility: {m['volatility'] * 100:.1f}% -> {c['volatility']:.1f}/100 risk points, "
            f"weight {w['volatility']}\n"
            f"- Maximum drawdown: {m['max_drawdown'] * 100:.1f}% -> {c['max_drawdown']:.1f}/100 risk points, "
            f"weight {w['max_drawdown']}\n"
            f"- Historical VaR (95%, 1-day): {m['var_95'] * 100:.1f}% -> {c['var_95']:.1f}/100 risk points, "
            f"weight {w['var_95']}\n"
            f"- Net news sentiment: {m['sentiment']:+.2f} on a -1 to +1 scale -> {c['sentiment']:.1f}/100 risk "
            f"points, weight {w['sentiment']}")


def news_block(news: list[dict], limit: int = 8) -> str:
    if not news:
        return "NO_NEWS: there are no local headlines for this ticker."
    latest = sorted(news, key=lambda n: (n["date"], n["id"]), reverse=True)[:limit]
    return "\n".join(f"[{n['id']}] {n['date']} | {n['headline']} | {n.get('summary', '')} | FinBERT: {n['label']} "
                     f"(positive {n['positive']:.2f}, negative {n['negative']:.2f})" for n in latest)


# ── guardrails (deterministic, no LLM) ────────────────────────────────
def brief_guardrail(news_ids: set[str], min_cites: int) -> Callable:
    example = min(news_ids, default="news:ACME:1")

    def check(output):  # no return annotation (see HIGHLIGHTS)
        text = output.raw
        ids = cited_ids(text)
        bad = sorted({i for i in ids if i not in news_ids})
        if bad:
            return False, f"Unknown news ids {bad}. Cite only ids shown in the news list, like [{example}]."
        bullets = [ln for ln in text.splitlines() if ln.strip().startswith(("-", "*", "•")) or re.match(r"\s*\d+\.", ln)]
        if not bullets or len(set(ids)) < min_cites:
            return False, (f"Write 3-6 bullet points, each ending with a news id in square brackets like [{example}], "
                           f"using at least {min_cites} different ids.")
        return True, text
    return check


def memo_guardrail(news_ids: set[str], required: tuple[str, ...]) -> Callable:
    def check(output):  # no return annotation (see HIGHLIGHTS)
        text = output.raw
        missing = missing_sections(text, required)
        if missing:
            return False, ("Format the memo in Markdown with these sections: "
                           + ", ".join(f"'## {s}'" for s in required) + f". Missing: {missing}.")
        bad = sorted({i for i in cited_ids(text) if i not in news_ids})
        if bad:
            return False, f"Unknown news ids {bad}. Use only ids from the news brief."
        return True, text
    return check


# ── team ──────────────────────────────────────────────────────────────
def build_crew(mode: str = "grounded", llm: LLM | None = None, risk: dict | None = None,
               news: list[dict] = (), feedback: str = "") -> Crew:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cfg = get_config()
    llm = llm or _default_llm()
    ids = {n["id"] for n in news}
    min_cites = min(cfg.verify.min_citations, len(ids))
    facts, block = facts_text(risk), news_block(news, cfg.crew.news_items)
    common = dict(llm=llm, allow_delegation=False, max_iter=cfg.crew.max_iter, verbose=False)
    fix = f"\nA previous attempt was rejected for these reasons, fix them:\n{_esc(feedback)}\n" if feedback else ""
    sections = ", ".join(f"'## {s}'" for s in cfg.verify.required_sections)

    scout = Agent(role="News Scout",
                  goal="Surface the local financial news that matters for the risk of {ticker} ({company}), citing every headline id.",
                  backstory=("You are a financial news researcher. You only report what the local headlines say, "
                             "and you never invent headlines, dates or ids."),
                  **common)
    analyst = Agent(role="Risk Analyst",
                    goal="Explain the Risk Score of {ticker} ({company}) in a short investment memo.",
                    backstory=("You are a careful risk analyst. Code computes every number; you explain them. "
                               "You never calculate, round differently or invent a number."),
                    tools=risk_tools(risk["ticker"], facts, block) if mode == "tool" else [], **common)

    brief = Task(description=("Read the local news for {ticker} below. Each line has its id, date, headline, summary "
                              "and FinBERT sentiment. Write 3-6 bullet points about the news that matters most for the "
                              "stock's risk, newest first. End every bullet with its id in square brackets, e.g. "
                              f"[{min(ids, default='news:ACME:1')}]. Use only ids from this list.\n\nNEWS:\n{_esc(block)}\n"),
                 expected_output="3-6 bullet points, each ending with a news id like [news:ACME:3].",
                 agent=scout, guardrail=brief_guardrail(ids, min_cites), guardrail_max_retries=cfg.crew.guardrail_retries)
    how = ("First call get_risk_metrics and get_news_sentiment with the ticker {ticker} to get the computed numbers. "
           if mode == "tool" else f"Use ONLY these numbers:\n{_esc(facts)}\n\n")
    memo = Task(description=(how + f"Write an investment memo in Markdown with exactly these sections: {sections}. "
                             "In Risk Summary, state the Risk Score and its band. In Key Drivers, explain which metrics "
                             "push the score up or down, quoting each number exactly as given. In News Impact, use the "
                             "news brief and keep its ids in square brackets. Never compute, change or invent a number, "
                             "and do not put other numbers right after a metric name." + fix),
                expected_output=f"A short Markdown memo with the sections {sections}; numbers exactly as given.",
                agent=analyst, context=[brief], guardrail=memo_guardrail(ids, cfg.verify.required_sections),
                guardrail_max_retries=cfg.crew.guardrail_retries)
    return Crew(agents=[scout, analyst], tasks=[brief, memo], process=Process.sequential, memory=False, verbose=False)


def _esc(text: str) -> str:
    """CrewAI fills {placeholders} in task text; escape braces coming from headlines or feedback."""
    return text.replace("{", "(").replace("}", ")")


def _default_llm() -> LLM:
    from src.llm.factory import build_llm
    return build_llm()
