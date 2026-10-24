"""
The research team: Researcher -> Writer -> Reviewer, plus a single-agent baseline.

HIGHLIGHTS:
* role/goal/backstory তিনটাই agent-এর system prompt-এ ঢোকে — specific লিখলে
  output-এর মান সরাসরি বাড়ে।
* Task ``context=[...]`` = dependency: আগের task-এর output পরের task-এর prompt-এ যায়।
* প্রতিটা task-এ **guardrail** — code-level check, fail করলে feedback সহ task আবার চলে
  (``guardrail_max_retries``)। Guardrail ``(True, output.raw)`` ফেরত দেয়
  (``(True, output)`` দিলে CrewAI 1.15-এ structured output হারায়)।
* ``allow_delegation=False``, ``max_iter`` ছোট — 8B model-এ delegation আর লম্বা
  loop দুটোই "Invalid Format" loop-এর বড় উৎস।
* **Grounded retrieval (default):** live পরীক্ষায় llama3.1:8b CrewAI-র ভেতরে
  search tool **একবারও ডাকেনি** — tool-call JSON text হিসেবে লিখে নিজেই "search
  result" আর ``[rag:15]`` ধরনের id বানিয়ে নিয়েছে। তাই default mode-এ retrieval
  হয় code-এ (BM25), passage-গুলো researcher-এর task-এ বসানো হয়; researcher
  বাছাই আর সংক্ষেপ করে। ``tool`` mode-এ researcher নিজে tool ডাকে — সেই failure
  দেখানোর জন্য রাখা পরীক্ষা।
* ``single`` mode: একই passage আর একই guardrail সহ এক agent — multi-agent আসলে
  কিছু যোগ করছে কিনা মাপার baseline (Cognition-এর "Don't Build Multi-Agents")।
"""
from __future__ import annotations

import re
from typing import Callable

import config  # noqa: F401 - telemetry env vars first
from crewai import LLM, Agent, Crew, Process, Task

from config import get_config
from src.tools.corpus_search import CorpusSearchTool, _cached_corpus
from src.verify.citations import cited_ids

MODES = ("sequential", "single", "tool", "hierarchical")

CITE_RULE = ("Cite every fact with its exact id in square brackets, e.g. [rag:2]. "
             "Use only ids that appear in the search results or notes. Never invent an id.")


# ── guardrails (deterministic, no LLM) ────────────────────────────────
def research_guardrail(corpus_dir: str) -> Callable:
    corpus = _cached_corpus(corpus_dir)

    def check(output):  # no return annotation: with `from __future__ import annotations` it becomes a string, which CrewAI rejects
        text = output.raw
        bullets = [l for l in text.splitlines() if l.strip().startswith(("-", "*", "•")) or re.match(r"\s*\d+\.", l)]
        ids = cited_ids(text)
        bad = sorted({i for i in ids if i not in corpus.by_id})
        if bad:
            return False, f"Unknown citation ids {bad}. Only cite ids shown by search_corpus, like [rag:2]."
        if len(bullets) < 4 or len(set(ids)) < 3:
            return False, ("Give at least 4 bullet points, each ending with a citation id like [rag:2], using at "
                           "least 3 different ids. Call search_corpus again if you need more facts.")
        return True, text
    return check


def report_guardrail(corpus_dir: str, notes_task: Task | None) -> Callable:
    corpus = _cached_corpus(corpus_dir)

    def check(output):  # no return annotation: with `from __future__ import annotations` it becomes a string, which CrewAI rejects
        text = output.raw
        ids = cited_ids(text)
        bad = sorted({i for i in ids if i not in corpus.by_id})
        if not re.search(r"^# \S", text, re.M) or len(re.findall(r"^## \S", text, re.M)) < 2:
            return False, "Format the report in Markdown: one '# Title' line and at least two '## Section' headings."
        if bad:
            return False, f"Unknown citation ids {bad}. Use only ids from the research notes."
        if len(ids) < get_config().verify.min_citations:
            return False, f"Cite at least {get_config().verify.min_citations} facts with ids like [rag:2]."
        if notes_task is not None and notes_task.output is not None:
            allowed = set(cited_ids(notes_task.output.raw))
            extra = sorted({i for i in ids if i not in allowed})
            if extra:
                return False, f"These ids are not in the research notes: {extra}. Use only facts from the notes."
        return True, text
    return check


# ── team ──────────────────────────────────────────────────────────────
def build_crew(mode: str = "sequential", llm: LLM | None = None, passages: str = "",
               reviewer_notes: str = "") -> Crew:
    """``passages`` = retrieved source text (grounded modes). In ``tool`` mode the researcher searches itself."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cfg = get_config()
    llm = llm or _default_llm()
    corpus_dir = str(cfg.data.corpus_dir)
    tool = CorpusSearchTool(corpus_dir=corpus_dir, top_k=cfg.crew.search_top_k, passage_chars=cfg.crew.passage_chars)
    common = dict(llm=llm, allow_delegation=False, max_iter=cfg.crew.max_iter, verbose=False)
    retries = cfg.crew.guardrail_retries
    fix = f"\nA previous attempt was rejected for these reasons, fix them:\n{_esc(reviewer_notes)}\n" if reviewer_notes else ""
    sources = f"\nSOURCE PASSAGES (the only facts you may use):\n{_esc(passages)}\n" if passages else ""
    search_tools = [tool] if mode == "tool" else []

    if mode == "single":
        solo = Agent(role="Research Analyst and Writer",
                     goal="Research {topic} in the local documents and write a short, fully cited report.",
                     backstory="You work alone and only state what the source passages say.",
                     **common)
        task = Task(description=("Write a Markdown report about: {topic}, using ONLY the source passages below. "
                                 "Use a '# Title', at least two '## ' sections and a short conclusion. "
                                 f"{CITE_RULE}{sources}{fix}"),
                    expected_output="A Markdown report where every factual sentence ends with a citation id.",
                    agent=solo, guardrail=report_guardrail(corpus_dir, None), guardrail_max_retries=retries)
        return Crew(agents=[solo], tasks=[task], process=Process.sequential, verbose=False)

    researcher = Agent(role="Senior Research Specialist",
                       goal="Collect accurate facts about {topic} from the local documents, each with its citation id.",
                       backstory=("You are meticulous. You search before you write, you copy facts faithfully, and "
                                  "you never add anything the documents do not say."),
                       tools=search_tools, **common)
    writer = Agent(role="Technical Report Writer",
                   goal="Turn the research notes into a clear, structured Markdown report about {topic}.",
                   backstory="You write for busy engineers: short sections, concrete facts, and a citation on every fact.",
                   **common)
    reviewer = Agent(role="Critical Fact-Checker",
                     goal="Make sure every claim in the report is supported by the research notes, then return the final report.",
                     backstory=("You are a strict editor. You delete or fix any sentence the notes do not support, "
                                "and you never invent facts or citation ids."),
                     **common)

    how = ("Use search_corpus 2-3 times with different keywords to find facts about: {topic}. "
           if mode == "tool" else "From the source passages below, pick the facts relevant to: {topic}. ")
    research = Task(description=(how + f"Write 5-8 bullet points, one fact each, in your own short words. {CITE_RULE}{sources}"),
                    expected_output="5-8 bullet points, each a single fact ending with a citation id like [rag:2].",
                    agent=researcher, guardrail=research_guardrail(corpus_dir), guardrail_max_retries=retries)
    write = Task(description=("Using ONLY the research notes, write a Markdown report about {topic} with a '# Title', "
                              f"at least two '## ' sections and a short conclusion. Keep each fact's citation id.{fix}"),
                 expected_output="A Markdown report; every factual sentence ends with a citation id from the notes.",
                 agent=writer, context=[research], guardrail_max_retries=retries)
    write.guardrail = report_guardrail(corpus_dir, research)
    review = Task(description=("Check every sentence of the draft against the research notes. Remove or correct "
                               "anything the notes do not support. Keep the Markdown structure and the citation ids. "
                               "Return ONLY the final report - no notes or comments about your edits."),
                  expected_output="The final Markdown report, every factual sentence cited.",
                  agent=reviewer, context=[research, write], guardrail_max_retries=retries)
    review.guardrail = report_guardrail(corpus_dir, research)

    if mode == "hierarchical":
        # A manager LLM assigns the tasks. Small local models are weak managers (they often fail to call
        # the delegation tool) — kept as an experiment; sequential is the reliable default.
        return Crew(agents=[researcher, writer, reviewer], tasks=[research, write, review],
                    process=Process.hierarchical, manager_llm=llm, verbose=False)
    return Crew(agents=[researcher, writer, reviewer], tasks=[research, write, review],
                process=Process.sequential, verbose=False)


def _esc(text: str) -> str:
    """CrewAI fills {placeholders} in task text; escape braces coming from documents or feedback."""
    return text.replace("{", "(").replace("}", ")")


def _default_llm() -> LLM:
    from src.llm.factory import build_llm
    return build_llm()
