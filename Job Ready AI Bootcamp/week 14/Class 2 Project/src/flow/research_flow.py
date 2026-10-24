"""
A CrewAI Flow around the crew: run -> verify in code -> (retry once) -> publish | needs_human.

HIGHLIGHTS: Crew = agent-রা নিজেরা কাজ ভাগ করে (autonomy)। Flow = আমরা code-এ
ক্রম ঠিক করি (control) — @start, @listen, @router। ভাগাভাগি: retrieval আর
"পাস করল কিনা / থামবে কখন" — deterministic code; বাছাই, লেখা, সম্পাদনা — agent।
Retry-র hard cap আছে (MAST failure mode: "not knowing when to stop")। Guardrail
retry শেষ হলে CrewAI exception ছোড়ে — Flow সেটা ধরে round-টাকে fail গণ্য করে।
"""
from __future__ import annotations

import time

import config  # noqa: F401 - telemetry env vars first
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from config import get_config
from src.crew.team import build_crew
from src.tools.corpus_search import _cached_corpus
from src.verify.citations import verify


class ResearchState(BaseModel):
    topic: str = ""
    mode: str = "sequential"
    passages: list[str] = Field(default_factory=list)   # retrieved passage ids
    notes: str = ""
    report: str = ""
    verification: dict = Field(default_factory=dict)
    feedback: str = ""
    rounds: int = 0
    seconds: float = 0.0
    token_usage: dict = Field(default_factory=dict)
    outcome: str = ""            # published | needs_human


class ResearchFlow(Flow[ResearchState]):
    llm = None                   # injected (tests pass a fake BaseLLM)
    on_task = None               # optional callback(task_output) for the UI

    @start()
    def retrieve(self):
        """Deterministic retrieval in code (BM25) — skipped in 'tool' mode, where the agent searches itself."""
        if self.state.mode == "tool":
            return ""
        cfg = get_config()
        hits = _cached_corpus(str(cfg.data.corpus_dir)).search(self.state.topic, k=cfg.crew.retrieve_k)
        self.state.passages = [p.id for p, _ in hits]
        return "\n\n".join(f"[{p.id}] ({p.title}) {p.text[:cfg.crew.passage_chars]}" for p, _ in hits)

    @listen(retrieve)
    def run_team(self, passages: str):
        cfg = get_config()
        corpus = _cached_corpus(str(cfg.data.corpus_dir))
        t0 = time.perf_counter()
        for self.state.rounds in range(1, cfg.crew.flow_max_rounds + 1):
            crew = build_crew(self.state.mode, llm=self.llm, passages=passages, reviewer_notes=self.state.feedback)
            if self.on_task:
                crew.task_callback = self.on_task
            try:
                out = crew.kickoff(inputs={"topic": self.state.topic})
            except Exception as e:  # noqa: BLE001 - guardrail retries exhausted, malformed agent output, ...
                self.state.verification = {"error": f"{type(e).__name__}: {e}"[:300]}
                self.state.feedback = f"The previous attempt failed: {str(e)[:300]}"
                continue
            self.state.report = out.raw
            self.state.notes = out.tasks_output[0].raw if self.state.mode != "single" else ""
            self.state.token_usage = _usage(out)
            v = verify(self.state.report, corpus, notes=self.state.notes or None, min_support=cfg.verify.min_support)
            self.state.verification = v.summary()
            if self._passes():
                break
            self.state.feedback = v.feedback() or "Too many sentences are uncited or unsupported; cite every fact."
        self.state.seconds = round(time.perf_counter() - t0, 1)

    @router(run_team)
    def gate(self):
        return "ok" if self._passes() else "escalate"

    def _passes(self) -> bool:
        s, cfg = self.state.verification, get_config().verify
        if "error" in s or not self.state.report:
            return False
        return (not s.get("invalid_ids") and not s.get("not_in_notes") and s.get("cited", 0) >= cfg.min_citations
                and s.get("weak", 0) <= cfg.max_weak_ratio * s.get("cited", 0)
                and s.get("uncited", 0) <= cfg.max_uncited_ratio * max(s.get("sentences", 1), 1))

    @listen("ok")
    def publish(self):
        self.state.outcome = "published"
        return self.state.report

    @listen("escalate")
    def needs_human(self):
        self.state.outcome = "needs_human"
        return self.state.report


def _usage(out) -> dict:
    u = getattr(out, "token_usage", None)
    if u is None:
        return {}
    data = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    return {k: v for k, v in data.items() if isinstance(v, (int, float))}


def run_research(topic: str, mode: str = "sequential", llm=None, on_task=None) -> ResearchState:
    flow = ResearchFlow()
    flow.llm, flow.on_task = llm, on_task
    flow.kickoff(inputs={"topic": topic, "mode": mode})
    return flow.state
