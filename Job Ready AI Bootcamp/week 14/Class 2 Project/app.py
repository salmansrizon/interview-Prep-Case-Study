"""
Streamlit UI — Collaborative Multi-Agent Research Team (Week 14, Class 2).

HIGHLIGHTS: এই file শুধু ``run_research()`` ডাকে আর ফলাফল দেখায়। দেখার জিনিস:
(১) কোন passage code retrieve করল, (২) প্রতিটা agent-এর output (task callback),
(৩) code-এর citation verification — কোন বাক্য supported, কোনটা weak/uncited।
Multi-agent বনাম single-agent পাশাপাশি চালিয়ে তুলনা করা যায়।
"""
from __future__ import annotations

import config  # noqa: F401 - telemetry env vars before crewai
import pandas as pd
import streamlit as st

from config import get_config
from src.crew.team import MODES
from src.flow.research_flow import run_research
from src.llm.factory import ollama_status
from src.tools.corpus_search import _cached_corpus
from src.verify.citations import verify

cfg = get_config()
st.set_page_config(page_title=cfg.app.page_title, page_icon=cfg.app.page_icon, layout="wide")
corpus = _cached_corpus(str(cfg.data.corpus_dir))

with st.sidebar:
    st.markdown(f"**Model:** `{cfg.llm.model}` (Ollama, local)")
    mode = st.radio("Team", MODES, format_func={
        "sequential": "Researcher → Writer → Reviewer (grounded)",
        "single": "Single agent baseline (grounded)",
        "tool": "Researcher uses the search tool itself (experiment)",
        "hierarchical": "Hierarchical manager (experiment)"}.get)
    st.caption("Telemetry off · memory off · retrieval over the local corpus only")
    with st.expander(f"Corpus ({len(corpus.passages)} passages)"):
        st.dataframe(pd.DataFrame([{"id": p.id, "section": p.title} for p in corpus.passages]),
                     hide_index=True, width="stretch")

st.title(f"{cfg.app.page_icon} Multi-Agent Research Team")
st.caption("CrewAI Flow: retrieve (code) → research → write → review (agents) → verify citations (code) → publish or escalate")

preset = st.selectbox("Pick a topic", cfg.app.topics)
topic = st.text_input("...or type your own", value=preset)

if st.button("▶ Run research team", type="primary", disabled=not topic.strip()):
    ok, msg = ollama_status()
    if not ok:
        st.error(msg)
        st.stop()
    log = st.status(f"{mode} team working on: {topic}", expanded=True)
    outputs = []

    # CrewAI Flows run steps in worker threads; Streamlit calls only work in the script thread,
    # so the callback just collects outputs and the page renders them afterwards.
    state = run_research(topic, mode, on_task=outputs.append)
    for out in outputs:
        log.write(f"✓ **{out.agent}** finished ({len(out.raw)} chars)")
    log.update(label=f"Done in {state.seconds}s · {state.rounds} round(s)", state="complete", expanded=False)

    s = state.verification
    if state.outcome == "published":
        st.success(f"Published · citation precision {s.get('citation_precision')} · "
                   f"{s.get('supported')}/{s.get('cited')} cited sentences supported")
    else:
        st.error(f"Needs human review after {state.rounds} round(s): " + (s.get("error") or
                 f"{s.get('supported')}/{s.get('cited')} cited sentences supported, {s.get('uncited')} of "
                 f"{s.get('sentences')} sentences uncited, invalid ids {s.get('invalid_ids')}"))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Report")
        st.markdown(state.report or "_no report produced_")
        st.download_button("Download report.md", state.report or "", file_name="report.md")
    with right:
        st.subheader("Verification (code, not LLM)")
        st.json({k: v for k, v in s.items()}, expanded=False)
        if state.report:
            v = verify(state.report, corpus, notes=state.notes or None, min_support=cfg.verify.min_support)
            st.dataframe(pd.DataFrame([{"verdict": c.verdict, "support": c.support, "ids": ", ".join(c.ids),
                                        "sentence": c.sentence[:120]} for c in v.checks]),
                         hide_index=True, width="stretch")
        if state.passages:
            st.caption("Retrieved by code (BM25): " + ", ".join(state.passages))
        if state.token_usage:
            st.caption(f"Token usage: {state.token_usage}")
    with st.expander("Each agent's output"):
        for o in outputs:
            st.markdown(f"**{o.agent}**")
            st.code(o.raw[:3000], language="markdown")
