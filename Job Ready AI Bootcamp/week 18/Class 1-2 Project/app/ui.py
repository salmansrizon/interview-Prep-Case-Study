"""Streamlit UI — a thin client. It talks to the FastAPI service over HTTP only (HC_API_URL).

  uvicorn src.api.main:app --port 8000      # terminal 1 (HC_BACKEND=fake for a no-Ollama demo)
  streamlit run ui.py                       # terminal 2  ->  http://localhost:8501

HIGHLIGHTS: UI-তে কোনো model, Chroma বা guardrail নেই — শুধু ``streamlit`` + ``httpx``। সব সিদ্ধান্ত API-তে,
তাই Week 18-এ UI আর API আলাদা container-এ যায়, আর UI-কে ``HC_API_URL=http://api:8000`` বললেই চলে।
Route রং দিয়ে দেখানো হয়: answered (সবুজ), abstained (ধূসর), refused (কমলা), escalated (লাল)।
Streamlit-এর usage statistics ``.streamlit/config.toml``-এ বন্ধ।
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("HC_API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "How do I prepare an ORS packet?",
    "What are the clinic opening hours?",
    "My name is Rahim, phone 01712-345678. How do I request a copy of my records?",
    "How many mg of paracetamol should I take?",
    "My father has chest pain and is sweating",
    "What is the capital of France?",
]
ROUTE_STYLE = {"answered": st.success, "abstained": st.info, "refused": st.warning, "escalated": st.error}

st.set_page_config(page_title="Greenfield Clinic Assistant", page_icon="🩺", layout="wide")

with st.sidebar:
    st.markdown(f"**API:** `{API_URL}`")
    try:
        ready = httpx.get(f"{API_URL}/ready", timeout=5).json()
        (st.success if ready.get("status") == "ready" else st.warning)(
            f"{ready.get('status')} · backend `{ready.get('backend')}` · {ready.get('indexed_sections')} sections\n\n"
            f"LLM: {ready.get('llm')}")
        pages = httpx.get(f"{API_URL}/v1/documents", timeout=5).json()["pages"]
        with st.expander(f"Handbook ({len(pages)} pages)"):
            st.dataframe(pages, hide_index=True, width="stretch")
    except httpx.HTTPError as exc:
        st.error(f"API not reachable: {type(exc).__name__}. Start it with: uvicorn src.api.main:app --port 8000")
    st.caption("Local only: Ollama + ChromaDB on this machine, telemetry off, PII redacted before logging.")

st.title("🩺 Greenfield Community Clinic Assistant")
st.caption("Answers only from the clinic handbook. Educational, fictional clinic handbook; not medical advice. "
           "In an emergency in Bangladesh, call 999.")

preset = st.selectbox("Try an example", EXAMPLES)
question = st.text_area("...or type your question", value=preset, max_chars=1000)

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        r = httpx.post(f"{API_URL}/v1/ask", json={"question": question}, timeout=180)
    except httpx.HTTPError as exc:
        st.error(f"API not reachable: {type(exc).__name__}")
        st.stop()
    body = r.json()
    if r.status_code != 200:
        st.error(f"{r.status_code} {body.get('error')}: {body.get('detail')}")
        st.stop()
    ROUTE_STYLE[body["route"]](f"**{body['route'].upper()}**" + (f" · rule `{body['rule']}`" if body["rule"] else ""))
    st.markdown(body["answer"])
    st.caption(body["disclaimer"])
    if body["citations"]:
        st.subheader("Sources" if body["route"] == "answered" else "General handbook information")
        for c in body["citations"]:
            with st.expander(f"[{c['n']}] {c['page']} - {c['section']}  ({c['id']}, distance {c['distance']})"):
                st.write(c["snippet"])
    left, right = st.columns(2)
    with left:
        st.markdown(f"**PII redacted:** {body['redaction_count']} {body['redactions'] or ''}")
        st.code(body["redacted_question"], language=None)
    with right:
        st.markdown("**Timings (ms)**")
        st.json(body["timings_ms"], expanded=False)
        if body["flags"]:
            st.caption(f"flags: {body['flags']}")
