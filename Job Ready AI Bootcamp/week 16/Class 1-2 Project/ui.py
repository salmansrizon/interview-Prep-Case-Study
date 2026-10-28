"""
Thin Streamlit UI — talks to the FastAPI service over HTTP only.

Run:  uvicorn api.main:app --port 8000      (terminal 1)
      streamlit run ui.py                   (terminal 2; MKT_API_URL changes the API address)

HIGHLIGHTS: UI-তে কোনো business logic নেই — শুধু ``httpx`` দিয়ে API ডাকে আর ফল দেখায়। তাই একই API
CLI, notebook বা অন্য app-ও ব্যবহার করতে পারে, আর UI ভাঙলে Risk Score বদলায় না। Analysis কয়েক মিনিট
লাগে, তাই read timeout লম্বা; ফল ``st.session_state``-এ রাখা, কারণ download button চাপলে Streamlit
পুরো script আবার চালায়।
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.getenv("MKT_API_URL", "http://localhost:8000")
st.set_page_config(page_title="Market Intelligence Team", page_icon="📈", layout="wide")
client = httpx.Client(base_url=API, timeout=httpx.Timeout(10.0, read=900.0))


def call(method: str, path: str, **kw) -> httpx.Response:
    try:
        r = client.request(method, path, **kw)
    except httpx.HTTPError as e:
        st.error(f"API not reachable at {API} ({e}). Start it: uvicorn api.main:app --port 8000")
        st.stop()
    if r.is_error:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        st.error(f"{r.status_code} {body.get('error', '')}: {body.get('detail', r.text)}")
        st.stop()
    return r


with st.sidebar:
    try:
        ready = client.get("/ready").json()        # 503 is fine here: /v1/risk works without Ollama
    except httpx.HTTPError:
        ready = {}
    st.caption(f"API `{API}` · backend `{ready.get('backend')}` · {ready.get('status')}: {ready.get('detail')}")
    tickers = call("GET", "/v1/tickers").json()["tickers"]
    labels = {t["ticker"]: f"{t['ticker']} — {t['name']} ({t['source']})" for t in tickers}
    ticker = st.selectbox("Ticker", list(labels), format_func=labels.get)
    mode = st.radio("Agent mode", ("grounded", "tool"), format_func={
        "grounded": "grounded — numbers given in the task (default)",
        "tool": "tool — Risk Analyst must call tools (failure demo)"}.get)
    st.caption("Educational project, synthetic data, not investment advice.")

st.title("📈 Offline Market Intelligence Team")
st.caption("Risk Score (code) → News Scout Agent → Risk Analyst Agent → verify (code) → PDF memo or needs_human")

risk = call("GET", f"/v1/risk/{ticker}").json()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Risk Score", f"{risk['risk_score']:.1f} / 100", risk["band"])
c2.metric("Volatility (ann.)", f"{risk['metrics']['volatility'] * 100:.1f}%")
c3.metric("Max drawdown", f"{risk['metrics']['max_drawdown'] * 100:.1f}%")
c4.metric("VaR 95% 1-day", f"{risk['metrics']['var_95'] * 100:.1f}%")
c5.metric("Net sentiment", f"{risk['metrics']['sentiment']:+.2f}")
st.dataframe([{"metric": k, "risk points": v, "weight": risk["weights"][k]} for k, v in risk["components"].items()],
             hide_index=True)
with st.expander(f"News + FinBERT sentiment ({len(risk['news'])})"):
    st.dataframe(risk["news"], hide_index=True)

if st.button("▶ Run News Scout + Risk Analyst", type="primary"):
    with st.spinner("Agents working (1-3 minutes on llama3.1:8b)..."):
        st.session_state["result"] = call("POST", f"/v1/analyze/{ticker}", params={"mode": mode}).json()

res = st.session_state.get("result")
if res and res["ticker"] == ticker:
    if res["outcome"] == "published":
        st.success(f"Published after {res['rounds']} round(s), {res['seconds']}s")
        pdf = call("GET", f"/v1/memo/{ticker}").content
        st.download_button("Download PDF memo", pdf, file_name=f"{ticker}-memo.pdf", mime="application/pdf")
    else:
        st.error(f"Needs human review after {res['rounds']} round(s)")
    left, right = st.columns([3, 2])
    left.subheader("News brief (News Scout Agent)")
    left.markdown(res["brief"] or "_empty_")
    left.subheader("Memo (Risk Analyst Agent)")
    left.markdown(res["memo"] or "_empty_")
    right.subheader("Verification (code, not LLM)")
    right.json(res["verification"])
