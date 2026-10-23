"""
Streamlit chat UI for the ReAct Travel Agent (Week 13, Class 2).

HIGHLIGHTS:
* Conversational — আগের user/assistant turn ``history`` হিসেবে agent-এ যায়,
  তাই "ওটাই বুক করো" ধরনের follow-up কাজ করে।
* Human-in-the-loop, code-এ: "Require my approval" চালু থাকলে book_flight-এর
  approval gate booking আটকে ``pending`` হিসেবে রাখে। Model কিছু বুক করতে
  পারে না — **আপনি** Approve বাটন চাপলে code নিজে বুক করে।
* প্রতিটা উত্তরের নিচে Thought → Action → Observation trace আর typed stop
  status, যাতে loop-এর ভেতরটা দেখা যায়।
* এই file কখনো ``ollama`` import করে না — সব কাজ ``src/``-এ।
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from config import get_config
from src.agent.react_agent import run_agent
from src.llm.client import chat
from src.tools.travel import TravelTools

cfg = get_config()
st.set_page_config(page_title=cfg.app.page_title, page_icon=cfg.app.page_icon, layout="wide")

ss = st.session_state
if "tools" not in ss:
    ss.tools = TravelTools(cfg.data.flights_path, max_results=cfg.agent.max_results)
    ss.chat = []          # [{"role", "content", "trace"?, "status"?}]
    ss.pending = None     # booking details waiting for the human

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", list(dict.fromkeys([cfg.llm.model, *cfg.llm.compare_models, "qwen3:8b"])))
    mode = st.radio("ReAct mode", ["native", "text"], horizontal=True,
                    help="native = Ollama tool_calls; text = classic Thought/Action/Observation parsing")
    require_approval = st.toggle("Require my approval before booking", value=True)
    if st.button("Reset conversation & bookings"):
        ss.tools.reset()
        ss.chat, ss.pending = [], None
        st.rerun()
    st.subheader("Bookings")
    st.dataframe(pd.DataFrame(ss.tools.bookings or [{"booking_ref": "—"}]), hide_index=True, width="stretch")
    with st.expander("Flight database"):
        st.dataframe(pd.DataFrame(ss.tools.flights), hide_index=True, width="stretch")

st.title(f"{cfg.app.page_icon} ReAct Travel Agent")
st.caption("Local llama3.1:8b on Ollama · deterministic tools · typed stop reasons · booking needs your approval")


def show_trace(trace: list[dict], status: str) -> None:
    with st.expander(f"Trace · status = {status} · {len(trace)} action(s)"):
        if trace:
            st.dataframe(pd.DataFrame(trace), hide_index=True, width="stretch")
        else:
            st.write("No tool was called.")


for m in ss.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("trace") is not None:
            show_trace(m["trace"], m["status"])

# ── pending human approval ────────────────────────────────────────────
if ss.pending:
    st.warning(f"Approve booking: **{ss.pending['summary']}**?")
    c1, c2 = st.columns(2)
    if c1.button("✅ Approve", type="primary"):
        d, ss.pending = ss.pending, None
        ss.tools.approve = lambda details: True      # the human said yes — code books, not the model
        obs = ss.tools.book_flight(d["flight_id"], d["passenger"])
        msg = (f"Booked ✅ **{obs['booking_ref']}** — {d['summary']}" if obs.get("ok")
               else f"Could not book: {obs.get('error')} ({obs.get('hint', '')})")
        ss.chat.append({"role": "assistant", "content": msg})
        st.rerun()
    if c2.button("❌ Reject"):
        ss.chat.append({"role": "assistant", "content": f"Okay — nothing booked ({ss.pending['summary']})."})
        ss.pending = None
        st.rerun()

prompt = st.chat_input("Ask about flights, e.g. " + cfg.app.examples[1])
if prompt:
    ss.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    def approve(details: dict) -> bool:
        if not require_approval:
            return True
        ss.pending = details        # hold it for the human; tell the model it was not booked
        return False

    ss.tools.approve = approve
    history = [{"role": m["role"], "content": m["content"]} for m in ss.chat[:-1]][-8:]
    a = cfg.agent
    with st.chat_message("assistant"), st.spinner(f"{model} · {mode} ReAct loop running..."):
        res = run_agent(prompt, ss.tools, chat, history=history, mode=mode, model=model,
                        max_steps=a.max_steps, max_repeat_calls=a.max_repeat_calls,
                        max_consecutive_errors=a.max_consecutive_errors, deadline_seconds=a.deadline_seconds,
                        max_llm_retries=a.max_llm_retries, nudge_after_tools=a.nudge_after_tools)
    answer = res.answer if res.status != "llm_failed" else f"⚠️ {res.answer}"
    trace = [{"step": s.n, "thought": s.thought[:120], "action": s.tool, "args": json.dumps(s.args, default=str),
              "observation": json.dumps(s.observation, ensure_ascii=False)[:200], "prompt_tokens": s.prompt_tokens}
             for s in res.steps]
    ss.chat.append({"role": "assistant", "content": answer, "trace": trace, "status": res.status})
    st.rerun()
