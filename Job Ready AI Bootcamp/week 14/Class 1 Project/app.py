"""
Streamlit UI — Self-Correcting LLM Coder (Week 14, Class 1).

HIGHLIGHTS:
* Graph চলে ``stream(stream_mode="updates")`` দিয়ে — প্রতিটা node শেষ হলেই
  UI-তে দেখা যায় (generate → static_check → run_visible → ...)।
* Approval = LangGraph ``interrupt()``। Graph থেমে SQLite-এ checkpoint হয়;
  Approve/Reject বাটন ``Command(resume=...)`` দিয়ে **একই thread_id**-তে
  আবার চালায়। Browser refresh হলেও state checkpoint-এ থাকে।
* "Time travel" expander: ``get_state_history`` — প্রতিটা checkpoint।
* এই file ``ollama`` import করে না; সব কাজ ``src/``-এ।
"""
from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from config import get_config
from src.graph.coder import build_graph, initial_state, run_config, sqlite_checkpointer
from src.llm.client import OllamaUnavailable, generate
from src.tasks.bank import TASKS, custom_task

cfg = get_config()
st.set_page_config(page_title=cfg.app.page_title, page_icon=cfg.app.page_icon, layout="wide")
ss = st.session_state


@st.cache_resource
def checkpointer():
    return sqlite_checkpointer()


with st.sidebar:
    st.markdown(f"**Model:** `{cfg.llm.model}`")
    max_attempts = st.slider("Retry budget (max attempts)", 1, 6, cfg.agent.max_attempts)
    require_approval = st.toggle("Require my approval at the end", value=True)
    choice = st.selectbox("Task", [*TASKS, "custom"], format_func=lambda k: k if k == "custom" else f"{k} — {TASKS[k].func_name}")

st.title(f"{cfg.app.page_icon} Self-Correcting LLM Coder")
st.caption("LangGraph loop: write code → static check → run tests in a subprocess → feed failures back → retry → hidden tests → your approval")

if choice == "custom":
    func = st.text_input("Function name", "double")
    prompt = st.text_area("Task", "Write double(x: int) -> int that returns x multiplied by 2.")
    raw = st.text_area("Tests — one per line: `call -> expected`", "double(2) -> 4\ndouble(-3) -> -6")
    tests = [tuple(p.strip() for p in line.split("->", 1)) for line in raw.splitlines() if "->" in line]
    task = custom_task(func, prompt, tests)
else:
    task = TASKS[choice]
    st.markdown(f"**Task:** {task.prompt}")
st.dataframe(pd.DataFrame(task.visible, columns=["visible test call", "expected"]), hide_index=True, width="stretch")
if task.hidden:
    st.caption(f"Plus {len(task.hidden)} hidden tests the model never sees (they run only after the visible ones pass).")


def graph_for(params: dict):
    return build_graph(generate, max_attempts=params["max_attempts"], require_approval=params["approval"],
                       checkpointer=checkpointer())


def drive(payload) -> None:
    """Stream the graph until it ends or pauses at the approval interrupt."""
    graph = graph_for(ss.params)
    log = st.status("Running the self-correction loop...", expanded=True)
    ss.pending = None
    try:
        for chunk in graph.stream(payload, run_config(ss.thread_id), stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    ss.pending = update[0].value
                    continue
                for line in (update or {}).get("history", []):
                    log.write(f"`{node}` — {line}")
                if node == "generate":
                    log.code(update["code"], language="python")
    except OllamaUnavailable as e:
        log.update(label="Ollama unavailable", state="error")
        st.error(str(e))
        return
    log.update(label="Paused for your approval" if ss.pending else "Finished", state="complete", expanded=False)


if st.button("▶ Run self-correcting loop", type="primary"):
    ss.thread_id = f"ui-{uuid.uuid4().hex[:10]}"
    ss.params = {"max_attempts": max_attempts, "approval": require_approval}
    drive(initial_state(task))

if ss.get("pending"):
    p = ss.pending
    st.warning(f"Approve this code? visible tests {p['visible']}, hidden tests {p['hidden']}")
    st.code(p["code"], language="python")
    c1, c2 = st.columns(2)
    if c1.button("✅ Approve"):
        drive(Command(resume=True))
        st.rerun()
    if c2.button("❌ Reject"):
        drive(Command(resume=False))
        st.rerun()

if ss.get("thread_id") and not ss.get("pending"):
    snap = graph_for(ss.params).get_state(run_config(ss.thread_id))
    v = snap.values
    if v.get("status"):
        ok = v["status"] == "done"
        (st.success if ok else st.error)(
            f"status = **{v['status']}** after {v.get('attempts')} attempt(s) · visible "
            f"{v.get('visible_passed', 0)}/{v.get('visible_total', 0)} · hidden {v.get('hidden_passed', 0)}/{v.get('hidden_total', 0)}")
        st.code(v.get("code", ""), language="python")
        with st.expander("History"):
            st.write("\n".join(f"- {h}" for h in v.get("history", [])))
        with st.expander("Time travel — every checkpoint of this run (SQLite)"):
            rows = [{"step": s.metadata.get("step"), "next": ", ".join(s.next) or "END",
                     "attempts": s.values.get("attempts"), "status": s.values.get("status", "")}
                    for s in graph_for(ss.params).get_state_history(run_config(ss.thread_id))]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
