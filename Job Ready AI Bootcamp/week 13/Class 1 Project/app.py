"""
Streamlit UI for the Tool-Calling Agent (Week 13, Class 1).

HIGHLIGHTS: এই file কখনো ``ollama`` import করে না, tool-ও নিজে চালায় না —
শুধু ``src.agent.loop.run_agent()`` ডাকে আর ফলাফল দেখায়। মূল দেখার জিনিস
হলো trace: model কোন tool কোন argument দিয়ে ডাকল, আর কী ফেরত পেল। এটাই
tool calling-এর "magic" খুলে দেখায় — model নিজে কিছু চালায় না।
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import CFG
from src.agent.loop import run_agent
from src.llm.client import OllamaConnectionError
from src.tools.filesystem import list_files
from src.tools.registry import TOOL_SCHEMAS

st.set_page_config(page_title=CFG.app.page_title, page_icon=CFG.app.page_icon, layout="wide")
st.title(f"{CFG.app.page_icon} Math & File Inspector")
st.caption("Local tool-calling agent on Ollama. The model only *requests* tools; Python runs them.")

with st.sidebar:
    model = CFG.llm.model
    st.markdown(f"**Model:** `{model}`")
    max_steps = st.slider("max_steps", 1, 10, CFG.agent.max_steps)
    st.markdown("**Sandbox files**")
    st.code(list_files(CFG.agent.sandbox_dir), language="text")
    with st.expander("Tool schemas the model sees"):
        st.json(TOOL_SCHEMAS)

example = st.selectbox("Try an example", ("",) + CFG.app.examples)
question = st.text_input("Ask something", value=example)

if st.button("Run agent", type="primary", disabled=not question.strip()):
    with st.spinner(f"{model} thinking..."):
        try:
            res = run_agent(question, model=model, max_steps=max_steps)
        except OllamaConnectionError as exc:
            st.error(str(exc))
            st.stop()
    (st.success if res.status == "answered" else st.warning)(res.answer)
    st.caption(f"status={res.status} · LLM calls={res.llm_calls} · tool calls={len(res.steps)}")
    if res.steps:
        st.subheader("Trace")
        st.dataframe(pd.DataFrame([
            {"step": s.step, "tool": s.tool, "args": str(s.args), "result": s.result[:200], "ms": round(s.ms, 2)}
            for s in res.steps
        ]), width="stretch", hide_index=True)
    else:
        st.info("The model answered without calling any tool.")
