"""The agent loop: model -> tool calls -> results -> model ... -> answer."""

from __future__ import annotations

from pathlib import Path

from config import AGENT
from src.llm.client import chat
from src.tools.registry import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = (
    "You are a helpful assistant with tools. Use `calculate` for any math and "
    "`list_files`/`read_file` for files in the sandbox. Never guess numbers or "
    "file contents. When you have the answer, reply briefly in plain text."
)


def run_agent(question: str, sandbox: Path = AGENT.sandbox_dir,
              max_steps: int = AGENT.max_steps, model: str | None = None,
              chat_fn=chat) -> tuple[str, list[dict]]:
    """Return (final_answer, trace). trace = list of {tool, args, result}.

    HIGHLIGHTS: loop-এর শেষ হওয়ার দুটো শর্ত — (১) model আর কোনো tool call
    করেনি (মানে উত্তর তৈরি), (২) max_steps শেষ। দ্বিতীয়টা না থাকলে ছোট model
    একই tool বারবার ডেকে চিরকাল ঘুরতে পারে। tool result ``role="tool"``
    message হিসেবে history-তে ফেরত যায়, তাই model পরের turn-এ সেটা দেখে।
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    trace: list[dict] = []
    for _ in range(max_steps):
        reply = chat_fn(messages, TOOL_SCHEMAS, model=model)
        if not reply["tool_calls"]:
            return reply["content"], trace
        messages.append({
            "role": "assistant", "content": reply["content"],
            "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                           for c in reply["tool_calls"]],
        })
        for call in reply["tool_calls"]:
            result = run_tool(call["name"], call["arguments"], sandbox, AGENT.max_file_chars)
            trace.append({"tool": call["name"], "args": call["arguments"], "result": result})
            messages.append({"role": "tool", "content": result, "tool_name": call["name"]})
    return "Stopped: max_steps reached without a final answer.", trace
