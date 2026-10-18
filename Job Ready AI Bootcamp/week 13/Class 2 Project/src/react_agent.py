"""
Hand-written ReAct loop: Thought -> Action -> Observation -> ... -> Final Answer.

HIGHLIGHTS: কোনো framework নেই — পুরো agent একটা while-loop। প্রতিটা step-এ
model একটা JSON দেয়: {"thought", "action", "action_input"} অথবা
{"thought", "final_answer"}। আমরা action চালিয়ে ফলাফল "Observation:" হিসেবে
conversation-এ ফেরত দিই। চারটা সুরক্ষা:
  ১. max_steps      — অসীম loop বন্ধ।
  ২. retry          — LLM call fail হলে সীমিতবার আবার চেষ্টা; তারপর হাল ছেড়ে দেওয়া।
  ৩. error as data  — ভাঙা JSON বা tool error exception হয়ে crash করে না,
                      Observation হয়ে model-কে শুধরানোর সুযোগ দেয়।
  ৪. loop detection — একই action+input বারবার হলে থামিয়ে দেওয়া।
Model বাছাইয়ের reference table: config.py (llama3.1:8b vs mistral:7b vs llama3.2:3b)।
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from src.tools import TOOL_DOC, ToolError, TravelTools

SYSTEM_PROMPT = f"""You are a travel agent. Solve the task step by step using tools.
Tools:
{TOOL_DOC}
Reply with EXACTLY ONE JSON object per turn, nothing else.
To use a tool: {{"thought": "...", "action": "<tool name>", "action_input": {{...}}}}
When finished:  {{"thought": "...", "final_answer": "..."}}
After each action you will receive an Observation. Never invent flights or booking refs;
only book after searching. If a tool errors, read the message and fix your next action."""


@dataclass
class Step:
    n: int
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class AgentResult:
    status: str            # "done" | "max_steps" | "loop_detected" | "llm_failed"
    answer: str
    steps: list[Step]


def parse_step(text: str) -> dict:
    """Extract the JSON object from model text (tolerates ```json fences / extra prose)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    d = json.loads(m.group(0))
    if not isinstance(d, dict) or not ("final_answer" in d or "action" in d):
        raise ValueError("JSON needs 'action' or 'final_answer'")
    return d


def run_agent(task: str, tools: TravelTools, chat: Callable[[list[dict]], str],
              max_steps: int = 8, max_llm_retries: int = 2, max_repeat_actions: int = 2,
              on_step: Callable[[Step], None] | None = None) -> AgentResult:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    steps: list[Step] = []
    seen: dict[str, int] = {}
    retries = 0
    n = 0
    while n < max_steps:
        try:
            raw = chat(messages)
        except Exception as e:  # LLM transport failure -> bounded retry, no step consumed
            retries += 1
            if retries > max_llm_retries:
                return AgentResult("llm_failed", f"LLM unavailable: {e}", steps)
            continue
        n += 1
        step = Step(n)
        try:
            d = parse_step(raw)
        except ValueError as e:  # malformed output (JSONDecodeError is a ValueError): tell the model
            step.observation = f"ERROR: invalid reply ({e}). Reply with one JSON object only."
            _record(step, steps, on_step, messages, raw)
            continue
        step.thought = str(d.get("thought", ""))
        if "final_answer" in d and "action" not in d:
            step.observation = "(final)"
            _emit(step, steps, on_step)
            return AgentResult("done", str(d["final_answer"]), steps)
        step.action, step.action_input = str(d["action"]), d.get("action_input") or {}
        key = step.action + json.dumps(step.action_input, sort_keys=True, default=str)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > max_repeat_actions:
            step.observation = "ERROR: repeated identical action; stopping."
            _emit(step, steps, on_step)
            return AgentResult("loop_detected", "Stopped: agent kept repeating the same action.", steps)
        try:
            step.observation = json.dumps(tools.call(step.action, step.action_input), default=str)
        except ToolError as e:
            step.observation = f"ERROR: {e}"
        _record(step, steps, on_step, messages, raw)
    return AgentResult("max_steps", f"Stopped: no final answer within {max_steps} steps.", steps)


def _emit(step, steps, on_step):
    steps.append(step)
    if on_step:
        on_step(step)


def _record(step, steps, on_step, messages, raw):
    _emit(step, steps, on_step)
    messages.append({"role": "assistant", "content": raw})
    messages.append({"role": "user", "content": f"Observation: {step.observation}"})
