"""Offline tests: the LLM is replaced by a scripted fake chat function."""

import sys

import pytest

from src.agent.loop import run_agent
from src.tools.calculator import calculate
from src.tools.filesystem import list_files, read_file
from src.tools.registry import run_tool


def test_calculator():
    assert calculate("(12+8)*3") == "60"
    assert calculate("sqrt(16)") == "4.0"
    assert calculate("1/0").startswith("Error")
    assert calculate("__import__('os')").startswith("Error")
    assert calculate("9**9**9").startswith("Error")


def test_sandbox(tmp_path):
    box = tmp_path / "box"
    box.mkdir()
    (box / "a.txt").write_text("hi")
    (tmp_path / "secret.txt").write_text("nope")
    assert list_files(box) == "a.txt"
    assert read_file(box, "a.txt") == "hi"
    assert "escapes" in read_file(box, "../secret.txt")
    assert "escapes" in read_file(box, str(tmp_path / "secret.txt"))
    assert "escapes" in list_files(box, "..")


def test_run_tool_errors(tmp_path):
    assert "unknown tool" in run_tool("rm_rf", {}, tmp_path)
    assert "bad arguments" in run_tool("read_file", {}, tmp_path)


def _script(*replies):
    it = iter(replies)
    return lambda messages, tools, model=None: next(it)


def test_agent_uses_tool_then_answers(tmp_path):
    fake = _script(
        {"content": "", "tool_calls": [{"name": "calculate", "arguments": {"expression": "6*7"}}]},
        {"content": "42", "tool_calls": []},
    )
    answer, trace = run_agent("6*7?", sandbox=tmp_path, chat_fn=fake)
    assert answer == "42" and trace[0]["result"] == "42"


def test_agent_stops_at_max_steps(tmp_path):
    def forever(m, t, model=None):
        return {"content": "", "tool_calls": [{"name": "list_files", "arguments": {}}]}

    answer, trace = run_agent("x", sandbox=tmp_path, max_steps=3, chat_fn=forever)
    assert answer.startswith("Stopped") and len(trace) == 3


def test_client_wraps_failures(monkeypatch):
    from src.llm import client

    monkeypatch.setitem(sys.modules, "ollama", None)  # import ollama -> ImportError
    with pytest.raises(client.OllamaConnectionError):
        client.chat([], [])
