"""
Offline tests for the Tool-Calling Agent. Run: pytest tests -q

HIGHLIGHTS: কোনো test real Ollama ছোঁয় না (ADR 0006)। Agent loop-এ
``chat_fn`` হিসেবে scripted fake function দেওয়া হয়, আর client test-এ
``sys.modules['ollama']``-এ fake module বসানো হয়। Tool-গুলো pure Python,
তাই সরাসরি real input দিয়ে test হয় — বিশেষ করে attack input দিয়ে।
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

from src.agent.loop import run_agent
from src.tools.calculator import calculate
from src.tools.filesystem import list_files, read_file
from src.tools.registry import TOOL_NAMES, TOOL_SCHEMAS, run_tool


# ── Calculator: correctness + injection/DoS attempts ──────────────────────
@pytest.mark.parametrize("expr,expected", [
    ("(12+8)*3", "60"), ("(12+8)*3/4", "15"), ("sqrt(16)", "4"), ("2**10", "1024"),
    ("-3 + 5", "2"), ("round(pi, 2)", "3.14"), ("7 // 2", "3"),
])
def test_calculator_math(expr, expected):
    assert calculate(expr) == expected


@pytest.mark.parametrize("attack", [
    "__import__('os').system('echo pwned')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "9**9**9", "10**1000", "'a'*10", "True+True", "x+1",
    "1" * 500,
])
def test_calculator_rejects_attacks(attack):
    assert calculate(attack).startswith("Error")


def test_calculator_division_by_zero():
    assert calculate("1/0") == "Error: division by zero"


# ── Filesystem sandbox: traversal, absolute path, symlink, extension ──────
@pytest.fixture
def box(tmp_path):
    root = tmp_path / "box"
    root.mkdir()
    (root / "a.txt").write_text("hi")
    (root / "run.sh").write_text("rm -rf /")
    (tmp_path / "secret.txt").write_text("nope")
    return root


def test_sandbox_reads_inside(box):
    assert list_files(box) == "a.txt\nrun.sh"
    assert read_file(box, "a.txt") == "hi"


def test_sandbox_blocks_escape(box, tmp_path):
    assert "escapes" in read_file(box, "../secret.txt")
    assert "escapes" in read_file(box, str(tmp_path / "secret.txt"))
    assert "escapes" in list_files(box, "..")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need admin on Windows")
def test_sandbox_blocks_symlink_escape(box, tmp_path):
    (box / "link.txt").symlink_to(tmp_path / "secret.txt")
    assert "escapes" in read_file(box, "link.txt")


def test_sandbox_hides_dotfiles(box):
    (box / ".env").write_text("API_KEY=secret")
    assert ".env" not in list_files(box)
    assert "hidden" in read_file(box, ".env")


def test_sandbox_extension_allowlist_and_truncation(box):
    assert "not allowed" in read_file(box, "run.sh")
    (box / "big.txt").write_text("x" * 50)
    assert read_file(box, "big.txt", max_chars=10).endswith("[truncated]")


# ── Registry: schema shape + argument validation ──────────────────────────
def test_schemas_are_ollama_shaped():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names == list(TOOL_NAMES) == ["calculate", "list_files", "read_file"]
    calc = TOOL_SCHEMAS[0]["function"]["parameters"]
    assert calc["required"] == ["expression"] and calc["properties"]["expression"]["type"] == "string"


def test_run_tool_errors_become_strings(tmp_path):
    assert "unknown tool" in run_tool("rm_rf", {}, tmp_path)
    assert "bad arguments" in run_tool("read_file", {}, tmp_path)
    assert "bad arguments" in run_tool("calculate", {"expression": 12}, tmp_path)  # strict: int is not str
    assert run_tool("calculate", {"expression": "2+2"}, tmp_path) == "4"


def test_run_tool_reports_invented_fields_with_example(tmp_path):
    # Real llama3.1:8b failure seen live: {"a": "90", "b": "75"} instead of {"expression": ...}
    out = run_tool("calculate", {"a": "90", "b": "75"}, tmp_path)
    assert "Extra inputs" in out and '{"expression": "(90+75)/2"}' in out


# ── Agent loop with a scripted fake LLM ───────────────────────────────────
def _script(*replies):
    it = iter(replies)
    return lambda messages, tools, model=None: next(it)


def _call(name, **args):
    return {"name": name, "arguments": args}


def test_agent_uses_tool_then_answers(tmp_path):
    fake = _script({"content": "", "tool_calls": [_call("calculate", expression="6*7")]},
                   {"content": "42", "tool_calls": []})
    res = run_agent("6*7?", sandbox=tmp_path, chat_fn=fake)
    assert (res.answer, res.status, res.llm_calls) == ("42", "answered", 2)
    assert res.steps[0].result == "42"


def test_agent_runs_parallel_calls_in_order(tmp_path):
    (tmp_path / "n.txt").write_text("code 7")
    seen = []

    def fake(messages, tools, model=None):
        seen.append([m["role"] for m in messages])
        if len(seen) == 1:
            return {"content": "", "tool_calls": [_call("calculate", expression="1+1"),
                                                  _call("read_file", path="n.txt")]}
        return {"content": "done", "tool_calls": []}

    res = run_agent("q", sandbox=tmp_path, chat_fn=fake)
    assert [s.tool for s in res.steps] == ["calculate", "read_file"]
    # assistant tool_calls message comes BEFORE the two tool results
    assert seen[1] == ["system", "user", "assistant", "tool", "tool"]


def test_agent_recovers_from_bad_tool_call(tmp_path):
    fake = _script({"content": "", "tool_calls": [_call("calculator", expression="1+1")]},
                   {"content": "", "tool_calls": [_call("calculate", expression="1+1")]},
                   {"content": "2", "tool_calls": []})
    res = run_agent("1+1", sandbox=tmp_path, chat_fn=fake)
    assert res.status == "answered" and "unknown tool" in res.steps[0].result


def test_agent_stops_at_max_steps(tmp_path):
    def forever(messages, tools, model=None):
        return {"content": "", "tool_calls": [_call("list_files")]}

    res = run_agent("x", sandbox=tmp_path, max_steps=3, chat_fn=forever)
    assert res.status == "max_steps" and len(res.steps) == 3


# ── LLM client: typed response parsing + failure wrapping ─────────────────
def _fake_ollama(message):
    mod = types.ModuleType("ollama")

    class Client:
        def __init__(self, **kw):
            pass

        def chat(self, **kw):
            assert kw["options"]["temperature"] == 0.0 and "seed" in kw["options"]
            return SimpleNamespace(message=message)

    mod.Client = Client
    return mod


def test_client_parses_typed_tool_calls(monkeypatch):
    from src.llm import client

    call = SimpleNamespace(function=SimpleNamespace(name="calculate", arguments={"expression": "1+1"}))
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama(SimpleNamespace(content=None, tool_calls=[call])))
    out = client.chat([], [])
    assert out == {"content": "", "tool_calls": [{"name": "calculate", "arguments": {"expression": "1+1"}}]}


def test_client_wraps_failures(monkeypatch):
    from src.llm import client

    monkeypatch.setitem(sys.modules, "ollama", None)  # import ollama -> ImportError
    with pytest.raises(client.OllamaConnectionError):
        client.chat([], [])
