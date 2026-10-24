"""
Offline tests for the Self-Correcting LLM Coder. Run: pytest tests -q

HIGHLIGHTS: LLM সবসময় scripted fake (ADR 0006)। Sandbox আর static check
real — সত্যিকারের subprocess চলে, কারণ এগুলোই project-এর নিরাপত্তা আর
সঠিকতার কেন্দ্র। Graph test-এ routing, retry budget, hidden-test রহস্য রাখা,
interrupt/resume আর SQLite checkpointer — সব assert করা হয়।
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from langgraph.types import Command

from src.evaluation.bench import run_bench, summarize
from src.graph.coder import build_graph, build_prompt, initial_state, run_config, sqlite_checkpointer
from src.sandbox.runner import run_tests
from src.sandbox.static_check import extract_code, static_check
from src.tasks.bank import TASKS, custom_task

BANNED_IMPORTS = ("os", "subprocess", "socket")
BANNED_CALLS = ("eval", "exec", "open", "__import__")

ROMAN_OK = ("```python\ndef roman_to_int(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
            "    t=0\n    for i,c in enumerate(s):\n        if i+1<len(s) and v[c]<v[s[i+1]]: t-=v[c]\n"
            "        else: t+=v[c]\n    return t\n```")
ROMAN_NO_SUBTRACT = "```python\ndef roman_to_int(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    return sum(v[c] for c in s)\n```"
# Hard-codes the three visible examples: passes visible, fails hidden.
ROMAN_CHEAT = ("```python\ndef roman_to_int(s):\n    return {'III': 3, 'IV': 4, 'MCMXCIV': 1994}.get(s, 0)\n```")


def fake(*replies):
    it = iter(replies)
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return next(it)

    llm.prompts = prompts
    return llm


# ── code extraction + static check ────────────────────────────────────
def test_extract_code_takes_last_fenced_block():
    text = "First try:\n```python\nx = 1\n```\nBetter:\n```python\ndef f():\n    return 2\n```\nDone."
    assert extract_code(text) == "def f():\n    return 2"
    assert extract_code("def g(): pass") == "def g(): pass"


@pytest.mark.parametrize("code,needle", [
    ("def f(:", "SyntaxError"),
    ("def other(): return 1", "must define a function named `f`"),
    ("import os\ndef f(): return os.getcwd()", "Importing `os`"),
    ("from subprocess import run\ndef f(): pass", "Importing `subprocess`"),
    ("def f(): return eval('1')", "Calling `eval()`"),
    ("def f(): return ().__class__", "Dunder attribute"),
])
def test_static_check_flags_problems(code, needle):
    problems = static_check(code, "f", BANNED_IMPORTS, BANNED_CALLS)
    assert any(needle in p for p in problems), problems


def test_static_check_allows_clean_code():
    assert static_check("import re\nfrom collections import Counter\ndef f(x):\n    return x", "f",
                        BANNED_IMPORTS, BANNED_CALLS) == []


# ── sandbox runner (real subprocess) ──────────────────────────────────
def test_runner_reports_per_test_diffs():
    rep = run_tests("def roman_to_int(s): return 6", TASKS["roman"].visible)
    assert (rep.passed, rep.total, rep.all_passed) == (0, 3, False)
    assert "roman_to_int('IV'): returned 6 (int), expected 4" in rep.feedback()


def test_runner_handles_exceptions_and_expected_raises():
    rep = run_tests("def second_largest(nums):\n    u = sorted(set(nums))\n    return u[-2]",
                    TASKS["second_largest"].visible)
    assert rep.passed == 2
    assert "raised IndexError" in rep.feedback() and "expected ValueError" in rep.feedback()


def test_runner_type_must_match():
    rep = run_tests("def excel_column(n): return 'A' if n == 1 else 0", [("excel_column(1)", "'A'"), ("excel_column(2)", "'B'")])
    assert rep.passed == 1 and "(int)" in rep.feedback()


def test_runner_timeout_and_load_errors():
    assert "Timeout" in run_tests("while True:\n    pass", [("1", "1")], timeout=2).feedback()
    assert "code failed to load: SyntaxError" in run_tests("def f(:", [("f()", "1")]).feedback()


def test_runner_isolated_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "shh")
    rep = run_tests("def f():\n    import os\n    return os.environ.get('SECRET_TOKEN')", [("f()", "None")])
    assert rep.all_passed   # the child gets a scrubbed environment


def test_task_bank_is_self_consistent():
    for t in TASKS.values():
        assert t.visible and t.hidden and t.func_name in t.prompt
        assert not set(t.visible) & set(t.hidden)


# ── graph: routing, budget, hidden tests, interrupt ───────────────────
def run(graph, task, tid):
    return graph.invoke(initial_state(task), run_config(tid))


def test_first_attempt_success():
    g = build_graph(fake(ROMAN_OK), require_approval=False)
    out = run(g, TASKS["roman"], "t1")
    assert out["status"] == "done" and out["attempts"] == 1 and out["first_attempt_passed"]
    assert (out["visible_passed"], out["hidden_passed"]) == (3, 3)


def test_repair_feeds_failing_tests_back():
    llm = fake(ROMAN_NO_SUBTRACT, ROMAN_OK)
    out = run(build_graph(llm, require_approval=False), TASKS["roman"], "t2")
    assert out["status"] == "done" and out["attempts"] == 2 and not out["first_attempt_passed"]
    assert "roman_to_int('IV'): returned 6 (int), expected 4" in llm.prompts[1]
    assert "Your previous attempt" in llm.prompts[1]


def test_static_failure_is_fed_back_without_running():
    llm = fake("```python\nimport os\ndef roman_to_int(s): return 0\n```", ROMAN_OK)
    out = run(build_graph(llm, require_approval=False), TASKS["roman"], "t3")
    assert out["status"] == "done" and "Importing `os`" in llm.prompts[1]
    assert any("static check FAILED" in h for h in out["history"])


def test_hidden_tests_catch_cheating_but_stay_hidden():
    llm = fake(ROMAN_CHEAT, ROMAN_OK)
    out = run(build_graph(llm, require_approval=False), TASKS["roman"], "t4")
    assert out["status"] == "done" and out["attempts"] == 2
    retry_prompt = llm.prompts[1]
    assert "hidden" in retry_prompt and "LVIII" not in retry_prompt and "58" not in retry_prompt


def test_hidden_fail_when_budget_runs_out():
    out = run(build_graph(fake(ROMAN_CHEAT, ROMAN_CHEAT), max_attempts=2, require_approval=False),
              TASKS["roman"], "t5")
    assert out["status"] == "hidden_fail" and out["hidden_passed"] < out["hidden_total"]


def test_give_up_keeps_best_attempt():
    worse = "```python\ndef roman_to_int(s): return 0\n```"
    out = run(build_graph(fake(ROMAN_NO_SUBTRACT, worse), max_attempts=2, require_approval=False),
              TASKS["roman"], "t6")
    assert out["status"] == "gave_up" and out["code"] == extract_code(ROMAN_NO_SUBTRACT)
    assert out["best_passed"] == 1      # 'III' passes with plain summation


def test_approval_interrupt_then_resume():
    g = build_graph(fake(ROMAN_OK))
    first = g.invoke(initial_state(TASKS["roman"]), run_config("t7"))
    payload = first["__interrupt__"][0].value
    assert "def roman_to_int" in payload["code"] and payload["hidden"] == "3/3"
    assert g.invoke(Command(resume=True), run_config("t7"))["status"] == "done"
    g2 = build_graph(fake(ROMAN_OK))
    g2.invoke(initial_state(TASKS["roman"]), run_config("t8"))
    assert g2.invoke(Command(resume=False), run_config("t8"))["status"] == "rejected"


def test_sqlite_checkpointer_persists_and_time_travels(tmp_path):
    db = tmp_path / "cp.sqlite"
    g = build_graph(fake(ROMAN_NO_SUBTRACT, ROMAN_OK), require_approval=False, checkpointer=sqlite_checkpointer(db))
    g.invoke(initial_state(TASKS["roman"]), run_config("persist"))
    # a brand-new graph object reading the same file sees the finished run
    g2 = build_graph(fake(), require_approval=False, checkpointer=sqlite_checkpointer(db))
    snap = g2.get_state(run_config("persist"))
    assert snap.values["status"] == "done" and snap.values["attempts"] == 2
    history = list(g2.get_state_history(run_config("persist")))
    assert len(history) > 5 and history[-1].values.get("attempts", 0) == 0   # oldest checkpoint first state


def test_custom_task_without_hidden_tests():
    t = custom_task("double", "Write double(x) returning 2*x.", [("double(2)", "4")])
    out = run(build_graph(fake("```python\ndef double(x):\n    return 2 * x\n```"), require_approval=False), t, "t9")
    assert out["status"] == "done" and out["hidden_total"] == 0


def test_prompt_contains_examples_and_name():
    p = build_prompt(initial_state(TASKS["excel"]))
    assert "`excel_column`" in p and "excel_column(28)  ->  'AB'" in p and "previous attempt" not in p


def test_bench_summary_counts_repairs():
    script = {"roman": [ROMAN_NO_SUBTRACT, ROMAN_OK]}
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return script["roman"][len(calls) - 1]

    rows = run_bench(llm, task_ids=["roman"])
    s = summarize(rows)
    assert s["pass@1"] == 0 and s["pass_after_repair"] == 1 and s["repaired"] == 1


def test_client_wraps_failures(monkeypatch):
    from src.llm import client

    monkeypatch.setitem(sys.modules, "ollama", None)
    with pytest.raises(client.OllamaUnavailable):
        client.generate("x")


def test_client_sends_greedy_options(monkeypatch):
    from src.llm import client

    seen = {}
    mod = types.ModuleType("ollama")

    class Client:
        def __init__(self, **kw):
            pass

        def chat(self, **kw):
            seen.update(kw)
            return SimpleNamespace(message=SimpleNamespace(content="```python\nx=1\n```"))

    mod.Client = Client
    monkeypatch.setitem(sys.modules, "ollama", mod)
    assert client.generate("task").startswith("```python")
    assert seen["options"]["temperature"] == 0.0 and seen["model"] == "llama3.1:8b"
