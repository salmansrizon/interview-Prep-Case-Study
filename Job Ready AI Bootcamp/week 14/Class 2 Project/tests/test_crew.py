"""No Ollama, no network: only construction, wiring and file writing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from crewai import Process

import app
from src.crew import build_crew
from src.llm import build_llm, ollama_status
from src.report import save_report


def test_llm_is_ollama():
    assert build_llm().model == "ollama/llama3.1:8b"


def test_sequential_wiring():
    c = build_crew("RAG")
    assert c.process == Process.sequential
    assert [t.async_execution for t in c.tasks] == [False] * 3
    write, review = c.tasks[1], c.tasks[2]
    assert write.context == [c.tasks[0]] and review.context == [write]
    assert c.tasks[0].agent.role == "Senior Research Specialist"


def test_parallel_wiring():
    c = build_crew("RAG", "parallel")
    assert [t.async_execution for t in c.tasks] == [True, True, False, False]
    assert c.tasks[2].context == c.tasks[:2]


def test_hierarchical_has_manager():
    c = build_crew("RAG", "hierarchical")
    assert c.process == Process.hierarchical and c.manager_llm is not None


def test_bad_mode():
    with pytest.raises(ValueError):
        build_crew("x", "nope")


def test_output_file_set():
    assert build_crew("x", output_file="o.md").tasks[-1].output_file == "o.md"


def test_save_report(tmp_path):
    p = save_report("# Hi", "My Topic!", tmp_path)
    assert p.read_text(encoding="utf-8") == "# Hi\n" and p.name.startswith("my-topic-")


def test_status_unreachable():
    ok, msg = ollama_status(base_url="http://127.0.0.1:1")
    assert not ok and "Ollama" in msg


def test_app_exits_gracefully_without_ollama(monkeypatch, capsys):
    monkeypatch.setattr(app, "ollama_status", lambda m: (False, "no ollama"))
    assert app.main(["topic"]) == 1 and "no ollama" in capsys.readouterr().out
