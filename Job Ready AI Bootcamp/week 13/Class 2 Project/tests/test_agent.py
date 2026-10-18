"""Tests use a scripted fake LLM: no Ollama needed.  Run: pytest tests -q"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.react_agent import parse_step, run_agent  # noqa: E402
from src.tools import ToolError, TravelTools  # noqa: E402

FLIGHTS = str(Path(__file__).resolve().parents[1] / "data" / "flights.json")


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, messages):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def act(name, **kw):
    return {"thought": "t", "action": name, "action_input": kw}


@pytest.fixture
def tools():
    # tempfile (not pytest tmp_path): avoids Windows basetemp permission issues
    with tempfile.TemporaryDirectory() as d:
        yield TravelTools(FLIGHTS, str(Path(d) / "b.json"))


def test_search_sorted_and_filters(tools):
    r = tools.search_flights("dac", "cxb", date="2026-11-05")
    assert [f["id"] for f in r] == ["BG101", "BG102"]
    assert tools.search_flights("DAC", "CXB", max_price=6600)[0]["id"] == "BG101"
    with pytest.raises(ToolError):
        tools.search_flights("XXX", "CXB")


def test_booking_rules(tools):
    b = tools.book_flight("BG101", "Rahim")
    assert b["status"] == "CONFIRMED" and b["booking_ref"] == "BK0001"
    with pytest.raises(ToolError, match="already"):
        tools.book_flight("BG101", "Rahim")
    with pytest.raises(ToolError, match="sold out"):
        tools.book_flight("BG102", "Karim")
    with pytest.raises(ToolError, match="No flight"):
        tools.book_flight("ZZ9", "Karim")


def test_happy_path_search_then_book(tools):
    llm = FakeLLM([act("search_flights", origin="DAC", destination="CXB", date="2026-11-05"),
                   act("book_flight", flight_id="BG101", passenger_name="Rahim"),
                   {"thought": "ok", "final_answer": "Booked BK0001"}])
    res = run_agent("book", tools, llm)
    assert res.status == "done" and "BK0001" in res.answer
    assert [s.action for s in res.steps] == ["search_flights", "book_flight", ""]
    assert len(json.loads(tools.bookings_path.read_text())) == 1


def test_tool_error_is_fed_back_and_recovered(tools):
    llm = FakeLLM([act("book_flight", flight_id="BG102", passenger_name="Rahim"),  # sold out
                   act("book_flight", flight_id="BG101", passenger_name="Rahim"),
                   {"thought": "", "final_answer": "done"}])
    res = run_agent("book", tools, llm)
    assert res.status == "done"
    assert res.steps[0].observation.startswith("ERROR") and "sold out" in res.steps[0].observation


def test_malformed_json_and_unknown_tool_recover(tools):
    llm = FakeLLM(["blah blah", act("fly_plane"), {"thought": "", "final_answer": "ok"}])
    res = run_agent("x", tools, llm)
    assert res.status == "done"
    assert "invalid reply" in res.steps[0].observation and "Unknown tool" in res.steps[1].observation


def test_max_steps(tools):
    llm = FakeLLM([act("search_flights", origin="DAC", destination="CXB", max_price=p)
                   for p in (9000, 9001, 9002)])
    res = run_agent("x", tools, llm, max_steps=3)
    assert res.status == "max_steps" and len(res.steps) == 3


def test_loop_detection(tools):
    same = act("search_flights", origin="DAC", destination="CXB")
    res = run_agent("x", tools, FakeLLM([same] * 5), max_steps=8, max_repeat_actions=2)
    assert res.status == "loop_detected" and len(res.steps) == 3


def test_llm_retry_then_success_and_exhaustion(tools):
    ok = FakeLLM([ConnectionError("down"), {"thought": "", "final_answer": "hi"}])
    assert run_agent("x", tools, ok, max_llm_retries=2).status == "done"
    bad = FakeLLM([ConnectionError("down")] * 5)
    res = run_agent("x", tools, bad, max_llm_retries=2)
    assert res.status == "llm_failed" and bad.calls == 3


def test_parse_step_tolerates_fences():
    assert parse_step('```json\n{"action": "a"}\n```')["action"] == "a"
    with pytest.raises(ValueError):
        parse_step('{"foo": 1}')
