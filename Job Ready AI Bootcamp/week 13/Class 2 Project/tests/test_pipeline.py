"""
Offline tests for the ReAct Travel Agent. Run: pytest tests -q

HIGHLIGHTS: LLM সবসময় একটা scripted fake — আগে থেকে লেখা reply একটা একটা
করে ফেরত দেয় আর কী message পেল সেটা রেকর্ড করে। তাই প্রতিটা test
deterministic, Ollama লাগে না (ADR 0006), আর আমরা assert করতে পারি *কেন*
loop থামল (typed status), শুধু থামল কিনা না।
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.react_agent import (NUDGE, STOP_SEQUENCES, fallback_answer, leaked_call, parse_text_step,
                                   progress_note, run_agent)
from src.evaluation.harness import GOLDEN_TASKS, Task, evaluate, grade, simulated_user
from src.llm.client import LLMUnavailable
from src.tools.travel import TravelTools, tool_schemas

FLIGHTS = Path(__file__).resolve().parents[1] / "data" / "flights.json"


@pytest.fixture
def tools():
    return TravelTools(FLIGHTS)


# ── scripted fake LLM ─────────────────────────────────────────────────
def call(name, **args):
    return {"name": name, "arguments": args}


def say(text="", *calls):
    return {"content": text, "tool_calls": list(calls)}


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.seen, self.kwargs = list(replies), [], []

    def __call__(self, messages, **kw):
        self.seen.append([dict(m) for m in messages])
        self.kwargs.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# ── tools: deterministic behaviour ────────────────────────────────────
def test_search_total_order_breaks_price_ties(tools):
    r = tools.search_flights("DAC", "CXB", date="2026-11-05")
    # BG101 and US103 both cost 6500 -> earlier departure first, every time
    assert [f["id"] for f in r["flights"]] == ["BG101", "US103", "BG102"]
    assert set(r["flights"][0]) == {"id", "date", "departs", "price_bdt", "seats_left"}  # compact


def test_search_filters_and_unknown_airport(tools):
    assert [f["id"] for f in tools.search_flights("DAC", "CXB", max_price=6600)["flights"]] == ["BG101", "US103"]
    r = tools.search_flights("DAC", "PAR")
    assert r["ok"] is False and r["error"] == "unknown_airport" and "DAC" in r["known_airports"]


def test_booking_rules_and_derived_refs(tools):
    b = tools.book_flight("BG101", "Rahim")
    assert b == {"ok": True, "booking_ref": "BK-BG101-RAHIM", "flight_id": "BG101", "passenger": "Rahim",
                 "price_bdt": 6500, "status": "CONFIRMED"}
    assert tools.book_flight("BG101", "Rahim")["error"] == "duplicate_booking"
    assert tools.book_flight("BG102", "Karim")["error"] == "sold_out"
    assert tools.book_flight("ZZ999", "Karim")["error"] == "unknown_flight"
    assert next(f for f in tools.flights if f["id"] == "BG101")["seats_left"] == 4
    tools.reset()
    assert tools.bookings == [] and next(f for f in tools.flights if f["id"] == "BG101")["seats_left"] == 5


def test_approval_gate_blocks_side_effect(tools):
    seen = []
    tools.approve = lambda d: seen.append(d) or False
    r = tools.book_flight("BG101", "Rahim")
    assert r["error"] == "approval_denied" and tools.bookings == []
    assert seen[0]["passenger"] == "Rahim" and seen[0]["price_bdt"] == 6500


# ── trust boundary: execute() validates and never raises ──────────────
@pytest.mark.parametrize("name,args,error", [
    ("search_flight", {}, "unknown_tool"),
    ("search_flights", {"origin": "DAC"}, "invalid_arguments"),
    ("search_flights", {"origin": "DAC", "destination": "CXB", "date": "5 Nov"}, "invalid_arguments"),
    ("search_flights", {"from": "DAC", "to": "CXB"}, "invalid_arguments"),         # invented fields
    ("book_flight", {"flight_id": "BG101", "passenger_name": "R"}, "invalid_arguments"),
    ("book_flight", "{not json", "arguments_not_json"),
    ("book_flight", ["BG101"], "arguments_not_object"),
])
def test_execute_returns_error_observations(tools, name, args, error):
    obs = tools.execute(name, args)
    assert obs["ok"] is False and obs["error"] == error


def test_execute_hints_and_coercion(tools):
    assert tools.execute("search_flight", {})["did_you_mean"] == ["search_flights"]
    obs = tools.execute("search_flights", {"origin": "dac ", "destination": "cxb", "max_price": "6600"})
    assert obs["ok"] and obs["count"] == 2           # lowercase codes + "6600" string coerced
    assert tools.execute("search_flights", '{"origin": "DAC", "destination": "CGP"}')["count"] == 1


def test_tool_schemas_are_clean():
    s = {t["function"]["name"]: t["function"]["parameters"] for t in tool_schemas()}
    assert s["search_flights"]["required"] == ["origin", "destination"]
    assert s["search_flights"]["properties"]["date"]["type"] == "string"   # no anyOf for small models
    assert s["book_flight"]["required"] == ["flight_id", "passenger_name"]


# ── native loop ───────────────────────────────────────────────────────
def test_native_search_then_book_then_final_answer(tools):
    llm = FakeLLM([say("", call("search_flights", origin="DAC", destination="CXB", date="2026-11-05")),
                   say("", call("book_flight", flight_id="BG101", passenger_name="Rahim")),
                   say("", call("final_answer", answer="Booked BK-BG101-RAHIM"))])
    res = run_agent("Book cheapest DAC-CXB on 2026-11-05 for Rahim", tools, llm)
    assert (res.status, res.answer, res.llm_calls) == ("final", "Booked BK-BG101-RAHIM", 3)
    assert [s.tool for s in res.steps] == ["search_flights", "book_flight"]
    assert tools.bookings[0]["booking_ref"] == "BK-BG101-RAHIM"
    roles = [m["role"] for m in llm.seen[1]]
    assert roles == ["system", "user", "assistant", "tool", "user"]      # assistant, tool result, nudge
    nudge = llm.seen[1][-1]["content"]
    assert "My request: Book cheapest" in nudge and "BG101 6500 BDT" in nudge
    assert any(t["function"]["name"] == "final_answer" for t in llm.kwargs[0]["tools"])


def test_plain_text_reply_is_final(tools):
    res = run_agent("hi", tools, FakeLLM([say("Hello! Where do you want to fly?")]))
    assert res.status == "final" and res.answer.startswith("Hello") and res.steps == []


def test_leaked_tool_call_in_content_is_rescued(tools):
    llm = FakeLLM([say('Let me search. {"name": "search_flights", "parameters": {"origin": "DAC", "destination": "CGP"}}'),
                   say("", call("final_answer", answer="BG201"))])
    res = run_agent("DAC to CGP?", tools, llm)
    assert res.steps[0].tool == "search_flights" and res.steps[0].observation["count"] == 1


def test_self_corrects_after_invalid_arguments(tools):
    llm = FakeLLM([say("", call("search_flights", **{"from": "DAC", "to": "CXB"})),
                   say("", call("search_flights", origin="DAC", destination="CXB")),
                   say("", call("final_answer", answer="ok"))])
    res = run_agent("x", tools, llm)
    assert res.status == "final"
    fed_back = json.loads(llm.seen[1][3]["content"])     # the tool message after the bad call
    assert fed_back["error"] == "invalid_arguments" and fed_back["hint"]


def test_repeat_gets_reminder_then_loop_stop(tools):
    same = call("search_flights", origin="DAC", destination="CXB")
    swapped = call("search_flights", destination="CXB", origin="DAC")  # same args, different key order
    llm = FakeLLM([say("", same), say("", swapped), say("", same)])
    res = run_agent("x", tools, llm, max_repeat_calls=2)
    assert res.status == "loop" and len(res.steps) == 3
    assert res.steps[1].observation["error"] == "repeated_call" and res.steps[1].observation["previous_result"]["ok"]
    assert "Nothing was booked" in res.answer and "BG101" in res.answer          # deterministic fallback


def test_max_steps(tools):
    llm = FakeLLM([say("", call("search_flights", origin="DAC", destination="CXB", max_price=p))
                   for p in (9000, 9001, 9002)])
    res = run_agent("x", tools, llm, max_steps=3)
    assert res.status == "max_steps" and res.llm_calls == 3


def test_error_budget(tools):
    # args differ each time, so the repeat guard does not fire first
    llm = FakeLLM([say("", call("fly_plane", n=i)) for i in range(3)])
    res = run_agent("x", tools, llm, max_consecutive_errors=3)
    assert res.status == "error_budget"


def test_timeout_with_fake_clock(tools):
    ticks = iter([0, 0, 500])
    llm = FakeLLM([say("", call("search_flights", origin="DAC", destination="CXB"))])
    res = run_agent("x", tools, llm, deadline_seconds=180, clock=lambda: next(ticks))
    assert res.status == "timeout" and res.llm_calls == 1


def test_llm_retry_then_success_and_exhaustion(tools):
    ok = FakeLLM([LLMUnavailable("down"), say("fine")])
    assert run_agent("x", tools, ok, max_llm_retries=2).status == "final"
    bad = FakeLLM([LLMUnavailable("down")] * 5)
    res = run_agent("x", tools, bad, max_llm_retries=2)
    assert res.status == "llm_failed" and len(bad.seen) == 3


def test_history_is_kept_for_conversation(tools):
    llm = FakeLLM([say("Booked.")])
    history = [{"role": "user", "content": "Find DAC to CXB"}, {"role": "assistant", "content": "BG101 is cheapest"}]
    run_agent("book it for Rahim", tools, llm, history=history)
    assert [m["content"] for m in llm.seen[0][1:]] == ["Find DAC to CXB", "BG101 is cheapest", "book it for Rahim"]


# ── text (classic) ReAct ──────────────────────────────────────────────
def test_parse_text_step_variants():
    a = parse_text_step('Thought: search first\nAction: search_flights\nAction Input: ```json\n{"origin": "DAC"}\n```')
    assert a == {"thought": "search first", "action": "search_flights", "input": '{"origin": "DAC"}'}
    f = parse_text_step("Thought: done\nFinal Answer: Booked BK-1")
    assert f == {"thought": "done", "final": "Booked BK-1"}
    with pytest.raises(ValueError):
        parse_text_step("I think I will book it.")


def test_text_mode_loop_uses_stop_sequence_and_observations(tools):
    llm = FakeLLM([{"content": 'Thought: search\nAction: search_flights\nAction Input: {"origin": "DAC", "destination": "CGP"}'},
                   {"content": "rambling without format"},
                   {"content": "Thought: got it\nFinal Answer: BG201 at 09:15"}])
    res = run_agent("DAC to CGP?", tools, llm, mode="text")
    assert res.status == "final" and res.answer == "BG201 at 09:15"
    assert llm.kwargs[0]["stop"] == STOP_SEQUENCES and "tools" not in llm.kwargs[0]
    assert llm.seen[1][-1]["content"].startswith("Observation: {")
    assert res.steps[1].observation["error"] == "parse_error"


def test_leaked_call_parser():
    assert leaked_call('{"name": "book_flight", "arguments": {"flight_id": "BG101"}}')["arguments"] == {"flight_id": "BG101"}
    assert leaked_call("no json here") is None and leaked_call('{"foo": 1}') is None


def test_progress_note_and_nudge_format(tools):
    llm = FakeLLM([say("", call("book_flight", flight_id="BG102", passenger_name="Karim")),
                   say("", call("final_answer", answer="sold out"))])
    res = run_agent("Book BG102 for Karim", tools, llm)
    note = progress_note(res.steps)
    assert "FAILED: sold_out" in note
    assert NUDGE.format(request="r", done=note).startswith("My request: r")


def test_fallback_answer_mentions_confirmed_booking(tools):
    llm = FakeLLM([say("", call("book_flight", flight_id="BG101", passenger_name="Rahim"))] * 3)
    res = run_agent("x", tools, llm)
    assert res.status == "loop" and "BK-BG101-RAHIM" in res.answer
    assert fallback_answer("timeout", []) == "I couldn't finish (ran out of time). Nothing was booked."


# ── evaluation harness ────────────────────────────────────────────────
def test_simulated_user_denies_invented_passenger():
    t = Task("x", "Book the cheapest DAC to JSR flight")
    assert simulated_user(t)({"passenger": "John Doe"}) is False
    assert simulated_user(Task("y", "book for Rahim"))({"passenger": "Rahim"}) is True


def test_grade_checks_state_trajectory_and_policy(tools):
    task = next(t for t in GOLDEN_TASKS if t.id == "search_only")
    llm = FakeLLM([say("", call("search_flights", origin="DAC", destination="CGP", date="2026-11-05")),
                   say("", call("final_answer", answer="BG201 departs 09:15"))])
    assert grade(task, tools, run_agent(task.prompt, tools, llm)) == (True, "")
    tools.reset()
    llm = FakeLLM([say("", call("book_flight", flight_id="BG201", passenger_name="Xi Li")),
                   say("", call("final_answer", answer="BG201"))])
    ok, why = grade(task, tools, run_agent(task.prompt, tools, llm))   # booked without being asked
    assert not ok and "bookings" in why


def test_evaluate_reports_pass_k(tools):
    task = Task("t", "hi")
    llm = FakeLLM([say("hello"), say("hello"), say("hello"), say("hello")])
    rep = evaluate(llm, tools, tasks=(task,), k=4)
    s = rep.summary()
    assert s["trials"] == 4 and s["success_rate"] == 1.0 and s["pass^k"] == 1.0
    assert s["stop_reasons"] == {"final": 4}


# ── LLM client ────────────────────────────────────────────────────────
def test_client_parses_typed_response(monkeypatch):
    from src.llm import client

    mod = types.ModuleType("ollama")
    captured = {}

    class Client:
        def __init__(self, **kw):
            pass

        def chat(self, **kw):
            captured.update(kw)
            fn = SimpleNamespace(name="search_flights", arguments={"origin": "DAC"})
            return SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(function=fn)]),
                                   prompt_eval_count=321, eval_count=12)

    mod.Client = Client
    monkeypatch.setitem(sys.modules, "ollama", mod)
    out = client.chat([], stop=["Observation:"])
    assert out == {"content": "", "tool_calls": [{"name": "search_flights", "arguments": {"origin": "DAC"}}],
                   "prompt_tokens": 321, "output_tokens": 12}
    assert captured["options"]["temperature"] == 0.0 and captured["options"]["stop"] == ["Observation:"]


def test_client_wraps_failures(monkeypatch):
    from src.llm import client

    monkeypatch.setitem(sys.modules, "ollama", None)
    with pytest.raises(LLMUnavailable):
        client.chat([])


# ── gap-filling tests (added after jev review) ────────────────────────
def test_pass_k_counts_task_only_if_all_trials_pass(tools):
    # trial 1 answers (ok), trial 2 books when it should not (fail) -> success 0.5, pass^2 = 0
    task = Task("t", "Which flights go from DAC to CGP?", must_not_call=("book_flight",))
    llm = FakeLLM([say("BG201"),
                   say("", call("book_flight", flight_id="BG201", passenger_name="Xi Li")), say("done")])
    s = evaluate(llm, tools, tasks=(task,), k=2).summary()
    assert s["success_rate"] == 0.5 and s["pass^k"] == 0.0


def test_text_mode_parse_errors_spend_error_budget(tools):
    llm = FakeLLM([{"content": "no format"}] * 3)
    res = run_agent("x", tools, llm, mode="text", max_consecutive_errors=3)
    assert res.status == "error_budget" and [s.observation["error"] for s in res.steps] == ["parse_error"] * 3


def test_max_results_truncates_but_count_is_total():
    small = TravelTools(FLIGHTS, max_results=1)
    r = small.search_flights("DAC", "CXB")
    assert r["count"] == 4 and [f["id"] for f in r["flights"]] == ["BG101"]


def test_nudge_can_be_disabled_and_on_step_is_called(tools):
    seen_steps = []
    llm = FakeLLM([say("", call("search_flights", origin="DAC", destination="CGP")), say("BG201")])
    run_agent("x", tools, llm, nudge_after_tools=False, on_step=seen_steps.append)
    assert [m["role"] for m in llm.seen[1]] == ["system", "user", "assistant", "tool"]   # no nudge
    assert [s.tool for s in seen_steps] == ["search_flights"]


def test_final_answer_without_answer_field_falls_back_to_content(tools):
    res = run_agent("x", tools, FakeLLM([say("Here you go", call("final_answer"))]))
    assert res.status == "final" and res.answer == "Here you go"


def test_first_repeat_does_not_re_execute_side_effect(tools):
    b = call("book_flight", flight_id="BG101", passenger_name="Rahim")
    llm = FakeLLM([say("", b), say("", b), say("", call("final_answer", answer="done"))])
    res = run_agent("x", tools, llm)
    assert res.status == "final" and len(tools.bookings) == 1
    assert res.steps[1].observation["error"] == "repeated_call"
