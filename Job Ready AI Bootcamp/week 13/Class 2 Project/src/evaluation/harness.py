"""
Golden-task evaluation: end state + trajectory + pass^k.

HIGHLIGHTS: শুধু final answer দেখে agent grade করা ভুল — ভাগ্যক্রমে সঠিক
উত্তর আর policy ভাঙা (sold-out ফ্লাইট বুক, নাম ছাড়া বুক) দুটোই মিস হয়।
তাই তিন স্তরে মাপা হয়:

* **Outcome** — run শেষে ``tools.bookings`` (database state) expected-এর
  সাথে হুবহু মেলে কিনা, τ-bench-এর মতো।
* **Trajectory** — যে tool-গুলো ডাকা "আবশ্যক" সেগুলো ডাকা হয়েছে কিনা (subset match)।
* **Reliability** — একই task k বার চালিয়ে pass^k: k-টা run-এর **সবগুলো** সফল হলে তবেই pass।
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from src.agent.react_agent import run_agent
from src.tools.travel import TravelTools


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    bookings: frozenset = frozenset()           # expected {(flight_id, passenger)} at the end
    must_call: tuple[str, ...] = ()              # trajectory: these tools must appear
    answer_has: tuple[str, ...] = ()             # final answer must mention these (case-insensitive)
    must_not_call: tuple[str, ...] = ()          # policy: e.g. no book_flight attempt on a search-only request


GOLDEN_TASKS: tuple[Task, ...] = (
    Task("book_cheapest_tie", "Book the cheapest DAC to CXB flight on 2026-11-05 for Rahim",
         frozenset({("BG101", "Rahim")}), ("search_flights", "book_flight")),
    Task("book_single_seat", "Book the cheapest flight from DAC to DXB on 2026-11-10 for Nadia Islam",
         frozenset({("QR701", "Nadia Islam")}), ("search_flights", "book_flight")),
    Task("search_only", "Which flights go from DAC to CGP on 2026-11-05?",
         must_call=("search_flights",), answer_has=("BG201",), must_not_call=("book_flight",)),
    Task("cheapest_return", "What is the cheapest flight from CXB to DAC?",
         must_call=("search_flights",), answer_has=("BG501",), must_not_call=("book_flight",)),
    Task("sold_out", "Book flight BG102 for Karim", answer_has=("sold",)),
    Task("unknown_city", "Find me a flight from DAC to Paris", must_not_call=("book_flight",)),
    Task("missing_name", "Book the cheapest DAC to JSR flight on 2026-11-06", must_not_call=("book_flight",)),
    Task("over_budget", "Book a DAC to CXB flight on 2026-11-06 for Sara, but only if it costs under 6000 BDT",
         must_call=("search_flights",), must_not_call=("book_flight",)),
)


@dataclass
class Trial:
    task: str
    ok: bool
    status: str
    steps: int
    llm_calls: int
    invalid_calls: int
    seconds: float
    reason: str = ""


@dataclass
class Report:
    trials: list[Trial] = field(default_factory=list)

    def summary(self) -> dict:
        by_task: dict[str, list[bool]] = {}
        for t in self.trials:
            by_task.setdefault(t.task, []).append(t.ok)
        n = len(self.trials) or 1
        actions = sum(t.steps for t in self.trials) or 1
        return {
            "trials": len(self.trials),
            "success_rate": round(sum(t.ok for t in self.trials) / n, 3),
            "pass^k": round(sum(all(v) for v in by_task.values()) / (len(by_task) or 1), 3),
            "invalid_call_rate": round(sum(t.invalid_calls for t in self.trials) / actions, 3),
            "avg_llm_calls": round(sum(t.llm_calls for t in self.trials) / n, 2),
            "avg_seconds": round(sum(t.seconds for t in self.trials) / n, 2),
            "stop_reasons": dict(Counter(t.status for t in self.trials)),
        }


def grade(task: Task, tools: TravelTools, result) -> tuple[bool, str]:
    got = frozenset((b["flight_id"], b["passenger"]) for b in tools.bookings)
    if got != task.bookings:
        return False, f"bookings {sorted(got)} != expected {sorted(task.bookings)}"
    called = {s.tool for s in result.steps}
    missing = [t for t in task.must_call if t not in called]
    if missing:
        return False, f"never called {missing}"
    forbidden = [t for t in task.must_not_call if t in called]
    if forbidden:
        return False, f"policy: should not call {forbidden}"
    if result.status != "final":
        return False, f"stopped with {result.status}"
    lacking = [w for w in task.answer_has if w.lower() not in result.answer.lower()]
    if lacking:
        return False, f"answer missing {lacking}"
    return True, ""


def simulated_user(task: Task):
    """A deterministic stand-in for the human at the approval gate.

    The user approves a booking only for a passenger they actually named in their request —
    so an invented "John Doe" is denied, exactly as a real person would deny it.
    """
    return lambda details: details["passenger"].lower() in task.prompt.lower()


def evaluate(llm, tools: TravelTools, tasks=GOLDEN_TASKS, k: int = 1, mode: str = "native",
             model: str | None = None, **agent_kwargs) -> Report:
    report = Report()
    for task in tasks:
        for _ in range(k):
            tools.reset()
            tools.approve = simulated_user(task)
            t0 = time.perf_counter()
            res = run_agent(task.prompt, tools, llm, mode=mode, model=model, **agent_kwargs)
            ok, reason = grade(task, tools, res)
            invalid = sum(1 for s in res.steps if s.observation.get("error") in
                          {"unknown_tool", "invalid_arguments", "arguments_not_json", "parse_error", "repeated_call"})
            report.trials.append(Trial(task.id, ok, res.status, len(res.steps), res.llm_calls, invalid,
                                       round(time.perf_counter() - t0, 2), reason))
    return report
