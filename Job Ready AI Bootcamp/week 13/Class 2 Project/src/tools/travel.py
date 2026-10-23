"""
Travel tools: search_flights (read-only) and book_flight (side-effecting, gated).

HIGHLIGHTS — "deterministic" কোথায় সত্যি:
* Tool-গুলো pure function — একই input-এ সবসময় একই output। Sort একটা total
  order-এ (price, departs, id), তাই সমান দামের ফ্লাইটের ক্রম কখনো বদলায় না।
* Booking ref input থেকে derive হয় (``BK-BG101-RAHIM``), ``uuid4()`` না।
* State in-memory (``TravelTools`` instance) — test শেষে final state assert
  করা যায়, ঠিক τ-bench যেভাবে grade করে।
* ``execute()`` কখনো exception ছোড়ে না: সব ভুল একটা ``{"ok": False, ...}``
  observation হয়ে model-এর কাছে ফেরত যায়, যাতে সে পরের turn-এ শুধরে নেয়।
* book_flight-এর আগে ``approve()`` — human-in-the-loop gate, code-এ enforce
  করা। Prompt-এ "আগে confirm করো" বলা যথেষ্ট না।
"""
from __future__ import annotations

import copy
import difflib
import json
import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Approver = Callable[[dict], bool]  # gets booking details, returns True to allow


class _Args(BaseModel):
    # extra="forbid": model-এর বানানো field চুপচাপ বাদ না পড়ে error হয়।
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Args):
    origin: str = Field(description="3-letter IATA airport code, e.g. DAC")
    destination: str = Field(description="3-letter IATA airport code, e.g. CXB")
    date: str | None = Field(default=None, description="Travel date YYYY-MM-DD. Omit to search all dates.")
    max_price: int | None = Field(default=None, ge=0, description="Maximum price in BDT")

    @field_validator("origin", "destination")
    @classmethod
    def _iata(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError("must be a 3-letter airport code like DAC")
        return v

    @field_validator("date")
    @classmethod
    def _date(cls, v: str | None) -> str | None:
        if v in (None, ""):
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("must be YYYY-MM-DD")
        return v


class BookArgs(_Args):
    flight_id: str = Field(description="Flight id from search_flights results, e.g. BG101")
    passenger_name: str = Field(min_length=2, description="Full name of the passenger")

    @field_validator("flight_id")
    @classmethod
    def _fid(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("passenger_name")
    @classmethod
    def _name(cls, v: str) -> str:
        return " ".join(v.split())


SPECS: dict[str, tuple[type[_Args], str]] = {
    "search_flights": (SearchArgs,
        "Search the local flight database. Returns matching flights sorted cheapest first "
        "(ties broken by departure time). Always call this before booking; never invent flight ids."),
    "book_flight": (BookArgs,
        "Book ONE seat on a flight for a passenger. Only use a flight id returned by search_flights. "
        "Requires the user's approval; if approval is denied, tell the user and stop."),
}


def tool_schemas() -> list[dict]:
    """Ollama/OpenAI-style tool schemas, generated from the Pydantic models (single source of truth)."""
    out = []
    for name, (model, desc) in SPECS.items():
        params = model.model_json_schema()
        for k in ("title", "additionalProperties"):
            params.pop(k, None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
            prop.pop("default", None)
            if "anyOf" in prop:  # `str | None` -> plain "string"; optional = not in "required"
                prop["type"] = next(a["type"] for a in prop.pop("anyOf") if a.get("type") != "null")
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    return out


TOOL_DOC = "\n".join(
    f"- {n}({', '.join(m.model_fields)}): {d}" for n, (m, d) in SPECS.items()
)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")


class TravelTools:
    def __init__(self, flights_path: Path | str, approve: Approver | None = None, max_results: int = 5):
        self._seed = json.loads(Path(flights_path).read_text(encoding="utf-8"))
        self.flights = copy.deepcopy(self._seed)          # mutable in-memory state
        self.bookings: list[dict] = []
        self.approve: Approver = approve or (lambda details: True)
        self.max_results = max_results
        self.airports = sorted({f["origin"] for f in self.flights} | {f["destination"] for f in self.flights})

    def reset(self) -> None:
        self.flights, self.bookings = copy.deepcopy(self._seed), []

    # ── tools ──────────────────────────────────────────────────────────
    def search_flights(self, origin: str, destination: str, date: str | None = None,
                       max_price: int | None = None) -> dict:
        for code in (origin, destination):
            if code not in self.airports:
                return {"ok": False, "error": "unknown_airport", "code": code, "known_airports": self.airports,
                        "hint": "This database only covers these airports. Tell the user if their city is not served."}
        rows = [f for f in self.flights if f["origin"] == origin and f["destination"] == destination
                and (date is None or f["date"] == date)
                and (max_price is None or f["price_bdt"] <= max_price)]
        rows.sort(key=lambda f: (f["price_bdt"], f["departs"], f["id"]))  # total order -> stable
        keep = ("id", "date", "departs", "price_bdt", "seats_left")
        return {"ok": True, "count": len(rows),
                "flights": [{k: f[k] for k in keep} for f in rows[: self.max_results]]}

    def book_flight(self, flight_id: str, passenger_name: str) -> dict:
        flight = next((f for f in self.flights if f["id"] == flight_id), None)
        if flight is None:
            return {"ok": False, "error": "unknown_flight", "flight_id": flight_id,
                    "hint": "Call search_flights and use an id from its results."}
        if flight["seats_left"] <= 0:
            return {"ok": False, "error": "sold_out", "flight_id": flight_id,
                    "hint": "Pick another flight from the search results."}
        ref = f"BK-{flight_id}-{_slug(passenger_name)}"
        if any(b["booking_ref"] == ref for b in self.bookings):  # idempotent
            return {"ok": False, "error": "duplicate_booking", "booking_ref": ref,
                    "hint": "This passenger is already booked on this flight."}
        details = {"passenger": passenger_name, "flight_id": flight_id, "origin": flight["origin"],
                   "destination": flight["destination"], "date": flight["date"], "departs": flight["departs"],
                   "price_bdt": flight["price_bdt"]}
        summary = (f"{passenger_name} on {flight_id} {flight['origin']}->{flight['destination']} "
                   f"{flight['date']} {flight['departs']}, {flight['price_bdt']} BDT")
        if not self.approve({**details, "summary": summary}):
            return {"ok": False, "error": "approval_denied", "summary": summary,
                    "hint": "Nothing was booked. Tell the user the booking needs their confirmation."}
        flight["seats_left"] -= 1
        booking = {"booking_ref": ref, "flight_id": flight_id, "passenger": passenger_name,
                   "price_bdt": flight["price_bdt"], "status": "CONFIRMED"}
        self.bookings.append(booking)
        return {"ok": True, **booking}

    # ── trust boundary ─────────────────────────────────────────────────
    def execute(self, name: str, raw_args) -> dict:
        """Validate and run one tool call. Never raises."""
        if name not in SPECS:
            return {"ok": False, "error": "unknown_tool", "tool": name, "valid_tools": sorted(SPECS),
                    "did_you_mean": difflib.get_close_matches(name, SPECS, n=1)}
        if isinstance(raw_args, str):  # OpenAI-style JSON string, or text-ReAct input
            try:
                raw_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                return {"ok": False, "error": "arguments_not_json", "hint": "Send arguments as a JSON object."}
        if not isinstance(raw_args, dict):
            return {"ok": False, "error": "arguments_not_object", "hint": "Send arguments as a JSON object."}
        model, _ = SPECS[name]
        try:
            args = model.model_validate(raw_args)
        except ValidationError as e:
            return {"ok": False, "error": "invalid_arguments", "tool": name,
                    "details": [{"field": ".".join(map(str, d["loc"])) or "args", "msg": d["msg"]} for d in e.errors()],
                    "hint": "Correct the arguments and call again. Do not invent values; ask the user if unsure."}
        try:
            return getattr(self, name)(**args.model_dump())
        except Exception as e:  # noqa: BLE001 - a tool bug is still just an observation
            return {"ok": False, "error": "tool_failed", "msg": str(e)[:200]}
