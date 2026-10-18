"""
Tools the agent can call: search_flights and book_flight (simulated).

HIGHLIGHTS: Tool মানে সাধারণ Python function — LLM শুধু নাম আর argument
বেছে দেয়, execute করে আমাদের code। তাই এখানেই সব validation: LLM-এর দেওয়া
argument কখনো বিশ্বাসযোগ্য না। ভুল হলে ToolError raise হয়, আর loop সেটাকে
Observation বানিয়ে model-কে ফেরত দেয় যাতে সে নিজেকে শুধরে নিতে পারে।
"""
from __future__ import annotations

import json
from pathlib import Path


class ToolError(Exception):
    """Recoverable tool failure: the message goes back to the model as Observation."""


class TravelTools:
    def __init__(self, flights_path: str, bookings_path: str):
        self.flights = json.loads(Path(flights_path).read_text(encoding="utf-8"))
        self.bookings_path = Path(bookings_path)
        # ponytail: JSON file store, no locking; swap to SQLite if concurrent writers matter.

    def search_flights(self, origin: str, destination: str, date: str | None = None,
                       max_price: int | None = None) -> list[dict]:
        o, d = str(origin).upper().strip(), str(destination).upper().strip()
        known = {f["origin"] for f in self.flights} | {f["destination"] for f in self.flights}
        for code in (o, d):
            if code not in known:
                raise ToolError(f"Unknown airport code '{code}'. Known: {sorted(known)}")
        try:
            cap = None if max_price is None else int(max_price)
        except (TypeError, ValueError):
            raise ToolError("max_price must be an integer") from None
        out = [f for f in self.flights if f["origin"] == o and f["destination"] == d
               and (not date or f["date"] == date)
               and (cap is None or f["price_bdt"] <= cap)]
        return sorted(out, key=lambda f: f["price_bdt"])

    def book_flight(self, flight_id: str, passenger_name: str) -> dict:
        name = str(passenger_name or "").strip()
        if not name:
            raise ToolError("passenger_name is required")
        fid = str(flight_id).upper().strip()
        flight = next((f for f in self.flights if f["id"] == fid), None)
        if flight is None:
            raise ToolError(f"No flight with id '{flight_id}'. Use search_flights first.")
        if flight["seats_left"] <= 0:
            raise ToolError(f"Flight {fid} is sold out. Pick another flight.")
        bookings = self._load_bookings()
        if any(b["flight_id"] == fid and b["passenger"] == name for b in bookings):
            raise ToolError(f"{name} already booked on {fid}.")
        flight["seats_left"] -= 1  # in-memory only: the booking itself is simulated
        booking = {"booking_ref": f"BK{len(bookings) + 1:04d}", "flight_id": fid,
                   "passenger": name, "price_bdt": flight["price_bdt"], "status": "CONFIRMED"}
        bookings.append(booking)
        self.bookings_path.write_text(json.dumps(bookings, indent=2), encoding="utf-8")
        return booking

    def _load_bookings(self) -> list[dict]:
        if self.bookings_path.exists():
            return json.loads(self.bookings_path.read_text(encoding="utf-8"))
        return []

    def call(self, name: str, args: dict):
        fn = {"search_flights": self.search_flights, "book_flight": self.book_flight}.get(name)
        if fn is None:
            raise ToolError(f"Unknown tool '{name}'. Available: search_flights, book_flight")
        if not isinstance(args, dict):
            raise ToolError("action_input must be a JSON object")
        try:
            return fn(**args)
        except TypeError as e:  # wrong or missing argument names
            raise ToolError(f"Bad arguments for {name}: {e}") from e


TOOL_DOC = """- search_flights(origin, destination, date=None, max_price=None): airport codes like DAC, CXB; date "YYYY-MM-DD"; returns flights sorted by price.
- book_flight(flight_id, passenger_name): books ONE seat (simulated). Only call after searching and picking a real flight id."""
