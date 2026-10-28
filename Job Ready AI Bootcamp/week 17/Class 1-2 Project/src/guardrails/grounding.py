"""Guardrail 4 — Abstention + citation verification (the LLM's answer is checked by code, not trusted).

HIGHLIGHTS:
* দুই ধাপে Abstention: (১) retrieval-এর সবচেয়ে কাছের passage-ও ``max_distance``-এর বাইরে হলে
  LLM ডাকাই হয় না; (২) LLM উত্তর দিলে code দেখে ``[n]`` citation আসল কিনা (১ থেকে passage
  সংখ্যার মধ্যে)। একটাও valid citation না থাকলে উত্তরটা user-এর কাছে যায় না — Abstention।
* ভুল নম্বর (যেমন ৪টা passage-এ ``[7]``) text থেকে মুছে ফেলা হয় আর ``flags``-এ লেখা থাকে, যাতে
  eval আর monitoring গুনতে পারে model কত ঘন ঘন citation বানায়।
* Abstention ব্যর্থতা না — "আমার handbook-এ এটা নেই" একটা সঠিক, নিরাপদ উত্তর।

ponytail: শুধু নম্বর যাচাই হয়, বাক্যটা সত্যিই passage-এ আছে কিনা না। Ceiling: sentence-level support
check (Week 14-এর content-word overlap বা NLI model)।
"""
from __future__ import annotations

import re

NOT_IN_KB = ("I couldn't find this in the Greenfield Community Clinic handbook, so I won't guess. "
             "Please ask a clinician, the clinic pharmacist or the front desk.")
MODEL_ABSTAIN_TOKEN = "NOT_IN_KB"     # the system prompt tells the model to reply with exactly this

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def verify_citations(answer: str, n_passages: int) -> tuple[list[int], str, int]:
    """Return (valid passage numbers in order of first use, answer with invalid markers removed, invalid count)."""
    valid: list[int] = []
    invalid = 0

    def fix(m: re.Match) -> str:
        nonlocal invalid
        nums = [int(x) for x in m.group(1).split(",")]
        ok = [x for x in nums if 1 <= x <= n_passages]
        invalid += len(nums) - len(ok)
        valid.extend(ok)
        return f"[{', '.join(map(str, ok))}]" if ok else ""

    cleaned = re.sub(r"\s+([.,;])", r"\1", _CITE.sub(fix, answer)).strip()
    return list(dict.fromkeys(valid)), cleaned, invalid
