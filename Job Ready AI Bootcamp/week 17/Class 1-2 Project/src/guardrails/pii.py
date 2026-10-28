"""Guardrail 1 — PII redaction. Runs FIRST, before anything is logged, embedded, sent to the LLM or returned.

HIGHLIGHTS:
* প্রশ্নে নাম, phone, email, NID, MRN, জন্মতারিখ থাকলে সেগুলো ``[PHONE]``, ``[EMAIL]`` ধরনের
  placeholder হয়ে যায়। এর পরের প্রতিটা ধাপ (rules, embedding, Chroma, prompt, log) শুধু redacted
  text দেখে — raw প্রশ্ন কোথাও লেখা হয় না।
* বাংলা অঙ্ক (০-৯): ``str.translate`` দিয়ে ASCII অঙ্কে বদলে regex চালানো হয়। প্রতিটা অঙ্ক একটা
  character-ই থাকে, তাই span হুবহু মেলে, আর redaction মূল text-এর উপরেই হয়।
* Bangladesh mobile: ``01[3-9]`` + ৮টা অঙ্ক, সামনে ``+880``/``880`` থাকতে পারে, মাঝে space বা dash।
* Overlap: আগের pattern জেতে (phone আগে, সাধারণ ID digit-run পরে), তাই একই অঙ্ক দুবার গোনা হয় না।

ponytail: regex মানে recall অসম্পূর্ণ — ঠিকানা, ছোট হাতের নাম ("i am rahim"), বানান ভুল ধরে না।
Production-এ NER model (যেমন Presidio) + এই regex একসাথে লাগে; README-র Known Limits দেখুন।
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_DATE = (r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}"
         r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+\d{4}")

# (label, pattern). A named group "pii" marks the part to redact; otherwise the whole match. Order = priority.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?880[\s-]?|0)1[3-9](?:[\s-]?\d){8}(?!\d)")),          # Bangladesh mobile
    ("PHONE", re.compile(r"\+\d{1,3}(?:[\s.-]?\d){7,12}(?!\d)")),                             # other international
    ("MRN", re.compile(r"\bGCC[\s-]?\d{6}\b", re.I)),
    ("MRN", re.compile(r"\b(?:mrn|medical record (?:number|no\.?)|patient id)\s*[:#]?\s*(?P<pii>[A-Z0-9][A-Z0-9-]{3,})",
                       re.I)),
    ("ID", re.compile(r"\b(?:nid|national id(?:entity)?(?: card)?(?: number| no\.?)?|passport(?: number| no\.?)?)"
                      r"\s*[:#]?\s*(?P<pii>[A-Z0-9]{6,17})", re.I)),
    ("DOB", re.compile(rf"\b(?:dob|d\.o\.b\.?|date of birth|born on|born|birthday)\s*(?:is|:|-)?\s*(?P<pii>{_DATE})",
                       re.I)),
    ("DATE", re.compile(rf"\b(?:{_DATE})\b", re.I)),
    ("ID", re.compile(r"(?<!\d)\d{9,17}(?!\d)")),                                              # NID-like digit runs
    # first word in any case ("my name is rahim"), further words only if Capitalised ("Rahim Uddin")
    ("NAME", re.compile(r"(?i:\b(?:my name is|name is|name:|i am called))\s+"
                        r"(?P<pii>[A-Za-z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2})")),
    ("NAME", re.compile(r"আমার নাম\s+(?P<pii>\S+(?:\s+\S+)?)")),
]


_PLACEHOLDER = re.compile(r"\[(?:%s)\]" % "|".join(sorted({label for label, _ in PATTERNS})))


def strip_placeholders(text: str) -> str:
    """Drop "[PHONE]"-style tokens before embedding: they carry no topic and pull retrieval off-topic
    (measured: "My email is [EMAIL]. What are the early signs of dehydration?" 0.643 -> 0.538, right section)."""
    return " ".join(_PLACEHOLDER.sub(" ", text).split())


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> Redaction:
    probe = text.translate(_BN_DIGITS)          # same length as text, so spans line up
    spans: list[tuple[int, int, str]] = []
    for label, rx in PATTERNS:
        for m in rx.finditer(probe):
            s, e = m.span("pii") if "pii" in rx.groupindex else m.span()
            if not any(s < pe and ps < e for ps, pe, _ in spans):
                spans.append((s, e, label))
    out, last = [], 0
    for s, e, label in sorted(spans):
        out += [text[last:s], f"[{label}]"]
        last = e
    out.append(text[last:])
    return Redaction("".join(out), dict(Counter(label for *_, label in spans)))
