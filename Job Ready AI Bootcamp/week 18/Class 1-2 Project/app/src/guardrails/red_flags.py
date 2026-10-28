"""Guardrail 2 — Red-Flag Escalation. Deterministic rules, BEFORE retrieval and before any model call.

HIGHLIGHTS:
* Emergency-র সিদ্ধান্ত model নেয় না, code নেয়। কোনো rule মিললে retrieval বা LLM ডাকাই হয় না —
  সরাসরি একটা আগে থেকে লেখা, review করা "এখনই 999-এ call করুন" message। Model এই message-এর
  একটা শব্দও লেখে না, তাই hallucination বা "আশ্বাস দেওয়া" ভুলের সুযোগ নেই।
* Rule regex, তাই কয়েকটা বাংলা phrase-ও ধরা যায় ("বুকে ব্যথা", "শ্বাসকষ্ট") — retrieval
  English-only হলেও safety rule ভাষা-নিরপেক্ষ।
* Trade-off ইচ্ছাকৃত: false positive (অকারণে escalate) সস্তা, false negative (emergency মিস) দামি।
  তাই rule চওড়া; "heart attack-এর লক্ষণ কী?" ধরনের তথ্যের প্রশ্নও escalate হতে পারে।

ponytail: keyword rule — negation ("no chest pain") বা বানান ভুল বোঝে না। Ceiling: clinician-reviewed
symptom checker / classifier; তবুও এই rule-গুলো তার নিচে শেষ নিরাপত্তা জাল হিসেবে থাকবে।
"""
from __future__ import annotations

import re

EMERGENCY_MESSAGE = (
    "This may be a medical emergency. Call 999 now (the national emergency number in Bangladesh) or your "
    "local emergency number, or go to the nearest emergency department. If the person is unconscious, not "
    "breathing or bleeding heavily, get help immediately and do not wait for an appointment. "
    "This assistant cannot help in an emergency.")

SELF_HARM_MESSAGE = (
    "I'm really sorry you're going through this, and you deserve support right now. If you are thinking about "
    "suicide or harming yourself, or you are in danger, call 999 now (Bangladesh) or your local emergency "
    "number, or go to the nearest emergency department. If you can, tell someone you trust and stay with them. "
    "Greenfield Community Clinic also offers confidential counselling when you are safe.")

_I = re.I
RULES: list[tuple[str, re.Pattern]] = [
    ("self_harm", re.compile(r"suicid|kill (?:my|him|her|them)sel(?:f|ves)|end (?:my|his|her) (?:own )?life|"
                             r"self[- ]?harm|hurt(?:ing)? myself|want to die|don'?t want to (?:live|be alive)|"
                             r"আত্মহত্যা|নিজেকে শেষ|মরে যেতে চাই", _I)),
    ("chest_pain", re.compile(r"chest (?:pain|tightness|pressure|hurts)|pain in (?:my|his|her|the) chest|"
                              r"heart attack|বুকে ব্যথা", _I)),
    ("breathing", re.compile(r"(?:can'?t|cannot|unable to|struggling to|hard to) breathe|"
                             r"(?:difficulty|trouble|struggling) (?:breathing|to breathe)|gasping|"
                             r"severe(?:ly)? short(?:ness)? of breath|(?:lips|face) (?:is |are )?(?:turning )?(?:blue|grey)|"
                             r"শ্বাসকষ্ট|শ্বাস নিতে পারছ", _I)),
    ("stroke", re.compile(r"(?:face|mouth) (?:is )?droop|drooping (?:face|mouth)|slurred speech|"
                          r"sudden(?:ly)? (?:weak|numb|can'?t (?:speak|talk|see))|"
                          r"one side of (?:my|his|her|the) (?:body|face)|(?:having|had) a stroke|stroke symptoms", _I)),
    ("severe_bleeding", re.compile(r"(?:severe|heavy|heavily|uncontrolled|a lot of) bleeding|bleeding (?:heavily|a lot)|"
                                   r"(?:bleeding|blood) (?:won'?t|will not|doesn'?t|does not) stop|"
                                   r"(?:vomiting|coughing|throwing) up blood|blood in (?:my |his |her )?vomit|spurting", _I)),
    ("anaphylaxis", re.compile(r"anaphyla|(?:throat|tongue|lips?|face) (?:is |are )?(?:swelling|swollen|closing)|"
                               r"swelling (?:of|in) (?:the |my |his |her )?(?:throat|tongue|lips|face)|epipen", _I)),
    ("unconscious", re.compile(r"unconscious|unresponsive|passed out|fainted|collapsed|(?:won'?t|will not|can'?t) wake|"
                               r"seizure|convuls|অজ্ঞান|খিঁচুনি", _I)),
    ("poisoning", re.compile(r"(?:took|taken|had|swallowed) (?:an? )?overdose|overdosed|took too many (?:pills|tablets)|"
                             r"swallowed (?:bleach|poison|pills|tablets|batter(?:y|ies)|kerosene|chemicals?)|"
                             r"(?<!food )poison(?:ed|ing)", _I)),
    ("serious_burn", re.compile(r"(?:large|severe|deep|chemical|electrical) burn|burn(?:ed|t|s)? (?:on|to) (?:the |my |his |"
                                r"her )?(?:face|eyes?|airway)", _I)),
    ("meningitis_signs", re.compile(r"stiff neck|rash that (?:doesn'?t|does not|won'?t) fade", _I)),
]

# Fever in a baby under 3 months: two conditions, so a small function instead of one regex.
_INFANT = re.compile(r"newborn|\binfant|\b\d{1,2}[- ]?(?:day|week)s?[- ]?old|\b[0-2][- ]?months?[- ]?old|"
                     r"\b(?:one|two|1|2)[- ]month", _I)
_FEVER = re.compile(r"fever|temperature|\b3[89](?:\.\d)?\s*°?\s*c\b|জ্বর", _I)


def check(text: str) -> str | None:
    """Return the id of the first emergency rule that matches, or None."""
    for rule, rx in RULES:
        if rx.search(text):
            return rule
    if _INFANT.search(text) and _FEVER.search(text):
        return "infant_fever"
    return None


def message_for(rule: str) -> str:
    return SELF_HARM_MESSAGE if rule == "self_harm" else EMERGENCY_MESSAGE
