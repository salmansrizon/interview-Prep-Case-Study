"""Guardrail 3 — refuse personal diagnosis and individualised dosing (rule-based, before the LLM).

HIGHLIGHTS:
* "আমার কি diabetes?" বা "কত mg খাব?" — এর উত্তর দিতে রোগীর history, ওজন, অন্য ওষুধ, test লাগে।
  Handbook-এ সেটা নেই, model-এরও নেই। তাই rule মিললে LLM ডাকা হয় না: একটা স্থির refusal +
  clinician/pharmacist-এর কাছে যাওয়ার পরামর্শ।
* Refusal মানে খালি হাতে ফেরানো না: assistant তবুও retrieval চালায় আর threshold-এর ভেতরের
  handbook section-গুলো "সাধারণ তথ্য" হিসেবে citation-এ দেয় (যেমন OTC label পড়ার নিয়ম)।
  এই অংশের লেখাও code-এর, model-এর না — তাই refusal কখনো ঘুরপথে dose বলে ফেলে না।
* Rule-এর ভিত্তি first-person ("do I have", "should I take", "my dose")। "What is diabetes?"
  সাধারণ প্রশ্ন, refusal না।

ponytail: phrase list — "is 11 high for me?" ধরনের পরোক্ষ প্রশ্ন বাদ পড়তে পারে। Ceiling: ছোট labelled
classifier; eval set (data/eval) বাড়িয়ে miss মাপুন।
"""
from __future__ import annotations

import re

REFUSAL_MESSAGE = (
    "I can't diagnose a condition or recommend a medicine or dose for you personally. That needs a clinician "
    "or pharmacist who knows your history, other medicines and test results. Please book an appointment or a "
    "telehealth visit at Greenfield Community Clinic, or ask the clinic pharmacist. If your symptoms are severe "
    "or getting worse, seek urgent care.")
RELATED_NOTE = " General information from the clinic handbook that may help is listed below."

_I = re.I
_DRUG_UNIT = r"(?:mg|milligrams?|ml|mls|tablets?|pills?|capsules?|doses?|units?|puffs?|drops?)"
RULES: list[tuple[str, re.Pattern]] = [
    ("personal_diagnosis", re.compile(
        r"\bdo (?:i|you think i) have\b|\bhave i got\b|\bam i (?:sick|ill|diabetic|having|pregnant)\b|"
        r"\bmy (?:\w+ )?(?:sugar|pressure|bp|hba1c|cholesterol|levels?|readings?|results?) (?:is|was|are|were) \d|"
        r"what(?:'s| is) wrong with (?:me|my)|diagnose (?:me|my)|what (?:disease|illness|condition|infection) do i have|"
        r"\b(?:is|could) (?:it|this|my \w+) (?:be )?(?:cancer|serious|diabetes|dengue|an? infection)\b|"
        r"what (?:do|does) my (?:\w+ ){0,2}(?:results?|reports?|readings?|tests?|scans?|x-rays?) (?:mean|show|say)|"
        r"\b(?:is|are) my (?:\w+ )?(?:results?|readings?|levels?|sugar|pressure) (?:normal|ok|okay|bad|high|low|dangerous)", _I)),
    ("individual_dosing", re.compile(
        rf"how (?:many|much) {_DRUG_UNIT}?\s*(?:of \S+ )?(?:should|can|do|must|could) (?:i|we|he|she|my)\b|"
        rf"how (?:many|much) \S+ (?:should|can|do) (?:i|we) (?:take|give|use)|"
        r"what (?:dose|dosage|amount) (?:should|can|do) (?:i|we|he|she|my)|"
        r"\b(?:dose|dosage) for (?:me|my|a \d+)|"
        r"can i (?:double|increase|decrease|reduce|skip|stop|change|split) (?:my |the )?(?:\w+ ){0,3}"
        r"(?:dose|dosage|medicine|medication|tablets?|pills?|insulin)|"
        r"should i (?:stop|start|switch|increase|reduce) (?:taking\b|(?:my |the )?(?:\w+ )?"
        r"(?:medicine|medication|tablets?|pills?|insulin|dose|dosage|antibiotics?|inhaler))|"
        r"can i (?:take|give|mix) \S+(?: \S+)? (?:with|and|together)|"
        rf"\d+\s*{_DRUG_UNIT}\s+(?:for|to) (?:my|a) \d", _I)),
]


def check(text: str) -> str | None:
    """Return 'personal_diagnosis' / 'individual_dosing' when the question asks for individual advice, else None."""
    for rule, rx in RULES:
        if rx.search(text):
            return rule
    return None
