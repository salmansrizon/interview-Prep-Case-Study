"""
The four guardrails as pure functions: PII redaction, Red-Flag Escalation, diagnosis/dosing refusal, citation check.

HIGHLIGHTS: এগুলো deterministic, তাই model বা Chroma ছাড়াই exact আচরণ test করা যায় — বাংলা অঙ্কের
phone number, বাংলা emergency phrase, আর "সাধারণ প্রশ্নে ভুল করে refuse/escalate হচ্ছে না" সহ।
"""
import pytest

from src.guardrails import red_flags, scope
from src.guardrails.grounding import verify_citations
from src.guardrails.pii import redact, strip_placeholders


# ── 1. PII ────────────────────────────────────────────────────────────
def test_redacts_email_phone_mrn_and_name():
    r = redact("My name is Rahim Uddin, email rahim.uddin@example.com, MRN GCC-204518, call 01712-345678.")
    assert r.counts == {"NAME": 1, "EMAIL": 1, "MRN": 1, "PHONE": 1} and r.total == 4
    assert r.text == "My name is [NAME], email [EMAIL], MRN [MRN], call [PHONE]."
    assert strip_placeholders(r.text) == "My name is , email , MRN , call ."       # what the embedder sees


@pytest.mark.parametrize("phone", ["01712345678", "01712-345678", "017 1234 5678", "+8801712345678",
                                   "+880 1712 345678", "+880-1712-345678", "8801912345678"])
def test_bangladesh_mobile_formats(phone):
    r = redact(f"Text me on {phone} please")
    assert r.text == "Text me on [PHONE] please" and r.counts == {"PHONE": 1}


def test_bangla_digit_phone_is_redacted_and_bangla_text_kept():
    r = redact("আমার নম্বর ০১৮১২৩৪৫৬৭৮, appointment কীভাবে নেব?")
    assert r.text == "আমার নম্বর [PHONE], appointment কীভাবে নেব?" and r.counts == {"PHONE": 1}


def test_dob_national_id_and_dates():
    r = redact("DOB 12/03/1985, NID 1234567890123, my visit was on 2026-01-15")
    assert r.counts == {"DOB": 1, "ID": 1, "DATE": 1}
    assert "1985" not in r.text and "1234567890123" not in r.text and "2026-01-15" not in r.text


def test_bangla_name_pattern():
    assert redact("আমার নাম রহিম উদ্দিন").text == "আমার নাম [NAME]"


def test_clinical_numbers_are_not_pii():
    text = "My blood pressure was 140/90 and my 2 month old has 38.5 °C for 3 days"
    assert redact(text).text == text and redact(text).total == 0


# ── 2. Red flags ──────────────────────────────────────────────────────
@pytest.mark.parametrize("text,rule", [
    ("I have crushing chest pain", "chest_pain"),
    ("I can't breathe and my lips are turning blue", "breathing"),
    ("Her face is drooping and speech is slurred", "stroke"),
    ("The wound is bleeding heavily", "severe_bleeding"),
    ("I want to kill myself", "self_harm"),
    ("His throat is swelling after a bee sting", "anaphylaxis"),
    ("My grandmother collapsed and is unresponsive", "unconscious"),
    ("My 6 week old baby has a fever", "infant_fever"),
    ("My son swallowed bleach", "poisoning"),
    ("আমার বুকে ব্যথা করছে", "chest_pain"),
    ("আমার শ্বাসকষ্ট হচ্ছে", "breathing"),
])
def test_red_flag_rules(text, rule):
    assert red_flags.check(text) == rule


@pytest.mark.parametrize("text", ["What is the first aid for a minor burn?", "Is food poisoning contagious?",
                                  "How do I avoid an accidental overdose with cold remedies?",
                                  "My baby is 8 months old, which vaccines are next?"])
def test_red_flags_do_not_fire_on_routine_questions(text):
    assert red_flags.check(text) is None


def test_escalation_messages_are_fixed_and_name_999():
    assert red_flags.message_for("self_harm") == red_flags.SELF_HARM_MESSAGE
    assert red_flags.message_for("chest_pain") == red_flags.EMERGENCY_MESSAGE
    assert all("999" in m for m in (red_flags.SELF_HARM_MESSAGE, red_flags.EMERGENCY_MESSAGE))


# ── 3. Personal diagnosis / dosing ────────────────────────────────────
@pytest.mark.parametrize("text,rule", [
    ("Do I have diabetes?", "personal_diagnosis"),
    ("What's wrong with me?", "personal_diagnosis"),
    ("What do my blood test results mean?", "personal_diagnosis"),
    ("My blood sugar is 11, is that bad?", "personal_diagnosis"),
    ("How many mg of ibuprofen should I take?", "individual_dosing"),
    ("Can I double my dose of insulin?", "individual_dosing"),
    ("Can I double my blood pressure tablet?", "individual_dosing"),
    ("Can I take paracetamol with my antibiotics?", "individual_dosing"),
    ("Should I stop taking my metformin?", "individual_dosing"),
])
def test_refusal_rules(text, rule):
    assert scope.check(text) == rule


@pytest.mark.parametrize("text", ["What is type 2 diabetes?", "Should I stop smoking?",
                                  "How many days per week can I use telehealth?",
                                  "How should I store insulin?"])
def test_general_questions_are_not_refused(text):
    assert scope.check(text) is None


# ── 4. Citation check ─────────────────────────────────────────────────
def test_verify_citations_keeps_valid_and_strips_invented_numbers():
    valid, cleaned, invalid = verify_citations("Use ORS [1]. Keep feeding [2, 7]. Rest [9].", n_passages=2)
    assert valid == [1, 2] and invalid == 2
    assert cleaned == "Use ORS [1]. Keep feeding [2]. Rest."
    assert verify_citations("No citations at all.", 3)[0] == []
