"""
Evaluation runner (metrics math) + the bundled eval set run through the fake assistant + the PDF handbook build.

HIGHLIGHTS: Rule-ভিত্তিক route (escalated, refused) আর PII redaction embedder-নিরপেক্ষ, তাই fake
backend-এও সেগুলো ১০০% মিলতে হবে — data/eval-এ কেউ নতুন emergency প্রশ্ন যোগ করলে আর rule না মিললে
এই test fail করে। Answered/abstained route আসল embedder + Ollama-য় আপনার মেশিনে মাপতে হয়।
"""
from conftest import EVAL_FILE, KB_DIR
from samples.make_handbook import build
from src.assistant import Answer
from src.evaluation.runner import load_questions, run_eval
from src.guardrails.pii import Redaction
from src.rag.components import Hit
from src.rag.kb import load_kb


def _hit(i):
    return Hit(i, "p", "t", "x", 0.2)


def test_runner_metrics_on_scripted_answers():
    items = [
        {"id": "1", "question": "q1", "expected_route": "answered", "expected_source": ["kb:a:1"]},
        {"id": "2", "question": "q2", "expected_route": "escalated", "min_redactions": 1},
        {"id": "3", "question": "q3", "expected_route": "refused"},
        {"id": "4", "question": "q4", "expected_route": "abstained"},
        {"id": "5", "question": "boom", "expected_route": "answered"},
    ]
    scripted = {
        "q1": Answer("answered", "x [1]", Redaction("q1"), citations=[(1, _hit("kb:a:1"))], retrieved=[_hit("kb:a:1")],
                     timings_ms={"total_ms": 10}),
        "q2": Answer("answered", "unsafe!", Redaction("q2", {"PHONE": 1}), timings_ms={"total_ms": 20}),
        "q3": Answer("refused", "no", Redaction("q3"), timings_ms={"total_ms": 30}),
        "q4": Answer("abstained", "n/a", Redaction("q4"), retrieved=[_hit("kb:b:1")], timings_ms={"total_ms": 40}),
    }

    def ask(q):
        if q == "boom":
            raise ConnectionError("ollama down")
        return scripted[q]

    s = run_eval(ask, items)["summary"]
    assert s["n"] == 5 and s["route_accuracy"] == 0.6
    assert s["unsafe_answers"] == 1 and s["missed_escalations"] == 1
    assert s["retrieval_hit_rate"] == 1.0 and s["citation_hit_rate"] == 1.0 and s["pii_recall"] == 1.0
    assert s["confusion"] == {"escalated->answered": 1, "answered->error:ConnectionError": 1}
    assert s["per_route_recall"]["refused"] == {"n": 1, "recall": 1.0}


def test_bundled_eval_set_rule_routes_and_pii_are_exact(assistant):
    items = load_questions(EVAL_FILE)
    assert 30 <= len(items) <= 50
    kb_ids = {s.id for s in load_kb(KB_DIR)}
    assert all(set(it["expected_source"]) <= kb_ids for it in items if "expected_source" in it)  # labels exist
    report = run_eval(assistant.ask, items)
    for row in report["rows"]:
        if row["expected"] in ("escalated", "refused"):
            assert row["ok"], row
    s = report["summary"]
    assert s["unsafe_answers"] == 0 and s["missed_escalations"] == 0 and s["pii_recall"] == 1.0


def test_handbook_pdf_builds_from_the_markdown(tmp_path):
    out = tmp_path / "handbook.pdf"
    pages = build(out)
    assert out.read_bytes().startswith(b"%PDF-") and pages >= 31          # cover + one page per KB page
