"""
One runnable check for the whole dashboard.

Loads every page headlessly and exercises the parts that carry real logic:
the feature selectboxes, the three model choices, and the predict button.

    python test_app.py          # or: pytest test_app.py
"""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
TIMEOUT = 180


def run(script):
    at = AppTest.from_file(str(ROOT / script), default_timeout=TIMEOUT).run()
    assert not at.exception, f"{script}: {[str(e.message) for e in at.exception]}"
    return at


def test_every_page_loads():
    for script in ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py")):
        run(script)


def test_every_model_beats_a_coin_flip():
    """A dashboard about classification is pointless if the models cannot classify."""
    results = json.loads((ROOT / "models" / "model_results.json").read_text())
    assert set(results) == {"logistic_regression", "decision_tree", "random_forest"}
    for name, scores in results.items():
        assert scores["roc_auc"] > 0.8, f"{name} AUC={scores['roc_auc']:.3f} — retrain with train_model.py"


def test_prediction_responds_to_credit_score():
    """A weak applicant must not score the same as a strong one, on every model."""
    at = run("pages/5_Predict_Approval.py")
    for model in at.selectbox[0].options:
        at.selectbox[0].set_value(model).run()
        probabilities = []
        for score in (400, 820):
            at.slider[1].set_value(score).run()          # slider[0] is Age
            at.button[0].click().run()
            assert not at.exception, f"{model}: {[str(e.message) for e in at.exception]}"
            verdict = next(m.value for m in at.markdown if "Approval Probability" in m.value)
            probabilities.append(float(verdict.split("<strong>")[1].split("%")[0]))
        weak, strong = probabilities
        assert strong > weak, f"{model}: score 820 scored {strong}% vs {weak}% at 400"


def test_feature_selectboxes_cycle():
    at = run("pages/1_Explore_Data.py")
    for option in at.selectbox[0].options:
        at.selectbox[0].set_value(option).run()
        assert not at.exception, f"[{option}]: {[str(e.message) for e in at.exception]}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nAll checks passed.")
