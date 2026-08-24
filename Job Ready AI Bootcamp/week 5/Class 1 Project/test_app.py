"""
One runnable check for the whole dashboard.

Loads every page headlessly and exercises the widgets that used to break:
the boxplot API, the diverging learning rates, and the predict button.

    python test_app.py          # or: pytest test_app.py
"""
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


def test_gradient_descent_handles_every_learning_rate():
    """Big alphas must report divergence, not overflow to inf and blank the charts."""
    at = run("pages/3_Gradient_Descent.py")
    converged, diverged = [], []
    for alpha in at.select_slider[0].options:
        at.select_slider[0].set_value(alpha).run()
        assert not at.exception, f"alpha={alpha}: {[str(e.message) for e in at.exception]}"
        hit = at.metric[2].value.endswith("Yes")
        blew_up = bool(at.error)
        assert not (hit and blew_up), f"alpha={alpha} cannot both converge and diverge"
        if hit:
            converged.append(alpha)
            assert abs(float(at.metric[0].value) - 2.5) < 0.1
        elif blew_up:
            diverged.append(alpha)
    # The demo is only useful if the slider actually shows both outcomes.
    assert converged and diverged, f"converged={converged} diverged={diverged}"


def test_prediction_is_a_plausible_price():
    at = run("pages/6_Predict_Price.py")
    at.button[0].click().run()
    assert not at.exception
    price_block = next(m.value for m in at.markdown if "Estimated Market Value" in m.value)
    price = float(price_block.split("💰 $")[1].split("<")[0].replace(",", ""))
    assert 50_000 < price < 1_500_000, f"implausible prediction: {price}"


def test_feature_selectboxes_cycle():
    for script in ["pages/1_Explore_Data.py", "pages/5_Feature_Scaling.py"]:
        at = run(script)
        for option in at.selectbox[0].options:
            at.selectbox[0].set_value(option).run()
            assert not at.exception, f"{script} [{option}]: {[str(e.message) for e in at.exception]}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nAll checks passed.")
