"""
skill_check.py
--------------
Motive: Grade the student's answers in src/exercises.py automatically.
WHY: A lecture you only read is a lecture you only think you understood.
     This turns the notebook into something with a pass/fail answer.
WHAT IT DOES: Runs each exercise against randomised hidden test cases,
              checks the numbers AND checks the source for Python loops,
              then prints a scorecard.
ANALOGY: A driving examiner. Arriving at the destination is not enough —
         they also watch HOW you drove there.
"""
# The exercises module is assembled at runtime by _load_exercises(), so static
# analysis cannot see its exercise_* functions. Those lookups are correct.
# pylint: disable=no-member

import inspect
import linecache
import re
import types
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

EXERCISES_PATH = Path(__file__).resolve().parent / "exercises.py"


def _load_exercises() -> Any:
    """
    Loads exercises.py by compiling its text, fresh, on every run.

    WHY NOT A PLAIN `import`? Python caches compiled bytecode in __pycache__
    and decides the cache is still valid by comparing the source's timestamp
    AND its size. Change a file within the same second without changing its
    length — flipping a `+` to a `-` is exactly that — and the stale cache
    still looks valid. You would then be graded on your PREVIOUS answer and
    have no way to tell. Compiling the text ourselves skips the cache
    entirely, so the scorecard always reflects the file on disk.

    RETURN TYPE: `Any`, not `ModuleType`. The module is assembled at runtime,
    so no static checker can know which exercise_* functions live on it.

    ANALOGY: reading the student's actual exam paper instead of the
             photocopy you happened to take earlier.
    """
    if not EXERCISES_PATH.exists():
        raise FileNotFoundError(f"Cannot find exercises at {EXERCISES_PATH}")

    source = EXERCISES_PATH.read_text(encoding="utf-8")
    linecache.checkcache(str(EXERCISES_PATH))  # so inspect.getsource() is fresh too

    module = types.ModuleType("exercises_fresh")
    module.__file__ = str(EXERCISES_PATH)
    # exec is the whole point here — see the docstring above. It is running the
    # student's own file from their own project, not untrusted input.
    exec(compile(source, str(EXERCISES_PATH), "exec"), module.__dict__)  # pylint: disable=exec-used
    return module


# Status constants
PASS = "PASS"
LOOP = "LOOP"       # right answer, but written with a Python loop
FAIL = "FAIL"       # wrong answer
TODO = "TODO"       # not attempted yet


def _uses_python_loop(fn: Callable) -> bool:
    """
    Looks for a `for`/`while` loop in the function's own source.

    WHY source inspection instead of timing? Timing is noisy on a laptop and
    a loop over 10 elements is fast enough to hide. The source never lies.
    LIMITATION: this only reads the function body, so a loop hidden inside a
    helper you call elsewhere is not detected. That is fine — the point is to
    train the habit, not to build an escape-proof cage.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False

    # Strip the docstring so its prose cannot trigger a false positive
    source = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    source = re.sub(r"'''.*?'''", "", source, flags=re.DOTALL)
    # Drop comments too
    source = re.sub(r"#.*", "", source)

    return bool(re.search(r"\b(for|while)\b", source))


def _check(name: str, fn: Callable, runner: Callable[[], bool], hint: str) -> Dict[str, Any]:
    """Runs one exercise's test, catching the not-yet-attempted case."""
    try:
        correct = runner()
    except NotImplementedError:
        return {"name": name, "status": TODO, "detail": "not attempted yet", "hint": hint}
    # Catch EVERYTHING on purpose: a half-finished answer can raise any
    # exception at all, and one broken exercise must not take down the grader.
    except Exception as e:  # pylint: disable=broad-exception-caught
        detail = f"{type(e).__name__}: {e}"
        return {"name": name, "status": FAIL, "detail": detail[:70], "hint": hint}

    if not correct:
        return {"name": name, "status": FAIL, "detail": "returned the wrong values", "hint": hint}
    if _uses_python_loop(fn):
        return {"name": name, "status": LOOP,
                "detail": "correct, but uses a Python loop", "hint": hint}
    return {"name": name, "status": PASS, "detail": "correct and vectorized", "hint": ""}


# ----------------------------------------------------------------------
# The hidden test cases
# ----------------------------------------------------------------------
def _run_all_tests(seed: int = 0) -> List[Dict[str, Any]]:
    ex = _load_exercises()          # always the file as it is on disk right now
    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []

    # -- 1 --------------------------------------------------------------
    def t1() -> bool:
        out = ex.exercise_1_make_batch(4, 7)
        return (
            isinstance(out, np.ndarray)
            and out.shape == (4, 7)
            and out.dtype == np.float32
            and bool(np.all(out == 0))
        )
    results.append(_check("1. make_batch (shape + dtype)", ex.exercise_1_make_batch, t1,
                          "np.zeros((n, m), dtype=np.float32)"))

    # -- 2 --------------------------------------------------------------
    def t2() -> bool:
        batch = rng.standard_normal((5, 3))
        bias = rng.standard_normal(3)
        out = ex.exercise_2_add_bias(batch, bias)
        return out.shape == (5, 3) and np.allclose(out, batch + bias)
    results.append(_check("2. add_bias (broadcasting)", ex.exercise_2_add_bias, t2,
                          "batch + bias — broadcasting right-aligns the shapes"))

    # -- 3 --------------------------------------------------------------
    def t3() -> bool:
        batch = rng.standard_normal((4, 4))       # square on purpose: hides axis mistakes
        scale = np.array([1.0, 10.0, 100.0, 1000.0])
        out = ex.exercise_3_scale_rows(batch, scale)
        return out.shape == (4, 4) and np.allclose(out, batch * scale[:, None])
    results.append(_check("3. scale_rows (new axis)", ex.exercise_3_scale_rows, t3,
                          "batch * scale[:, None] — give scale a second axis"))

    # -- 4 --------------------------------------------------------------
    def t4() -> bool:
        batch = rng.standard_normal((6, 3))
        out = ex.exercise_4_feature_means(batch)
        return np.shape(out) == (3,) and np.allclose(out, batch.mean(axis=0))
    results.append(_check("4. feature_means (axis)", ex.exercise_4_feature_means, t4,
                          "batch.mean(axis=0) — collapse the sample axis"))

    # -- 5 --------------------------------------------------------------
    def t5() -> bool:
        m = rng.standard_normal((5, 4)) * 10
        out = ex.exercise_5_normalize_rows(m)
        return out.shape == (5, 4) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    results.append(_check("5. normalize_rows (keepdims)", ex.exercise_5_normalize_rows, t5,
                          "m / np.linalg.norm(m, axis=1, keepdims=True)"))

    # -- 6 --------------------------------------------------------------
    def t6() -> bool:
        m = rng.standard_normal((7, 5))
        thr = 0.25
        out = ex.exercise_6_count_above(m, thr)
        return int(out) == int((m > thr).sum())
    results.append(_check("6. count_above (boolean mask)", ex.exercise_6_count_above, t6,
                          "int((matrix > threshold).sum())"))

    # -- 7 --------------------------------------------------------------
    def t7() -> bool:
        m = rng.standard_normal((6, 8))
        out = ex.exercise_7_pairwise_cosine(m)
        unit = m / np.linalg.norm(m, axis=1, keepdims=True)
        return (
            out.shape == (6, 6)
            and np.allclose(np.diag(out), 1.0, atol=1e-6)
            and np.allclose(out, unit @ unit.T, atol=1e-6)
        )
    results.append(_check("7. pairwise_cosine (matmul)", ex.exercise_7_pairwise_cosine, t7,
                          "normalize the rows, then unit @ unit.T"))

    # -- 8 --------------------------------------------------------------
    def t8() -> bool:
        scores = rng.standard_normal(12)
        k = 4
        out = np.asarray(ex.exercise_8_top_k(scores, k))
        expected = np.argsort(-scores)[:k]
        return out.shape == (k,) and np.array_equal(out, expected)
    results.append(_check("8. top_k (argsort)", ex.exercise_8_top_k, t8,
                          "np.argsort(-scores)[:k] — negate to sort descending"))

    return results


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def run_skill_check(seed: int = 0) -> Dict[str, Any]:
    """Runs every exercise test and returns a structured scorecard."""
    results = _run_all_tests(seed)
    counts = {s: sum(1 for r in results if r["status"] == s) for s in (PASS, LOOP, FAIL, TODO)}
    total = len(results)
    # A looping answer is worth half — the maths is right, the NumPy is not
    score = counts[PASS] + 0.5 * counts[LOOP]

    return {
        "results": results,
        "counts": counts,
        "total": total,
        "score": score,
        "pct": round(100.0 * score / total, 1) if total else 0.0,
        "all_passed": counts[PASS] == total,
    }


def print_scorecard(card: Dict[str, Any]) -> None:
    """Prints the scorecard as a readable table."""
    symbol = {PASS: "✅", LOOP: "🟡", FAIL: "❌", TODO: "⬜"}

    print("\n" + "=" * 70)
    print("SKILL CHECK — src/exercises.py")
    print("=" * 70)

    for r in card["results"]:
        print(f"  {symbol[r['status']]} {r['name']:<34} {r['detail']}")
        if r["hint"] and r["status"] != PASS:
            print(f"       ↳ hint: {r['hint']}")

    c = card["counts"]
    print("-" * 70)
    print(f"  SCORE: {card['score']:g} / {card['total']}  ({card['pct']}%)")
    print(f"  ✅ {c[PASS]} vectorized   🟡 {c[LOOP]} looping   "
          f"❌ {c[FAIL]} wrong   ⬜ {c[TODO]} untouched")

    if card["all_passed"]:
        print("\n  🎉 Full marks. You can build a similarity engine from scratch.")
    elif c[TODO] == card["total"]:
        print("\n  👉 Open src/exercises.py and replace each `raise` with your answer.")
    else:
        print("\n  👉 Keep going — fix the ❌ and 🟡 rows, then run again.")
    print("=" * 70)
