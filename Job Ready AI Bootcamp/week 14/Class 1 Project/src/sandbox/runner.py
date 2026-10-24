"""
Run generated code against tests in a separate, limited Python process.

HIGHLIGHTS:
* প্রতিটা test আলাদা করে চলে আর ফলাফল JSON-এ ফেরত আসে — তাই feedback হয়
  "``roman_to_int('IV')`` returned 6, expected 4", শুধু একটা traceback না।
  Self-Debugging paper-এ unit-test feedback-ই সবচেয়ে বড় উন্নতি দিয়েছে।
* Isolation: আলাদা process, ``python -I`` (user site-packages আর PYTHON* env
  বাদ), খালি temp dir, wall-clock timeout, আর POSIX-এ ``RLIMIT_CPU`` /
  ``RLIMIT_AS`` / ``RLIMIT_FSIZE``।
* ponytail: এটা আসল sandbox না — network আর filesystem read বন্ধ না।
  Untrusted code production-এ Docker (``--network none``), gVisor বা
  Firecracker/e2b-তে চালান।
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_MARKER = "__RESULTS__"

# Runs inside the child process. Tests are OUR trusted strings; only solution.py is untrusted.
_HARNESS = r'''
import json, runpy, traceback
results = []
try:
    # `python -I` removes the script's folder from sys.path, so load the file directly.
    ns = runpy.run_path("solution.py", run_name="solution")
    load_error = None
except BaseException as e:
    ns, load_error = {}, f"{type(e).__name__}: {e}"
TESTS = json.loads(open("tests.json").read())
for call, expected in TESTS:
    if load_error:
        results.append({"call": call, "ok": False, "got": None, "error": "code failed to load: " + load_error})
        continue
    try:
        got = eval(call, dict(ns))
        if expected.startswith("raises "):
            results.append({"call": call, "ok": False, "got": repr(got), "error": f"expected {expected}, but returned {got!r}"})
        else:
            want = eval(expected)
            ok = got == want and type(got) == type(want)
            results.append({"call": call, "ok": ok, "got": repr(got),
                            "error": None if ok else f"returned {got!r} ({type(got).__name__}), expected {want!r}"})
    except BaseException as e:
        exp_exc = expected.split(" ", 1)[1] if expected.startswith("raises ") else None
        ok = exp_exc is not None and type(e).__name__ == exp_exc
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append({"call": call, "ok": ok, "got": None,
                        "error": None if ok else f"raised {type(e).__name__}: {e}" + ("" if exp_exc is None else f" (expected {exp_exc})")})
print("__RESULTS__" + json.dumps(results))
'''


@dataclass
class TestReport:
    results: list[dict] = field(default_factory=list)
    crashed: str = ""           # timeout / harness crash, if any

    @property
    def passed(self) -> int:
        return sum(r["ok"] for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return not self.crashed and self.total > 0 and self.passed == self.total

    def feedback(self, max_chars: int = 2000) -> str:
        """Failing-test diff for the model — the most useful feedback signal."""
        if self.crashed:
            return self.crashed
        lines = [f"- {r['call']}: {r['error']}" for r in self.results if not r["ok"]]
        return (f"{self.passed}/{self.total} tests passed. Failing tests:\n" + "\n".join(lines))[:max_chars]


def _limits(cpu_seconds: int, memory_mb: int):
    def apply():  # runs in the child just before exec (POSIX only)
        import resource
        for res, val in ((resource.RLIMIT_CPU, cpu_seconds),
                         (resource.RLIMIT_AS, memory_mb * 1024 * 1024),
                         (resource.RLIMIT_FSIZE, 1024 * 1024)):
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass  # e.g. RLIMIT_AS is not enforceable on macOS
    return apply


def run_tests(code: str, tests: tuple[tuple[str, str], ...] | list, timeout: int = 10,
              cpu_seconds: int = 10, memory_mb: int = 512) -> TestReport:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "solution.py").write_text(code, encoding="utf-8")
        Path(d, "tests.json").write_text(json.dumps([list(t) for t in tests]), encoding="utf-8")
        Path(d, "harness.py").write_text(_HARNESS, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", "harness.py"], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": "/usr/bin:/bin"},
                               preexec_fn=_limits(cpu_seconds, memory_mb) if sys.platform != "win32" else None)
        except subprocess.TimeoutExpired:
            return TestReport(crashed=f"Timeout: tests did not finish in {timeout}s (infinite loop?).")
    out = p.stdout
    if _MARKER not in out:
        tail = (p.stderr or out).strip()[-800:]
        return TestReport(crashed=f"The code crashed the test run (exit {p.returncode}): {tail}")
    return TestReport(json.loads(out.split(_MARKER, 1)[1].strip().splitlines()[0]))
