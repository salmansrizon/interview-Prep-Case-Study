"""Generated code + tests একটা আলাদা Python subprocess-এ চালানো।"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_code(text: str) -> str:
    """LLM-এর ```python fence থেকে শুধু code বের করে; fence না থাকলে পুরোটাই।"""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def run_tests(code: str, tests: str, timeout: int = 10) -> tuple[bool, str]:
    """(passed, output) ফেরত দেয়।

    HIGHLIGHTS: `-I` (isolated mode) + temp dir + timeout। Timeout না থাকলে
    LLM-এর লেখা `while True` পুরো app আটকে দেবে।
    ponytail: এটা সত্যিকারের sandbox না (file/network বন্ধ নয)। Production-এ Docker লাগবে।
    """
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "solution_check.py"
        f.write_text(code + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, "-I", str(f)],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: tests {timeout}s-এর মধ্যে শেষ হয়নি (infinite loop?)"
    out = (p.stdout + p.stderr).strip()
    return p.returncode == 0, out[-1500:]
