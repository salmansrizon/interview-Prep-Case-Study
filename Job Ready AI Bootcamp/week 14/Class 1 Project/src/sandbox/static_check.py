"""
Static checks on generated code BEFORE running it.

HIGHLIGHTS: এটা defence-in-depth, sandbox না। Blocklist দুর্বল — Python-এ
`().__class__.__bases__[0].__subclasses__()` দিয়ে banned import ছাড়াই
অনেক কিছু পাওয়া যায় (Week 13-এর calculator lesson)। তাই আসল সুরক্ষা হলো
process isolation (runner.py) — আর production-এ Docker/gVisor/Firecracker।
এই check-এর আসল লাভ: syntax error আর "function-ই নেই" ধরনের ভুল ধরা,
code চালানোর আগেই, আর model-কে পরিষ্কার feedback দেওয়া।
"""
from __future__ import annotations

import ast
import re

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Return the LAST ```python block (models often explain first, then give the final code)."""
    blocks = _FENCE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def static_check(code: str, func_name: str, banned_imports: tuple[str, ...],
                 banned_calls: tuple[str, ...]) -> list[str]:
    """Return a list of problems (empty = OK to run)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError on line {e.lineno}: {e.msg}"]
    problems = []
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if func_name not in defined:
        problems.append(f"The code must define a function named `{func_name}` (found: {sorted(defined) or 'none'}).")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            names = []
        for n in names:
            if n in banned_imports:
                problems.append(f"Importing `{n}` is not allowed; use only pure Python (re, math, collections are fine).")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned_calls:
            problems.append(f"Calling `{node.func.id}()` is not allowed.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__") \
                and node.attr not in {"__init__", "__name__"}:
            problems.append(f"Dunder attribute access `{node.attr}` is not allowed.")
    return list(dict.fromkeys(problems))  # de-duplicate, keep order
