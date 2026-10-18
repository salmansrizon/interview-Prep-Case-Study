"""Safe math evaluator: AST walk, no eval()."""

from __future__ import annotations

import ast
import math
import operator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log,
          "sin": math.sin, "cos": math.cos}
_CONSTS = {"pi": math.pi, "e": math.e}


def _ev(n: ast.AST) -> float:
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.Name) and n.id in _CONSTS:
        return _CONSTS[n.id]
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        l, r = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and (abs(r) > 100 or abs(l) > 1e6):
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](l, r)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in _FUNCS and not n.keywords):
        return _FUNCS[n.func.id](*[_ev(a) for a in n.args])
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a math expression; return the result or an 'Error: ...' string.

    HIGHLIGHTS: eval() কখনো না — model-এর লেখা string arbitrary code চালাতে
    পারত। AST whitelist-এ শুধু সংখ্যা, operator আর কয়েকটা math function আছে।
    Error exception না হয়ে string হিসেবে ফেরত যায়, যাতে model নিজে ভুল দেখে
    শুধরে নিতে পারে।
    """
    try:
        return str(_ev(ast.parse(expression.strip(), mode="eval").body))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return f"Error: {exc}"
