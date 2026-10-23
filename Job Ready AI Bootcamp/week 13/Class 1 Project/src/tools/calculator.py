"""Safe math evaluator: AST walk with a whitelist, never eval()."""

from __future__ import annotations

import ast
import math
import operator

from config import SAFETY

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
    # bool is a subclass of int — reject it explicitly so `True+True` is not math.
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.Name) and n.id in _CONSTS:
        return _CONSTS[n.id]
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        left, right = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and (abs(right) > SAFETY.max_exponent or abs(left) > 1e6):
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](left, right)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in _FUNCS and not n.keywords):
        return _FUNCS[n.func.id](*[_ev(a) for a in n.args])
    raise ValueError(f"unsupported expression element: {type(n).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a math expression; return the result or an 'Error: ...' string.

    HIGHLIGHTS: eval() কখনো না — LangChain-এর LLMMathChain (CVE-2023-29374)
    ঠিক এই ভুলে remote code execution খেয়েছিল। AST whitelist-এ শুধু সংখ্যা,
    operator আর কয়েকটা math function আছে; `__import__`, attribute access,
    string সব বাদ। Python docs সতর্ক করে যে ast.parse নিজেও বিশাল input-এ
    memory খেতে পারে, তাই আগে length cap। Error exception না হয়ে string
    হিসেবে ফেরত যায়, যাতে model পরের turn-এ নিজে শুধরে নিতে পারে।
    """
    expr = str(expression).strip()
    if len(expr) > SAFETY.max_expression_chars:
        return f"Error: expression longer than {SAFETY.max_expression_chars} chars"
    try:
        result = _ev(ast.parse(expr, mode="eval").body)
        if abs(result) > SAFETY.max_result_abs:
            return "Error: result too large"
        if isinstance(result, float) and result.is_integer():
            result = int(result)  # "15" reads better than "15.0" for the model
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError) as exc:
        return f"Error: {exc}"
