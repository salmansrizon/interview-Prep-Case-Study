"""Tool registry: Pydantic argument models -> JSON schemas for the model + a validating dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.tools.calculator import calculate
from src.tools.filesystem import list_files, read_file


class _Args(BaseModel):
    # extra="forbid": invented fields (e.g. {"a": 90, "op": "mean"}) are reported, not silently dropped.
    model_config = ConfigDict(extra="forbid")


class CalculateArgs(_Args):
    expression: str = Field(description="A math expression such as '(12+8)*3/4' or 'sqrt(16)'.")


class ListFilesArgs(_Args):
    path: str = Field(default=".", description="Directory relative to the sandbox root. Use '.' for the root.")


class ReadFileArgs(_Args):
    path: str = Field(description="File path relative to the sandbox root, e.g. 'notes.txt'.")


# HIGHLIGHTS: model কখনো Python function দেখে না, শুধু নাম + description +
# JSON schema দেখে। তাই "description"-ই আসলে prompt: কখন কোন tool ডাকতে
# হবে সেটা এখান থেকেই শেখে। Pydantic model একটাই source of truth — schema
# আর validation দুটোই এখান থেকে আসে, তাই দুটো আলাদা হয়ে যাওয়ার সুযোগ নেই।
_TOOLS: dict[str, tuple[type[_Args], str, Callable[..., str]]] = {
    "calculate": (
        CalculateArgs,
        "Evaluate a math expression exactly and return the number. Use this for ANY "
        "arithmetic, even simple sums — never compute numbers yourself. Supports + - * / // % **, "
        "parentheses, sqrt, abs, round, log, sin, cos, pi, e.",
        lambda a, sandbox: calculate(a.expression),
    ),
    "list_files": (
        ListFilesArgs,
        "List the files and folders inside a directory of the sandbox. Call this first when "
        "you do not know the exact file name.",
        lambda a, sandbox: list_files(sandbox, a.path),
    ),
    "read_file": (
        ReadFileArgs,
        "Read a UTF-8 text file (.txt, .csv, .md, .json) inside the sandbox and return its "
        "contents. Only files inside the sandbox can be read.",
        lambda a, sandbox: read_file(sandbox, a.path),
    ),
}


def _schema(name: str) -> dict:
    model, description, _ = _TOOLS[name]
    params = model.model_json_schema()
    params.pop("title", None)
    params.pop("additionalProperties", None)
    for prop in params.get("properties", {}).values():
        prop.pop("title", None)
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": params}}


# A concrete example in the error message helps small models more than the schema alone.
_EXAMPLES = {
    "calculate": '{"expression": "(90+75)/2"}',
    "list_files": '{"path": "."}',
    "read_file": '{"path": "notes.txt"}',
}

TOOL_SCHEMAS: list[dict] = [_schema(n) for n in _TOOLS]
TOOL_NAMES: tuple[str, ...] = tuple(_TOOLS)


def run_tool(name: str, args: dict | None, sandbox: Path) -> str:
    """Validate the model's arguments, then execute. Never raises.

    HIGHLIGHTS: model-এর দেওয়া argument কখনো বিশ্বাস করা যায় না — ভুল নাম,
    বাদ পড়া field, ভুল type সবই ঘটে। Pydantic আগে validate করে; ব্যর্থ হলে
    কী ভুল আর কী করতে হবে সেটা string হিসেবে model-এর কাছে ফেরত যায়
    (Anthropic-এর tool-design guide-এর পরামর্শ), যাতে পরের turn-এ সে নিজেই
    শুধরে নিতে পারে। Exception ছুড়লে পুরো loop crash করত।
    """
    if name not in _TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOL_NAMES)}"
    model, _, fn = _TOOLS[name]
    try:
        parsed = model.model_validate(args or {})
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
        return (f"Error: bad arguments for {name} ({problems}). "
                f"Expected exactly: {_EXAMPLES[name]}. Fix the arguments and call again.")
    return fn(parsed, sandbox)
