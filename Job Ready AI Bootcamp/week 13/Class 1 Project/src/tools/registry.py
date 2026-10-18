"""Tool registry: JSON schemas shown to the model + dispatcher."""

from __future__ import annotations

from pathlib import Path

from src.tools.calculator import calculate
from src.tools.filesystem import list_files, read_file

# HIGHLIGHTS: model কখনো function দেখে না, শুধু এই JSON schema দেখে। তাই
# "description" ই আসলে prompt — কখন কোন tool ডাকতে হবে সেটা এখান থেকেই শেখে।
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate a math expression exactly, e.g. '(12+8)*3'. Use for ANY arithmetic.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "math expression"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files in a directory inside the sandbox.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative dir, default '.'"}}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file inside the sandbox.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative file path"}},
            "required": ["path"]}}},
]


def run_tool(name: str, args: dict, sandbox: Path, max_chars: int = 4000) -> str:
    """Execute a tool by name. Never raises: errors become strings for the model.

    HIGHLIGHTS: model ভুল নাম বা ভুল argument পাঠাতে পারে — crash না করে error
    string ফেরত দিলে পরের turn-এ model নিজেই সেটা শুধরে নেয়।
    """
    try:
        if name == "calculate":
            return calculate(str(args["expression"]))
        if name == "list_files":
            return list_files(sandbox, str(args.get("path", ".")))
        if name == "read_file":
            return read_file(sandbox, str(args["path"]), max_chars)
        return f"Error: unknown tool {name}"
    except (KeyError, TypeError) as exc:
        return f"Error: bad arguments ({exc})"
