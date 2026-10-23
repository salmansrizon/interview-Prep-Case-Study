"""Read-only file tools confined to one sandbox directory."""

from __future__ import annotations

from pathlib import Path

from config import SAFETY


def _resolve(sandbox: Path, rel: str) -> Path:
    """Resolve ``rel`` inside ``sandbox`` or raise PermissionError.

    HIGHLIGHTS: ক্রমটা গুরুত্বপূর্ণ। ``is_relative_to()`` শুধু path-এর
    text তুলনা করে, ``..`` বোঝে না — তাই আগে ``resolve()`` (যা ``..``
    collapse করে আর symlink follow করে), তারপর check। শুধু string-এ ".."
    খোঁজা যথেষ্ট না: absolute path আর symlink সেটা বাইপাস করে।
    """
    root = sandbox.resolve()
    target = (root / rel).resolve()  # absolute rel দিলে root বাদ পড়ে, নিচে ধরা পড়ে
    if not target.is_relative_to(root):
        raise PermissionError("path escapes sandbox")
    # Hidden files (.env, .git/) inside the sandbox are secrets, not data.
    if any(part.startswith(".") for part in target.relative_to(root).parts):
        raise PermissionError("hidden files are not accessible")
    return target


def list_files(sandbox: Path, path: str = ".") -> str:
    try:
        d = _resolve(sandbox, path)
        if not d.is_dir():
            return f"Error: not a directory: {path}"
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir()
                       if not p.name.startswith("."))
        return "\n".join(names) or "(empty)"
    except PermissionError as exc:
        return f"Error: {exc}"


def read_file(sandbox: Path, path: str, max_chars: int = SAFETY.max_file_chars) -> str:
    try:
        f = _resolve(sandbox, path)
        if not f.is_file():
            return f"Error: file not found: {path}. Call list_files to see what exists."
        if f.suffix.lower() not in SAFETY.allowed_extensions:
            return f"Error: file type {f.suffix or '(none)'} not allowed; allowed: {', '.join(SAFETY.allowed_extensions)}"
        if f.stat().st_size > SAFETY.max_file_bytes:
            return f"Error: file larger than {SAFETY.max_file_bytes} bytes"
        text = f.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars] + ("\n...[truncated]" if len(text) > max_chars else "")
    except PermissionError as exc:
        return f"Error: {exc}"
