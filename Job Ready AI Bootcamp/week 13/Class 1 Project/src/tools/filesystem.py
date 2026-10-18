"""Read-only file tools confined to one sandbox directory."""

from __future__ import annotations

from pathlib import Path


def _resolve(sandbox: Path, rel: str) -> Path:
    """Resolve ``rel`` inside ``sandbox`` or raise PermissionError.

    HIGHLIGHTS: resolve() symlink আর ``..`` দুটোই সমাধান করে ফেলে, তারপর
    is_relative_to দিয়ে দেখা হয় ফলাফল sandbox-এর ভেতরেই কিনা। শুধু
    string-এ ".." খোঁজা যথেষ্ট না (symlink, absolute path বাইপাস করে)।
    """
    root = sandbox.resolve()
    target = (root / rel).resolve()  # absolute rel দিলে root বাদ পড়ে, নিচে ধরা পড়ে
    if not target.is_relative_to(root):
        raise PermissionError("path escapes sandbox")
    return target


def list_files(sandbox: Path, path: str = ".") -> str:
    try:
        d = _resolve(sandbox, path)
        if not d.is_dir():
            return f"Error: not a directory: {path}"
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir())
        return "\n".join(names) or "(empty)"
    except PermissionError as exc:
        return f"Error: {exc}"


def read_file(sandbox: Path, path: str, max_chars: int = 4000) -> str:
    try:
        f = _resolve(sandbox, path)
        if not f.is_file():
            return f"Error: file not found: {path}"
        text = f.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars] + ("\n...[truncated]" if len(text) > max_chars else "")
    except PermissionError as exc:
        return f"Error: {exc}"
