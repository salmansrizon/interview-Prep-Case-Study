"""
file_handler.py
---------------
Motive: Centralize all disk I/O operations.
WHY: If every module opens files directly, a path change means editing 10 files.
     A single utility means one point of change.
WHAT IT DOES: Reads raw text files, ensures output directories exist, writes sorted files.
ANALOGY: This is the *librarian* of the project. Instead of every student grabbing
         books from random shelves, the librarian knows exactly where everything is.
"""

import os
from pathlib import Path
from typing import List


def read_lines(file_path: str) -> List[str]:
    """
    Reads a text file and returns non-empty lines.

    WHY strip()? Raw files often have trailing newlines or spaces that break sorting.
    WHY filter? Empty lines add noise to datasets.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        # List comprehension: concise, fast, Pythonic.
        # WHY list comp vs for-loop? Less code, faster execution, standard in AI codebases.
        lines = [line.strip() for line in f if line.strip()]
    return lines


def ensure_dir(directory: str) -> None:
    """
    Creates a directory (and parents) if it does not exist.

    WHY exist_ok=True? Prevents crash if the folder already exists.
    WHAT IT DOES: Guarantees the output path is ready before we write.
    ANALOGY: Like checking that a folder exists in your filing cabinet before
             dropping documents into it.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def write_lines(file_path: str, lines: List[str]) -> None:
    """
    Writes a list of strings to a file, one per line.

    WHY \n.join()? More efficient than looping and writing one line at a time.
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
