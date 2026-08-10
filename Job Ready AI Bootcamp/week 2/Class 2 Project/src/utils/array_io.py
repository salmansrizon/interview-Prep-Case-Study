"""
array_io.py
-----------
Motive: All disk I/O for the similarity engine lives here, nowhere else.
WHY: If saving ever changes (npy -> npz -> parquet), exactly one file changes.
WHAT IT DOES: Reads the text corpus, saves/loads NumPy arrays, writes JSON reports.
ANALOGY: The loading dock of a warehouse. Everything entering or leaving the
         building passes through this one door, so you can inspect it all here.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

PathLike = Union[str, Path]


def read_corpus(file_path: PathLike) -> List[str]:
    """
    Reads the corpus, one document per line.

    WHY strip blank lines? An empty document embeds to an all-zero vector,
    whose norm is 0 — and dividing by 0 poisons the whole similarity matrix
    with NaN. Cheaper to drop it here than to debug NaN later.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {path}\n"
            f"  Expected a text file with one document per line."
        )

    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    if not docs:
        raise ValueError(f"Corpus {path} is empty — nothing to embed.")

    return docs


def save_array(arr: np.ndarray, file_path: PathLike) -> None:
    """
    Saves an ndarray to .npy (NumPy's own binary format).

    WHY .npy and not CSV? CSV stores numbers as text: it is ~5x larger,
    loses the exact float bits, and forgets the dtype and shape. .npy keeps
    all three and loads by memory-mapping the raw bytes.
    ANALOGY: CSV = describing a photo in words. .npy = keeping the photo.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_array(file_path: PathLike) -> np.ndarray:
    """Loads an ndarray previously written by save_array()."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Array not found: {path}. Run main.py first.")
    return np.load(path)


def _json_default(obj: Any) -> Any:
    """
    Converts NumPy scalars into plain Python types.

    WHY? NumPy reductions return np.float32/np.int64, and json.dump rejects
    them with "Object of type float32 is not JSON serializable". Every report
    write would crash without this adapter.
    ANALOGY: A travel plug. The appliance is fine, the socket is fine —
             they just need one adapter between them.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_json_report(data: Dict[str, Any], file_path: PathLike) -> None:
    """Writes a report as pretty-printed JSON, creating folders as needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
