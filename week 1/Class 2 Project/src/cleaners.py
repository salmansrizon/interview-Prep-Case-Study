"""
cleaners.py
-----------
Motive: Demonstrate advanced Python patterns (list comprehensions, lambda, map, filter)
        in a real-world data cleaning pipeline.
WHY: These are not academic toys. In AI Engineering, list comprehensions process
     millions of tokens; lambdas configure dynamic sorting; map/filter transform batches.
WHAT IT DOES: Provides pure functions that clean, filter, and transform raw records.
ANALOGY: This is the *quality control assembly line* in a factory. Each station
         (function) performs one precise task: remove dirt, check weight, apply label.
"""

from typing import List, Dict, Any, Callable, Optional


def strip_whitespace(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes leading/trailing whitespace from all string values in records.

    WHY list comprehension? 
      - It is 2-3x faster than equivalent for-loops in CPython.
      - It expresses "transform every item" more declaratively.

    WHAT IT DOES: Iterates every record, iterates every key-value pair,
                  strips strings, copies non-strings as-is.

    ANALOGY: Like a car wash dryer that blows water off every car that passes through.
    """
    return [
        {
            # Dict comprehension: transform each value based on its type.
            # WHY isinstance check? We only strip strings; stripping an int would crash.
            key: value.strip() if isinstance(value, str) else value
            for key, value in record.items()
        }
        for record in records  # Outer loop: every record in the list.
    ]


def remove_empty_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drops records that have no meaningful data.

    WHY filter + lambda? 
      - filter is lazy (memory efficient in Python 3).
      - lambda creates an anonymous function inline — perfect for one-off predicates.

    WHAT IT DOES: Keeps only records where the 'text' key exists and is non-empty.

    ANALOGY: A bouncer at a club. The lambda is the rule: "ID and non-empty name required."
             filter applies that rule to everyone in line.
    """
    # list() wraps filter because filter returns an iterator, not a list.
    # WHY? Students often forget this and try to index filter objects, causing TypeError.
    return list(filter(
        lambda r: r.get("text") and str(r.get("text")).strip(),
        records
    ))


def filter_by_status(
    records: List[Dict[str, Any]],
    allowed: List[str],
) -> List[Dict[str, Any]]:
    """
    Retains only records whose 'status' is in the allowed list.

    WHY? In production, you often ignore 'deleted' or 'banned' records before
         feeding data to a model. Garbage in, garbage out.
    """
    # Convert to set for O(1) lookup. 
    # WHY? "if x in list" is O(n); "if x in set" is O(1). At scale, this matters.
    allowed_set = set(allowed)
    return [r for r in records if r.get("status") in allowed_set]


def filter_by_length(
    records: List[Dict[str, Any]],
    min_len: int = 5,
    max_len: int = 500,
) -> List[Dict[str, Any]]:
    """
    Removes records with text too short (noise) or too long (costly for LLMs).

    WHY min/max? LLMs charge per token. A 10,000-character record costs 20x more
    than a 500-character one. Filtering early saves money.

    ANALOGY: An airport security scanner rejects bags that are too small (nothing inside)
             or too large (cannot fit through the X-ray).
    """
    return [
        r for r in records
        if min_len <= len(str(r.get("text", ""))) <= max_len
    ]


def sort_by_text_length(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts records by the length of their text field, shortest first.

    WHY sorted() + lambda? 
      - sorted() returns a new list (non-destructive).
      - lambda defines the sort key inline without polluting the namespace with a named function.

    WHAT IT DOES: For each record, lambda extracts len(record["text"]), and sorted()
                  arranges records by that number.

    ANALOGY: A librarian sorting books by page count. The lambda is the ruler
             that measures each book; sorted() places them on the shelf.
    """
    return sorted(records, key=lambda r: len(str(r.get("text", ""))))


def normalize_keys(
    records: List[Dict[str, Any]],
    lowercase: bool = False,
) -> List[Dict[str, Any]]:
    """
    Optionally lowercases all dictionary keys for consistency.

    WHY? APIs return "UserName", "user_name", "username". Normalizing prevents
    KeyError when different sources use different conventions.
    """
    if not lowercase:
        return records

    return [
        {key.lower(): value for key, value in record.items()}
        for record in records
    ]


def apply_cleaning_pipeline(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Chains all cleaning functions in a declarative pipeline.

    WHY pipeline? Each function is pure (no side effects) and composable.
    You can reorder, add, or remove steps by editing this function — no scattered changes.

    WHAT IT DOES:
      1. Strip whitespace
      2. Remove empty records
      3. Filter by allowed statuses
      4. Filter by text length
      5. Normalize keys
      6. Sort by length

    ANALOGY: An assembly line in a bottling plant. The bottle moves from station
             to station: rinse, fill, cap, label. Each station does one thing perfectly.
    """
    cleaning = config.get("cleaning", {})

    # Step 1: Strip whitespace
    if cleaning.get("strip_whitespace", True):
        records = strip_whitespace(records)

    # Step 2: Remove empty records
    if cleaning.get("remove_empty", True):
        records = remove_empty_records(records)

    # Step 3: Filter by status
    allowed = cleaning.get("allowed_statuses", [])
    if allowed:
        records = filter_by_status(records, allowed)

    # Step 4: Filter by length
    min_len = cleaning.get("min_text_length", 0)
    max_len = cleaning.get("max_text_length", float("inf"))
    if min_len or max_len != float("inf"):
        records = filter_by_length(records, min_len, max_len)

    # Step 5: Normalize keys
    if cleaning.get("lowercase_keys", False):
        records = normalize_keys(records, lowercase=True)

    # Step 6: Sort by length (useful for batching in LLM inference)
    records = sort_by_text_length(records)

    return records
