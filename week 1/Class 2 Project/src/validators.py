"""
validators.py
-------------
Motive: Ensure cleaned data matches an expected schema before it reaches a model.
WHY: LLMs and ML models expect specific fields. A missing 'text' field or a 
     string where an int should be will crash inference or corrupt training.
WHAT IT DOES: Uses jsonschema to validate each record against config/schema.json.
ANALOGY: This is the *final inspector* at the factory. Before boxes ship,
         they are weighed, measured, and checked against the order sheet.
"""

import json
from jsonschema import validate, ValidationError
from typing import List, Dict, Any, Tuple


def load_schema(path: str = "config/schema.json") -> Dict[str, Any]:
    """
    Loads the JSON schema from disk.

    WHY separate function? Schema loading is a one-time cost. 
    We load it once and pass it around, not re-read it for every record.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_records(
    records: List[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits records into valid and invalid buckets.

    RETURNS: (valid_records, invalid_records)

    WHY not crash on first invalid record? In production, you want to process
    99% of good data and quarantine 1% of bad data, not throw away everything.

    ANALOGY: A grape sorting machine. Good grapes go to the wine press;
             rotten grapes go to compost. Neither stops the line.
    """
    valid = []
    invalid = []

    for record in records:
        try:
            validate(instance=record, schema=schema)
            valid.append(record)
        except ValidationError as e:
            # WHY store the error message? So engineers can debug WHY it failed.
            record["_validation_error"] = str(e.message)
            invalid.append(record)

    return valid, invalid
