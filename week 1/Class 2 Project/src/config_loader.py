"""
config_loader.py
----------------
Motive: Centralize all JSON and YAML configuration loading.
WHY: If 5 modules each open config files, a path change means 5 edits.
     One loader = one point of truth.
WHAT IT DOES: Loads YAML/JSON, validates structure, returns typed Python objects.
ANALOGY: This is the *project manager* who reads the blueprint and hands out
         specific instructions to each construction crew.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """
    Loads and caches configuration files.

    WHY a class? Can hold cached configs in memory to avoid re-reading disk.
    WHAT IF we used raw open() everywhere? Slower, harder to mock in tests,
    and impossible to add validation later without refactoring.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Loads a YAML file safely.

        WHY yaml.safe_load? The unsafe loader can execute arbitrary Python code
        embedded in YAML. safe_load restricts to standard data types only.
        ANALOGY: safe_load is like a security guard who checks IDs at the door.
        """
        if filename in self._cache:
            return self._cache[filename]

        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._cache[filename] = data
        return data

    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        Loads a JSON file.

        WHY json.load vs json.loads? load reads from a file object; 
        loads parses a string. Using the right tool avoids unnecessary string buffering.
        """
        if filename in self._cache:
            return self._cache[filename]

        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._cache[filename] = data
        return data

    def get_parser_config(self) -> Dict[str, Any]:
        """
        Convenience method to extract the nested parser block.

        WHY? Callers should not need to know the YAML structure (model_config.yaml).
        This method isolates the "where" from the "what."
        """
        full = self.load_yaml("model_config.yaml")
        return full.get("parser", {})
