"""
main.py
-------
Motive: The single entry point for the entire application.
WHY one entry point? Users should not hunt through src/ to find what to run.
    `python main.py` is the universal convention.
WHAT IT DOES: Loads config, initializes the sorter, executes sorting, prints results.
ANALOGY: This is the *ignition key* of the car. It does not do the driving,
        but it starts the engine, checks the fuel, and puts everything in motion.
"""

import os
import yaml
from src.text_sorter import TextSorter


def load_config(path: str = "config/settings.yaml") -> dict:
    """
    Loads YAML configuration safely.

    WHY yaml.safe_load? Prevents arbitrary code execution from malicious config files.
    WHAT IF we used json? JSON is fine, but YAML supports comments — critical for teaching.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if isinstance(config, dict) else {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def main():
    # 1. Load configuration.
    # WHY config file? Changing sort strategy should not require editing Python code.
    config = load_config()
    sort_config = config.get("sorting") or {}

    strategy = sort_config.get("strategy") or "alphabetical"
    input_dir = sort_config.get("input_dir") or "data/raw"
    output_dir = sort_config.get("output_dir") or "data/sorted"
    categories = sort_config.get("categories") or []

    # 2. Create a sample raw file if none exists (for first-time users).
    # WHY? Demos should work out-of-the-box. Nothing is more frustrating than
    #      cloning a repo and having it crash because data/ is empty.
    sample_file = os.path.join(input_dir, "sample_texts.txt")
    if not os.path.exists(sample_file):
        os.makedirs(input_dir, exist_ok=True)
        sample_lines = [
            "The new iPhone was released yesterday",
            "Manchester United won the match 3-0",
            "The parliament passed a new education bill",
            "A new Marvel movie broke box office records",
            "Python 3.12 improves performance significantly",
            "The Olympics opening ceremony was spectacular",
        ]
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sample_lines))
        print(f"[INFO] Created sample file: {sample_file}")

    # 3. Initialize the sorter with config-driven strategy.
    sorter = TextSorter(strategy=strategy, categories=categories)

    # 4. Execute sorting.
    print(f"[INFO] Running sort strategy: '{strategy}'")
    result = sorter.sort(sample_file, output_dir)

    # 5. Report results.
    print("\n[SUCCESS] Sorting complete! Output files:")
    for key, path in result.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    # WHY this guard? If another script imports main.py, this block does not run.
    # It allows main.py to be both a script and a module.
    main()
