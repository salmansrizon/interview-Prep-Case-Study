"""
main.py
-------
Motive: Single entry point for the data parsing project.
WHAT IT DOES: Creates sample raw data if missing, runs the parser, prints a summary.
ANALOGY: The *start button* on a washing machine. You press it; the machine handles
         fill, wash, rinse, spin in the correct order.
"""

import json
import os
from src.parser import DataParser


def create_sample_data(raw_dir: str = "data/raw") -> None:
    """
    Generates messy sample data so the project runs immediately.

    WHY? Students should see output on the first run, not spend 20 minutes
    formatting test data. The sample data INTENTIONALLY includes errors
    (empty text, bad status, missing fields) to demonstrate cleaning.
    """
    os.makedirs(raw_dir, exist_ok=True)
    sample_path = os.path.join(raw_dir, "sample_raw.jsonl")

    # These records are intentionally dirty to test every cleaner function.
    samples = [
        {"id": 1, "text": "  Python is great for AI  ", "status": "active", "timestamp": "2024-01-15T10:00:00Z"},
        {"id": 2, "text": "", "status": "active", "timestamp": "2024-01-15T10:05:00Z"},  # Empty text
        {"id": 3, "text": "Short", "status": "deleted", "timestamp": "2024-01-15T10:10:00Z"},  # Bad status
        {"id": 4, "text": "Machine learning transforms industries by finding patterns in massive datasets", "status": "active", "timestamp": "2024-01-15T10:15:00Z"},
        {"id": 5, "text": "OK", "status": "pending", "timestamp": "2024-01-15T10:20:00Z"},  # Too short
        {"id": 6, "text": "   ", "status": "active", "timestamp": "2024-01-15T10:25:00Z"},  # Whitespace only
        {"id": 7, "text": "Natural language processing enables machines to understand human context and sentiment", "status": "pending", "timestamp": "2024-01-15T10:30:00Z"},
        {"id": 8, "text": "x" * 600, "status": "active", "timestamp": "2024-01-15T10:35:00Z"},  # Too long
        {"id": 9, "text": "Computer vision systems detect objects in real-time video streams", "status": "active", "timestamp": "2024-01-15T10:40:00Z"},
        {"id": 10, "text": "Reinforcement learning trains agents through trial and error in simulated environments", "status": "active", "timestamp": "2024-01-15T10:45:00Z"},
    ]

    with open(sample_path, "w", encoding="utf-8") as f:
        for record in samples:
            f.write(json.dumps(record) + "\n")

    print(f"[INFO] Created sample raw data: {sample_path}")


def main():
    # Ensure sample data exists.
    create_sample_data()

    # Run the pipeline.
    parser = DataParser()
    summary = parser.run()

    # Print formatted summary.
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()
