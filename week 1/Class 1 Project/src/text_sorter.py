"""
text_sorter.py
--------------
Motive: Automate the sorting of text lines using either pure Python or LLM assistance.
WHY: Sorting is not just alphabetical. In AI, "sorting" means categorization,
     prioritization, and routing. This module demonstrates both classical and AI-powered sorting.
WHAT IT DOES: Takes raw text, applies a strategy (alphabetical, length, or LLM category),
              and writes sorted output files.
ANALOGY: This is a *smart mailroom*. Traditional mailrooms sort by zip code (alphabetical).
         Our mailroom can read the letter and decide if it is a bill, invitation, or spam (LLM category).
"""

from typing import List, Callable, Dict
from src.llm_client import OllamaClient
from src.utils.file_handler import read_lines, write_lines


class TextSorter:
    """
    Orchestrates text sorting based on a configurable strategy.

    WHY a class? Holds configuration (strategy, categories) and state (llm_client).
    """

    def __init__(self, strategy: str = "alphabetical", categories: List[str] = None):
        self.strategy = strategy
        self.categories = categories or []
        # WHY lazy init? Do not spawn an LLM connection unless the strategy needs it.
        self._llm: OllamaClient = None

    @property
    def llm(self) -> OllamaClient:
        """
        Lazy initialization of the LLM client.

        WHY @property? The client is created only when first accessed.
        WHAT IF we created it in __init__? Every TextSorter instance would ping Ollama,
        even for alphabetical sorting that does not need it.
        ANALOGY: Like starting your car engine only when you actually press the gas pedal.
        """
        if self._llm is None:
            self._llm = OllamaClient()
        return self._llm

    def sort(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """
        Main entry point: reads raw text, sorts it, writes output files.

        RETURNS: A dictionary mapping strategy -> output file path.
        WHY return paths? The caller (main.py) may want to print a summary or open the files.
        """
        lines = read_lines(input_path)
        if not lines:
            raise ValueError(f"No valid lines found in {input_path}")

        # Strategy dispatch: a dictionary mapping strings to methods.
        # WHY dispatch dict? Cleaner than a chain of if/elif/else. 
        # Adding a new strategy means adding one dict entry, not editing a conditional tree.
        dispatch = {
            "alphabetical": self._sort_alphabetical,
            "length": self._sort_length,
            "category": self._sort_by_llm_category,
        }

        if self.strategy not in dispatch:
            raise ValueError(f"Unknown strategy: {self.strategy}. Choose from {list(dispatch.keys())}")

        # Call the chosen strategy function.
        result = dispatch[self.strategy](lines, output_dir)
        return result

    def _sort_alphabetical(self, lines: List[str], output_dir: str) -> Dict[str, str]:
        """
        Classical alphabetical sort, case-insensitive.

        WHY casefold()? Handles international characters better than lower().
        WHAT IT DOES: "Apple" and "apple" sort together; "ñ" sorts correctly.
        """
        sorted_lines = sorted(lines, key=str.casefold)
        out_path = f"{output_dir}/sorted_alphabetical.txt"
        write_lines(out_path, sorted_lines)
        return {"alphabetical": out_path}

    def _sort_length(self, lines: List[str], output_dir: str) -> Dict[str, str]:
        """
        Sort by line length (shortest first).

        WHY? In AI preprocessing, short prompts often get batched separately 
             from long prompts for token-efficiency.
        """
        sorted_lines = sorted(lines, key=len)
        out_path = f"{output_dir}/sorted_length.txt"
        write_lines(out_path, sorted_lines)
        return {"length": out_path}

    def _sort_by_llm_category(self, lines: List[str], output_dir: str) -> Dict[str, str]:
        """
        Uses the local LLM to categorize each line, then writes one file per category.

        WHY? This is the bridge between "run a local LLM" and "automate text sorting."
        The LLM reads each line and decides which bucket it belongs in.

        WHAT IT DOES: 
          1. For each line, asks the LLM: "Which category?"
          2. Groups lines by category.
          3. Writes category files.

        ANALOGY: This is a *librarian who has read every book*. Instead of sorting by 
                 title (alphabetical), they sort by genre because they understand the content.
        """
        # Validate Ollama is running before processing 1000 lines.
        if not self.llm.is_alive():
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")

        # Build a strict prompt to force the LLM to output ONLY the category name.
        # WHY strict prompt? LLMs are chatty. Without instructions, they might return
        # a paragraph instead of one word, breaking our parser.
        category_list = ", ".join(self.categories)
        buckets: Dict[str, List[str]] = {cat: [] for cat in self.categories}
        buckets["other"] = []  # Fallback bucket

        for line in lines:
            prompt = (
                f"Classify the following text into exactly one of these categories: "
                f"{category_list}.\n\n"
                f"Text: \"{line}\"\n\n"
                f"Respond with ONLY the category name, nothing else."
            )
            try:
                category = self.llm.generate(prompt).lower().strip()
                # WHY .strip('.")? The LLM might return 'technology.' or '"sports"'.
                category = category.strip('.,;:"\''')
                if category not in buckets:
                    category = "other"
                buckets[category].append(line)
            except Exception as e:
                # WHY catch here? One bad LLM call should not kill the entire batch.
                # We log and move on — a production-grade resilience pattern.
                print(f"[WARN] Failed to categorize line: {line[:50]}... Error: {e}")
                buckets["other"].append(line)

        # Write one file per category.
        result_paths = {}
        for category, cat_lines in buckets.items():
            if cat_lines:  # Skip empty categories.
                out_path = f"{output_dir}/category_{category}.txt"
                write_lines(out_path, cat_lines)
                result_paths[category] = out_path

        return result_paths
