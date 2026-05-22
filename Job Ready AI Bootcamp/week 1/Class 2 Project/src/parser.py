"""
parser.py
---------
Motive: Orchestrate the entire data parsing pipeline.
WHY: A production script is not a single 500-line block. It is a conductor
     that calls specialized modules (loader, cleaner, validator, API enricher).
WHAT IT DOES: Reads raw data, cleans it, validates it, enriches via API, writes clean output.
ANALOGY: This is the *restaurant manager*. They do not cook, wash dishes, or serve.
     They coordinate the kitchen (cleaners), the health inspector (validators),
     and the delivery driver (API client) so the customer gets a perfect meal.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from src.config_loader import ConfigLoader
from src.cleaners import apply_cleaning_pipeline
from src.validators import load_schema, validate_records
from src.api_client import SimulatedAPIClient


class DataParser:
    """
    End-to-end raw-to-clean data parser.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_loader = ConfigLoader(config_dir)
        self.parser_config = self.config_loader.get_parser_config()
        self.schema = load_schema(os.path.join(config_dir, "schema.json"))

        # Initialize API client from config.
        api_cfg = self.parser_config.get("api", {})
        self.api = SimulatedAPIClient(
            base_url=api_cfg.get("base_url", "https://api.example.com"),
            timeout=api_cfg.get("timeout", 10),
        )

    def read_raw(self) -> List[Dict[str, Any]]:
        """
        Reads all raw files from the configured input directory.

        WHY support multiple files? Real pipelines process batches dropped by
        upstream systems, not a single hardcoded file.

        WHAT IT DOES: Scans data/raw/, reads every .jsonl file, parses each line as JSON.
        """
        raw_dir = Path(self.parser_config["input"]["raw_dir"])
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

        records = []
        for file_path in raw_dir.glob("*.jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # WHY log and continue? One malformed line should not kill
                        # a batch of 10,000 valid lines.
                        print(f"  [WARN] Skipping malformed JSON in {file_path.name}:{line_num} — {e}")

        return records

    def write_clean(
        self,
        records: List[Dict[str, Any]],
        filename: str = "clean_data.jsonl",
    ) -> str:
        """
        Writes cleaned records to the output directory.

        WHY json.dumps with ensure_ascii=False? Preserves non-English characters
        (Chinese, Arabic, emoji) instead of escaping them to \uXXXX codes.
        """
        clean_dir = Path(self.parser_config["output"]["clean_dir"])
        clean_dir.mkdir(parents=True, exist_ok=True)

        out_path = clean_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return str(out_path)

    def write_quarantine(
        self,
        records: List[Dict[str, Any]],
        filename: str = "quarantine_data.jsonl",
    ) -> str:
        """
        Writes invalid records to a separate quarantine file for manual review.

        WHY separate file? You do not want bad data mixed with good data.
        Quarantine allows data engineers to inspect, fix, and re-ingest later.
        """
        clean_dir = Path(self.parser_config["output"]["clean_dir"])
        clean_dir.mkdir(parents=True, exist_ok=True)

        out_path = clean_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return str(out_path)

    def run(self) -> Dict[str, Any]:
        """
        Executes the full pipeline.

        RETURNS: A summary dict with counts, file paths, and API stats.
        """
        print("[1/5] Reading raw data...")
        raw_records = self.read_raw()
        print(f"      Found {len(raw_records)} raw records.")

        print("[2/5] Applying cleaning pipeline...")
        cleaned = apply_cleaning_pipeline(raw_records, self.parser_config)
        print(f"      {len(cleaned)} records after cleaning.")

        print("[3/5] Validating against schema...")
        valid, invalid = validate_records(cleaned, self.schema)
        print(f"      {len(valid)} valid, {len(invalid)} invalid.")

        print("[4/5] Enriching via API (with retry logic)...")
        api_cfg = self.parser_config.get("api", {})
        enriched = []
        for record in valid:
            result = self.api.enrich_with_retry(
                record,
                max_retries=api_cfg.get("retries", 3),
                delay=api_cfg.get("retry_delay", 2.0),
            )
            if result:
                enriched.append(result)
        print(f"      {len(enriched)} records successfully enriched.")

        print("[5/5] Writing output files...")
        clean_path = self.write_clean(enriched, "clean_data.jsonl")
        quarantine_path = self.write_quarantine(invalid, "quarantine_data.jsonl")

        return {
            "raw_count": len(raw_records),
            "clean_count": len(cleaned),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "enriched_count": len(enriched),
            "api_calls": self.api.call_count,
            "clean_file": clean_path,
            "quarantine_file": quarantine_path,
        }
