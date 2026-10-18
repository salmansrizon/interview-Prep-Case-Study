"""CLI: python app.py "Topic" [--mode sequential|parallel|hierarchical]"""
import argparse
import sys

from config import MODEL_NAME, OUTPUT_DIR
from src.crew import MODES, build_crew
from src.llm import build_llm, ollama_status
from src.report import save_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CrewAI research -> write -> review team")
    p.add_argument("topic")
    p.add_argument("--mode", choices=MODES, default="sequential")
    p.add_argument("--model", default=MODEL_NAME)
    a = p.parse_args(argv)

    ok, msg = ollama_status(a.model)
    if not ok:  # graceful: crash না করে পরিষ্কার নির্দেশ
        print(msg)
        return 1
    result = build_crew(a.topic, a.mode, llm=build_llm(a.model)).kickoff()
    print("Saved:", save_report(result.raw, a.topic, OUTPUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
