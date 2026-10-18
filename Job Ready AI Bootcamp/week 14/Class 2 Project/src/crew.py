"""Assemble the Crew. Three execution modes, same agents."""
from crewai import Crew, LLM, Process

from src.agents import build_agents
from src.llm import build_llm
from src.tasks import build_tasks

MODES = ("sequential", "parallel", "hierarchical")


def build_crew(topic: str, mode: str = "sequential", llm: LLM | None = None,
               output_file: str | None = None) -> Crew:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    llm = llm or build_llm()
    agents = build_agents(llm)
    tasks = build_tasks(agents, topic, parallel=(mode == "parallel"), output_file=output_file)
    # HIGHLIGHTS: sequential/parallel দুটোই Process.sequential — parallel-এ শুধু কিছু task async।
    # hierarchical-এ manager_llm একটা manager বানায় যে কাজ ভাগ করে ও ফল যাচাই করে;
    # manager-কে agents list-এ দেবেন না। ছোট model (3b) manager হিসেবে প্রায়ই loop করে — 8b+ ব্যবহার করুন।
    if mode == "hierarchical":
        return Crew(agents=list(agents.values()), tasks=tasks, process=Process.hierarchical,
                    manager_llm=llm)
    return Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential)
