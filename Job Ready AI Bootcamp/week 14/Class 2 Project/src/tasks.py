"""Tasks: what each agent must do, and how outputs flow between them."""
from crewai import Agent, Task


def build_tasks(agents: dict[str, Agent], topic: str, parallel: bool = False,
                output_file: str | None = None) -> list[Task]:
    r, w, v = agents["researcher"], agents["writer"], agents["reviewer"]

    if parallel:
        # HIGHLIGHTS: async_execution=True মানে task background-এ শুরু হয়, পরের task-এর অপেক্ষা করে না।
        # দুটো স্বাধীন research একসাথে চলে; writer-এর context=[...] দুটোর শেষ পর্যন্ত অপেক্ষা করে।
        # নিয়ম: শেষ task কখনো async হতে পারবে না। একটাই Ollama-তে চালালে গতি বাড়ার সীমা আছে।
        research = [
            Task(description=f"Research the background and key concepts of: {topic}.",
                 expected_output="10 bullet points of core facts.", agent=r, async_execution=True),
            Task(description=f"Research current trends, risks and use cases of: {topic}.",
                 expected_output="10 bullet points on trends, risks and use cases.",
                 agent=r, async_execution=True),
        ]
    else:
        research = [Task(description=f"Research the topic: {topic}. Cover background, trends, risks.",
                         expected_output="15 bullet points of facts, grouped by theme.", agent=r)]

    # HIGHLIGHTS: context=[...] হলো dependency — আগের task-এর output পরের task-এর prompt-এ যুক্ত হয়।
    write = Task(description=f"Using the research, write a full report on {topic} in Markdown "
                             "with: Title, Summary, 3-4 sections, Conclusion.",
                 expected_output="A complete Markdown report.", agent=w, context=research)
    review = Task(description="Review the draft against the research. Fix errors and "
                              "return the FINAL report in Markdown only.",
                  expected_output="The final polished Markdown report.", agent=v,
                  context=[write] + research if parallel else [write],
                  output_file=output_file)  # HIGHLIGHTS: output_file দিলে CrewAI নিজেই ফাইলে লেখে।
    return [*research, write, review]
