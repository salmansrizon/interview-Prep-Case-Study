"""Agents: role + goal + backstory define each teammate's behaviour."""
from crewai import Agent, LLM


def build_agents(llm: LLM) -> dict[str, Agent]:
    # HIGHLIGHTS: role = সে কে, goal = কী অর্জন করবে, backstory = কেমন মানসিকতায় কাজ করবে।
    # তিনটাই system prompt-এ ঢোকে, তাই specific লিখলে output-এর মান সরাসরি বাড়ে।
    # allow_delegation=False: agent-রা একে অপরকে কাজ ঠেলে দেবে না (hierarchical-এ manager সেটা করে)।
    researcher = Agent(
        role="Senior Research Specialist",
        goal="Collect accurate, well-organised facts about the topic.",
        backstory="You are a meticulous researcher who separates facts from speculation "
                  "and flags anything you are unsure about.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    writer = Agent(
        role="Technical Report Writer",
        goal="Turn research notes into a clear, structured Markdown report.",
        backstory="You write for busy engineers: short sections, concrete examples, no fluff.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    reviewer = Agent(
        role="Critical Reviewer",
        goal="Fix factual gaps and weak structure, and return the final polished report.",
        backstory="You are a strict editor. You never invent facts; you remove claims "
                  "that the research notes do not support.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    return {"researcher": researcher, "writer": writer, "reviewer": reviewer}
