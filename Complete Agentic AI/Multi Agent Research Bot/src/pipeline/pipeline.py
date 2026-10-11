from src.agents.agents import build_reader_agent, critic_chain, run_web_search, writer_chain


def run_search_pipeline(topic: str) -> dict:
    state = {}

    print(f"Running search agent for topic: {topic}")
    state["search_results"] = run_web_search(
        f"Find recent, reliable information about: {topic}"
    )

    print("Running reader agent for search results")
    reader_agent = build_reader_agent()
    reader_response = reader_agent.invoke({
        "messages": [(
            "user",
            f"Summarize these search results and scrape a relevant source if useful:\n"
            f"{state['search_results']}",
        )]
    })
    state["reader_results"] = reader_response["messages"][-1].content

    print("Running writer chain for reader results")
    state["writer_results"] = writer_chain.invoke({
        "topic": topic,
        "research": state["reader_results"],
    })

    print("Running critic chain for writer results")
    state["critic_results"] = critic_chain.invoke({"report": state["writer_results"]})

    print(f"Pipeline completed for topic: {topic}")
    return state
