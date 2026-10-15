import os
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from src.tools import scrape_url, web_search


load_dotenv()


def get_llm():
    api_key = os.getenv("OpenRouter_API_KEY") or "missing-api-key"
    # Groq keys start with "gsk_"; anything else is treated as an OpenRouter key.
    if api_key.startswith("gsk_"):
        model, base_url = "openai/gpt-oss-120b", "https://api.groq.com/openai/v1"
    else:
        model, base_url = "nvidia/nemotron-3-ultra-550b-a55b:free", "https://openrouter.ai/api/v1"
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0,
    )


llm = get_llm()


def run_web_search(query: str) -> str:
    results = [
        r for r in web_search.invoke({"query": query})
        if isinstance(r.get("url"), str) and r["url"].strip()
    ]
    if not results:
        raise RuntimeError("Web search returned no results. Try a different topic.")
    # Prefer HTML pages the reader can scrape; fall back to PDF snippets if that's all there is.
    pages = [r for r in results if not urlsplit(r["url"]).path.lower().endswith(".pdf")] or results
    return "\n\n".join(
        f"Title: {r.get('title') or 'Untitled result'}\n"
        f"URL: {r['url']}\n"
        f"Snippet: {r.get('snippet') or 'No summary available.'}"
        for r in pages
    )


# 2nd Agent: Scrape Agent
def build_scrape_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="You are a helpful assistant that can scrape web pages.",
    )


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You read web search results and explain the findings clearly. "
            "Use only the scrape_url tool when scraping a web page. Never call open_file "
            "or any tool that is not listed. If a page cannot be scraped, use its search snippet."
        ),
    )


# Write the chain of thought for the following prompt: "What are the latest advancements in AI research?"
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])
writer_chain = writer_prompt | llm | StrOutputParser()


# Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical reviewer. Evaluate the quality of research reports."),
    ("human", """Critically evaluate the following research report.
Report: {report}
Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:"""),
])
critic_chain = critic_prompt | llm | StrOutputParser()