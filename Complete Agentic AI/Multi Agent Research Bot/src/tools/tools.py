from langchain.tools import tool
import os
import re

import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from readability import Document
from rich import print
from tavily import TavilyClient


load_dotenv()


def _get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


@tool
def search_tavily(query: str):
    """
    Search the web using the Tavily API and return a normalized list of results.
    """
    if not query or not query.strip():
        return []

    client = _get_tavily_client()
    if client is None:
        raise RuntimeError("TAVILY_API_KEY is missing. Add it to your .env file.")

    try:
        response = client.search(query=query, max_results=5)
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed: {exc}") from exc

    raw_results = response.get("results") or [] if isinstance(response, dict) else response or []
    return [
        {
            "title": result.get("title"),
            "url": result.get("url"),
            "snippet": result.get("snippet") or result.get("content"),
        }
        for result in raw_results
        if isinstance(result, dict)
    ]


web_search = search_tavily


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    Uses multiple extraction strategies for better reliability.
    """

    if not url or not url.strip():
        return "No URL provided."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html = response.text

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
        )
        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return cleaned[:5000]

        doc = Document(html)
        clean_html = doc.summary()
        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return cleaned[:5000]

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        cleaned = re.sub(r'\s+', ' ', text)

        if cleaned:
            return cleaned[:5000]

        return "Could not extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."

    except requests.exceptions.HTTPError as exc:
        return f"HTTP error occurred: {str(exc)}"

    except Exception as exc:
        return f"Could not scrape URL: {str(exc)}"