import unittest
from unittest.mock import Mock, patch

from src.agents.agents import run_web_search


class WebSearchTests(unittest.TestCase):
    def test_returns_readable_pages_and_skips_pdf_results(self):
        client = Mock()
        client.search.return_value = {
            "results": [
                {
                    "title": "PDF report",
                    "url": "https://example.org/report.PDF?download=1",
                    "snippet": "PDF content",
                },
                {
                    "title": "HTML research",
                    "url": "https://example.org/research",
                    "content": "Recent evidence",
                },
            ]
        }

        with patch("src.tools.tools._get_tavily_client", return_value=client):
            results = run_web_search("AI research")

        self.assertIn("HTML research", results)
        self.assertIn("https://example.org/research", results)
        self.assertNotIn("PDF report", results)
        client.search.assert_called_once_with(
            query="AI research",
            max_results=5,
        )

    def test_falls_back_to_pdf_results_when_no_html_pages(self):
        client = Mock()
        client.search.return_value = {
            "results": [
                {
                    "title": "PDF report",
                    "url": "https://example.org/report.pdf",
                    "snippet": "PDF content",
                }
            ]
        }

        with patch("src.tools.tools._get_tavily_client", return_value=client):
            results = run_web_search("AI research")

        self.assertIn("PDF report", results)

    def test_raises_when_search_returns_nothing(self):
        client = Mock()
        client.search.return_value = {"results": []}

        with patch("src.tools.tools._get_tavily_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "no results"):
                run_web_search("AI research")

    def test_surfaces_tavily_errors(self):
        client = Mock()
        client.search.side_effect = Exception("rate limit exceeded")

        with patch("src.tools.tools._get_tavily_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "rate limit exceeded"):
                run_web_search("AI research")


if __name__ == "__main__":
    unittest.main()
