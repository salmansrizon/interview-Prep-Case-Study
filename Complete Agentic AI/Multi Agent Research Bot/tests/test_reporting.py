import unittest

from src.reporting import create_report_pdf, extract_overview, extract_sources


class ReportingTests(unittest.TestCase):
    def test_overview_uses_introduction_and_key_findings(self):
        report = """**Research Report – Example**
*Prepared: October 2026*

## 1. Introduction
This research reviews the latest developments and their effect on the field, with a focus on deployments and technical progress.

## 2. Key Findings
### 2.1. Adoption is accelerating
| Metric | Significance |
|-------|--------------|
| Market growth | Adoption is up |

**Why this is an advance**
1. Recent surveys show increased adoption across several industries.
### 2.2. Costs remain a barrier
Implementation costs continue to slow smaller organizations.

## 3. Conclusion
The field is continuing to evolve.
"""

        summary, takeaways = extract_overview(report)

        self.assertTrue(summary.startswith("This research reviews"), summary)
        self.assertNotIn("Research Report", summary)
        self.assertEqual(
            [heading for heading, _ in takeaways],
            ["Adoption is accelerating", "Costs remain a barrier"],
        )
        self.assertIn("increased adoption", takeaways[0][1])
        self.assertNotIn("|", takeaways[0][1])

    def test_sources_are_unique_and_punctuation_is_removed(self):
        sources = extract_sources(
            "URL: https://example.org/a. More: https://example.org/a",
            "See https://example.org/b).",
        )

        self.assertEqual(sources, ["https://example.org/a", "https://example.org/b"])

    def test_pdf_is_a_downloadable_pdf_document(self):
        pdf = create_report_pdf(
            "# Research Findings\n\nA readable report with **important evidence**.\n"
            "- First finding\n- Second finding",
            "Research brief",
        )

        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 1000)


if __name__ == "__main__":
    unittest.main()
