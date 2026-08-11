"""
Streamlit entry point for the Local Customer Feedback Analyzer.

HIGHLIGHTS: this file NEVER imports `transformers` and NEVER calls
`pandas.read_csv` directly. Every real piece of work — loading the model,
running inference, parsing a CSV, computing dashboard aggregates — is
delegated to `src/sentiment/classifier.py`, `src/ingestion/csv_loader.py`,
and `src/analytics/aggregator.py`. `app.py`'s only job is UI: collecting
input, calling into `src/`, and rendering what comes back. This mirrors
the separation already established across `src/` (see each module's own
HIGHLIGHTS docstring) and means the entire pipeline can be exercised —
and unit-tested — without ever touching Streamlit or a browser.

Run with: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import get_config
from src.analytics.aggregator import aggregate
from src.ingestion.csv_loader import extract_reviews, find_text_column, load_reviews_from_csv
from src.sentiment.classifier import classify_in_batches
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()

st.set_page_config(
    page_title="Local Customer Feedback Analyzer",
    page_icon="💬",
    layout="wide",
)

# HIGHLIGHTS: `st.session_state` is the ONLY place a batch of results lives
# between reruns. Streamlit reruns this entire script top-to-bottom on
# every widget interaction (a click, a slider drag, a new file upload) —
# without stashing the last-analyzed batch in session_state, switching
# from the Analyze tab to the Dashboard tab (which is itself just another
# rerun) would lose the results and the Dashboard would have nothing to
# show. `None` is the sentinel for "nothing analyzed yet this session".
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None


def _style_label(label: str) -> str:
    """Return an inline-CSS span that color-codes a sentiment label.

    HIGHLIGHTS: color is looked up from `config.UIConfig.label_colors`
    rather than hardcoded here, so the Overview/Analyze/Dashboard tabs and
    any future chart all agree on "Positive is green" from one source of
    truth instead of three copies that could drift out of sync.
    """
    color = cfg.ui.label_colors.get(label, "#333333")
    return f'<span style="color:{color}; font-weight:600;">{label}</span>'


def _render_results_table(df: pd.DataFrame) -> None:
    """Render a results DataFrame with color-coded labels and a
    needs-review highlight, using `st.markdown`'s HTML rendering.

    HIGHLIGHTS: `st.dataframe` alone can't color individual cell text by
    value without a Styler round-trip that's fiddly to keep readable in
    both Streamlit's light and dark themes. Building a small HTML table by
    hand keeps this predictable: labels get their configured color,
    needs-review rows get an explicit flag column instead of a color that
    might be invisible in one theme or the other.
    """
    if df.empty:
        st.info("No results to display yet.")
        return

    rows_html = []
    for _, row in df.iterrows():
        label_html = _style_label(row["label"])
        flag = (
            f'<span style="color:{cfg.ui.needs_review_color}; font-weight:600;">⚠ Needs Review</span>'
            if row["needs_review"]
            else "✓ Confident"
        )
        text = str(row["text"]).replace("<", "&lt;").replace(">", "&gt;")
        rows_html.append(
            f"<tr><td style='max-width:520px;'>{text}</td>"
            f"<td>{label_html}</td>"
            f"<td>{row['score']:.3f}</td>"
            f"<td>{flag}</td></tr>"
        )

    table_html = (
        "<table style='width:100%; border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='text-align:left;'>Review</th>"
        "<th style='text-align:left;'>Label</th>"
        "<th style='text-align:left;'>Confidence</th>"
        "<th style='text-align:left;'>Status</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _run_analysis(texts: list[str]) -> None:
    """Classify `texts` and store the results in session_state.

    HIGHLIGHTS: this is the single choke point every input mode (single
    review, multi-line paste, CSV upload) funnels through. Whatever the
    input mode, by the time it reaches here it's already a plain
    `list[str]` — the one shape `classify_in_batches()` understands. That
    means adding a fourth input mode later only requires converting it to
    a `list[str]` and calling this function, not duplicating the
    classify-and-store logic.
    """
    if not texts:
        st.warning("No non-empty review text found to analyze.")
        return

    with st.spinner(
        f"Classifying {len(texts)} review(s)... the sentiment model downloads "
        "once (~500MB) on first use and is cached after that."
    ):
        results = classify_in_batches(texts)

    st.session_state["last_results"] = results
    logger.info("Analyzed %d review(s) this session.", len(results))


# ── Tabs ─────────────────────────────────────────────────────────────────
tab_overview, tab_analyze, tab_dashboard = st.tabs(["Overview", "Analyze", "Dashboard"])


# ── Overview ─────────────────────────────────────────────────────────────
with tab_overview:
    st.title("💬 Local Customer Feedback Analyzer")
    st.markdown(
        """
        This app classifies customer reviews as **Positive**, **Negative**, or
        **Neutral** using a locally-run HuggingFace sentiment model — no data
        ever leaves your machine, and no third-party API is called.

        ### How it works

        1. **Input** — paste a single review, paste several (one per line), or
           upload a CSV of reviews.
        2. **Tokenize + Classify** — each review is passed through
           `cardiffnlp/twitter-roberta-base-sentiment-latest`, a RoBERTa model
           fine-tuned for genuine 3-class sentiment (not a binary model with a
           bolted-on "Neutral" threshold).
        3. **Confidence check** — every prediction comes with a confidence
           score. Anything below the configured threshold
           (**{threshold:.0%}** by default) is flagged **needs review**, so a
           human spot-checks exactly the predictions the model itself is
           least sure about — instead of every prediction, or none of them.
        4. **Aggregate dashboard** — once a batch has been analyzed, the
           Dashboard tab summarizes label distribution and how many reviews
           were flagged.

        ### Why this model

        The model is fine-tuned for **3-class** sentiment (Positive / Negative
        / Neutral), not the more commonly reached-for binary
        Positive/Negative default — because a real "Neutral" class needs a
        model that was actually trained to recognize it, not a threshold
        hack layered on top of a binary classifier. See the README's
        **Model Choice** section for the full comparison table and the
        honest caveat: this checkpoint was fine-tuned on tweets, so there is
        some domain shift when applied to longer, more formal product
        reviews.
        """.format(threshold=cfg.analysis.confidence_threshold)
    )

    st.subheader("Try an example")
    example = st.selectbox(
        "Pick a sample review to see it explained",
        options=cfg.ui.example_reviews,
    )
    st.caption(
        "This is just for illustration — head to the **Analyze** tab to "
        "actually classify text."
    )
    st.code(example, language=None)


# ── Analyze ──────────────────────────────────────────────────────────────
with tab_analyze:
    st.header("Analyze Reviews")
    mode = st.radio(
        "Input mode",
        options=["Single review", "Multiple reviews (paste)", "CSV upload"],
        horizontal=True,
    )

    if mode == "Single review":
        text = st.text_area(
            "Review text",
            placeholder="Paste a single customer review here...",
            max_chars=cfg.analysis.max_review_chars,
        )
        if st.button("Analyze", type="primary", disabled=not text.strip()):
            _run_analysis([text])

    elif mode == "Multiple reviews (paste)":
        raw = st.text_area(
            "One review per line",
            height=200,
            placeholder="Great product, works perfectly!\nBroke after a week, very disappointed.\n...",
        )
        if st.button("Analyze", type="primary", disabled=not raw.strip()):
            # Split on newlines and drop blank lines here in app.py — this
            # is UI-level parsing of a text widget's raw string, not CSV
            # parsing, so it doesn't belong in src/ingestion/csv_loader.py.
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            _run_analysis(lines)

    else:  # CSV upload
        st.caption(
            "The review-text column is auto-detected (looks for a column "
            "named review/text/feedback/comment, case-insensitive). A "
            f"sample file is available at `{cfg.sample_csv_path.name}`."
        )
        uploaded = st.file_uploader("Upload a CSV of reviews", type=["csv"])
        column_override = st.text_input(
            "Column name (optional — only needed if auto-detection fails)",
            value="",
        )

        if uploaded is not None:
            try:
                # `load_reviews_from_csv` accepts the file-like object
                # Streamlit hands back directly — see
                # src/ingestion/csv_loader.py's docstring.
                reviews = load_reviews_from_csv(
                    uploaded,
                    column=column_override.strip() or None,
                )
                st.success(f"Loaded {len(reviews)} non-empty review(s) from the CSV.")
                if st.button("Analyze", type="primary"):
                    _run_analysis(reviews)
            except ValueError as exc:
                # Auto-detection failed — surface the candidate columns so
                # the user knows what to type into the override box above,
                # rather than a raw traceback.
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - defensive UI guard
                st.error(f"Could not read that CSV: {exc}")

    st.divider()

    results = st.session_state["last_results"]
    if results:
        st.subheader(f"Results ({len(results)} review(s))")
        summary = aggregate([r for r in results])
        col1, col2, col3 = st.columns(3)
        col1.metric("Total analyzed", summary["total"])
        col2.metric("Flagged for review", summary["flagged_count"])
        col3.metric("Average confidence", f"{summary['average_confidence']:.1%}")
        _render_results_table(summary["dataframe"])
    else:
        st.info("Run an analysis above to see results here.")


# ── Dashboard ────────────────────────────────────────────────────────────
with tab_dashboard:
    st.header("Aggregate Dashboard")
    results = st.session_state["last_results"]

    if not results:
        st.info(
            "No batch has been analyzed yet this session. Head to the "
            "**Analyze** tab, classify some reviews, then come back here."
        )
    else:
        summary = aggregate(list(results))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total reviews", summary["total"])
        col2.metric("Flagged for review", summary["flagged_count"], f"{summary['flagged_percentage']}%")
        col3.metric("Avg. confidence", f"{summary['average_confidence']:.1%}")
        col4.metric("Distinct labels", sum(1 for c in summary["counts"].values() if c > 0))

        # HIGHLIGHTS: charts are built from `summary["counts"]` (already
        # computed once by aggregator.py) rather than re-deriving counts
        # from `summary["dataframe"]` here — one aggregation pass, reused
        # by both the metrics above and the charts below, so the numbers
        # can never drift out of sync with each other.
        counts_df = pd.DataFrame(
            {"label": list(summary["counts"].keys()), "count": list(summary["counts"].values())}
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Label distribution (bar)")
            try:
                import plotly.express as px

                fig = px.bar(
                    counts_df,
                    x="label",
                    y="count",
                    color="label",
                    color_discrete_map=cfg.ui.label_colors,
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:  # pragma: no cover - plotly always in requirements.txt
                st.bar_chart(counts_df.set_index("label"))

        with chart_col2:
            st.caption("Label distribution (share)")
            try:
                import plotly.express as px

                fig = px.pie(
                    counts_df,
                    names="label",
                    values="count",
                    color="label",
                    color_discrete_map=cfg.ui.label_colors,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:  # pragma: no cover
                st.dataframe(counts_df)

        st.caption(
            f"⚠ {summary['flagged_count']} of {summary['total']} review(s) "
            f"({summary['flagged_percentage']}%) fell below the "
            f"{cfg.analysis.confidence_threshold:.0%} confidence threshold "
            "and were flagged for human review."
        )

        if summary["trend"] is not None:
            st.caption("Daily trend (from a `date` column, if the CSV had one)")
            st.line_chart(summary["trend"])
