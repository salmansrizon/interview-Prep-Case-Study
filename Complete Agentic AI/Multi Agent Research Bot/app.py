import os
from html import escape
import streamlit as st
import time
from langchain_core.messages import HumanMessage
from src.agents.agents import run_web_search, build_reader_agent, writer_chain, critic_chain
from src.reporting import create_report_pdf, extract_overview, extract_sources

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --paper: #ffffff;
    --ink: #111111;
    --muted: #4b5563;
    --line: #dedede;
    --soft: #f7f7f6;
    --accent: #214e9b;
    --success: #176b43;
    --danger: #b42318;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    color: var(--ink);
    background: var(--paper);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: #fff !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 1180px;
    padding: 1.5rem 2.5rem 3.5rem;
}

.masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 1rem;
    border-bottom: 1px solid var(--line);
}
.brand {
    display: flex;
    align-items: center;
    color: var(--ink);
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    white-space: nowrap;
}
.brand-mark {
    display: inline-grid;
    place-items: center;
    width: 1.65rem;
    height: 1.65rem;
    margin-right: 0.55rem;
    border: 1px solid var(--ink);
    font-size: 0.8rem;
    vertical-align: middle;
}
.masthead-meta, .eyebrow, .step-num, .step-status {
    font-family: "Cascadia Code", Consolas, monospace;
}
.masthead-meta {
    color: var(--muted);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
}
.hero {
    padding: 3.3rem 0 2.6rem;
    border-bottom: 1px solid var(--line);
}
.eyebrow {
    margin: 0 0 0.8rem;
    color: var(--accent);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.hero h1 {
    max-width: 760px;
    margin: 0;
    color: var(--ink);
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 720;
    letter-spacing: -0.055em;
    line-height: 1.04;
}
.hero-sub {
    max-width: 620px;
    margin: 1rem 0 0;
    color: #30343b;
    font-size: 1.02rem;
    line-height: 1.65;
}
.workspace {
    padding-top: 2rem;
}
.input-card, .timeline-card {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: 0 3px 14px rgba(17, 17, 17, 0.035);
}
.input-card {
    padding: 1.4rem;
}
.section-heading {
    margin: 0 0 0.3rem;
    color: var(--ink);
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.025em;
}
.section-subtitle {
    margin: 0 0 1.2rem;
    color: var(--muted);
    font-size: 0.88rem;
}
.divider {
    height: 1px;
    margin: 2.2rem 0;
    background: var(--line);
}
.stTextInput > div > div > input {
    min-height: 48px;
    border: 1px solid #aeb4bc !important;
    border-radius: 5px !important;
    background: #fff !important;
    color: #111 !important;
    font-size: 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(33, 78, 155, 0.14) !important;
}
.stTextInput input::placeholder {
    color: #555 !important;
    opacity: 1 !important;
}
.stTextInput > label {
    color: #111 !important;
    font-size: 0.85rem !important;
    font-weight: 650 !important;
}
.stButton > button {
    min-height: 48px;
    border: 1px solid #111 !important;
    border-radius: 5px !important;
    background: #111 !important;
    color: #fff !important;
    font-size: 0.95rem !important;
    font-weight: 650 !important;
    transition: background-color 160ms ease, transform 160ms ease !important;
}
.stButton > button, .stButton > button p {
    color: #fff !important;
}
.stButton > button:hover {
    border-color: #333 !important;
    background: #333 !important;
    transform: translateY(-1px);
}

.stage-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 1.1rem 0 1.5rem;
}
.stage-card {
    min-height: 126px;
    padding: 1rem;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #fff;
    transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
}
.stage-card.running {
    border-color: #93b4e2;
    box-shadow: 0 5px 18px rgba(33, 78, 155, 0.1);
    animation: stage-push 240ms ease-out both;
}
.stage-card.done { border-color: #a8cdb5; }
.stage-card.failed { border-color: #e6aaa5; }
.stage-card-index {
    color: #687386;
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: 0.68rem;
}
.stage-card-title {
    margin-top: 0.65rem;
    color: #111;
    font-size: 0.95rem;
    font-weight: 700;
}
.stage-card-status {
    margin-top: 0.4rem;
    color: #566173;
    font-size: 0.76rem;
}
.stage-card.running .stage-card-status { color: #214e9b; }
.stage-card.done .stage-card-status { color: #176b43; }
.stage-card.failed .stage-card-status { color: #b42318; }
.report-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
    margin: 0.75rem 0 1rem;
}
.overview-summary {
    padding: 1rem 1.1rem;
    border-left: 3px solid var(--accent);
    background: #f6f8fb;
    color: #202938;
    line-height: 1.7;
}
.takeaway-card {
    height: 100%;
    padding: 1rem;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: #fff;
}
.takeaway-card strong { color: #111; }
.takeaway-card p {
    margin: 0.45rem 0 0;
    color: #4b5563;
    font-size: 0.88rem;
    line-height: 1.55;
}
.share-widget {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    min-height: 44px;
}
.share-widget button {
    min-height: 42px;
    padding: 0.55rem 1rem;
    border: 1px solid #c9ced6;
    border-radius: 5px;
    background: #fff;
    color: #111;
    cursor: pointer;
    font: 600 0.9rem Inter, "Segoe UI", Arial, sans-serif;
}
.share-widget button:hover { background: #f4f5f7; }
.share-widget [role="status"] { color: #176b43; font-size: 0.82rem; }
.report-source {
    overflow-wrap: anywhere;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--line);
}
.report-source a { color: #214e9b !important; }
@keyframes stage-push {
    from { opacity: 0.7; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.agent-timeline-detail {
    margin-top: 0.16rem;
    color: #4a5665;
    font-size: 0.82rem;
    line-height: 1.45;
}
[role="progressbar"] > div > div {
    background-color: #e7e7e7 !important;
    border-radius: 2px !important;
}
[role="progressbar"] > div > div > div {
    background: var(--accent) !important;
    border-radius: 2px !important;
}
.panel-label {
    color: #111;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--line);
}
.stSpinner > div { color: var(--accent) !important; }
div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #fff;
}
div[data-testid="stExpander"] summary, .notice, .stCaption {
    color: #222 !important;
}
.notice {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    color: #333;
    font-size: 0.75rem;
    text-align: left;
}
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4, [data-testid="stMarkdownContainer"] {
    color: #111;
}
.stMarkdown [data-testid="stMarkdownContainer"] *,
.stMarkdown [data-testid="stMarkdownContainer"] *::before,
.stMarkdown [data-testid="stMarkdownContainer"] *::after {
    color: #111 !important;
}
.stAlert p { color: #111 !important; }
.stDownloadButton > button {
    width: auto;
    padding: 0.55rem 1rem !important;
    background: #fff !important;
    color: #111 !important;
}
.stDownloadButton > button:hover {
    background: #f3f3f3 !important;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
@media (max-width: 640px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .hero { padding: 2.4rem 0 1.8rem; }
    .stage-grid {
        grid-template-columns: 1fr;
        gap: 0.55rem;
    }
    .stage-card { min-height: 0; }
    .masthead { gap: 0.5rem; }
    .masthead-meta {
        max-width: 45%;
        font-size: 0.58rem;
        text-align: right;
    }
    .input-card { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key in ("results", "running", "done"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else False
for key, default in (
    ("current_step", None),
    ("failed_step", None),
    ("stage_durations", {}),
    ("latest_output", None),
    ("latest_output_label", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="brand"><span class="brand-mark">R</span> RESEARCH STUDIO</div>
    <div class="masthead-meta">MULTI-AGENT WORKSPACE &nbsp;·&nbsp; WEB RESEARCH</div>
</div>
<div class="hero">
    <p class="eyebrow">Research workspace</p>
    <h1>Research, beyond the first result.</h1>
    <p class="hero-sub">
        Four focused AI stages gather sources, extract context, shape a report,
        and review the final brief.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Research input ───────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        '<div class="section-heading">Start a research brief</div>'
        '<p class="section-subtitle">Define a focused question. The agents will take it from there.</p>',
        unsafe_allow_html=True,
    )
    input_col, button_col = st.columns([5, 1.5], vertical_alignment="bottom")
    with input_col:
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. Grid-scale energy storage advances",
            key="topic_input",
            label_visibility="visible",
        )
    with button_col:
        run_btn = st.button("Generate research brief", use_container_width=True)

st.caption("Try: LLMs in healthcare · The future of AI agents · Clean energy storage")

st.markdown(
    '<div class="section-heading">Research pipeline</div>'
    '<p class="section-subtitle">Four focused stages, from source discovery to final review.</p>',
    unsafe_allow_html=True,
)

timeline_placeholder = st.empty()
latest_output_placeholder = st.empty()

pipeline_steps = (
    ("search", "Search Agent", "Gathers recent web information"),
    ("reader", "Reader Agent", "Scrapes & extracts deep content"),
    ("writer", "Writer Chain", "Drafts the full research report"),
    ("critic", "Critic Chain", "Reviews & scores the report"),
)


def render_timeline():
    results = st.session_state.results
    completed = sum(step in results for step, _, _ in pipeline_steps)
    current = st.session_state.current_step
    failed = st.session_state.failed_step

    with timeline_placeholder.container():
        st.progress(completed / len(pipeline_steps), text=f"{completed} of 4 stages complete")
        entries = []
        for index, (step, title, description) in enumerate(pipeline_steps):
            if step in results:
                state = "done"
                duration = st.session_state.stage_durations.get(step)
                status = f"Complete · {duration:.1f}s" if duration is not None else "Complete"
            elif step == failed:
                state, status = "failed", "Failed"
            elif st.session_state.running and step == current:
                state, status = "running", "In progress"
            else:
                state, status = "waiting", "Queued"
            entries.append(
                f'<div class="stage-card {state}">'
                f'<div class="stage-card-index">STAGE {index + 1:02}</div>'
                f'<div class="stage-card-title">{title}</div>'
                f'<div class="stage-card-status">{status}</div>'
                f'<div class="agent-timeline-detail">{description}</div>'
                '</div>'
            )
        st.markdown(
            '<div class="stage-grid">' + "".join(entries) + '</div>',
            unsafe_allow_html=True,
        )


def show_latest_output(label, output):
    st.session_state.latest_output_label = label
    st.session_state.latest_output = str(output)
    with latest_output_placeholder.container():
        st.markdown(f"**Latest update · {label}**")
        st.markdown(str(output)[:1600])


render_timeline()


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:

    if not topic.strip():
        st.warning("Please enter a research topic first.")

    else:
        missing_keys = [
            key for key in ("OpenRouter_API_KEY", "TAVILY_API_KEY")
            if not os.getenv(key)
        ]
        if missing_keys:
            st.error(
                "Live research needs these environment variables: "
                f"{', '.join(missing_keys)}. Add them to your .env file and retry."
            )
        else:
            st.session_state.results = {}
            st.session_state.running = True
            st.session_state.done = False
            st.session_state.current_step = "search"
            st.session_state.failed_step = None
            st.session_state.stage_durations = {}
            st.session_state.latest_output = None
            st.session_state.latest_output_label = None
            st.rerun()


if st.session_state.running and not st.session_state.done:

    results = st.session_state.results
    topic_val = st.session_state.topic_input

    try:
        st.session_state.current_step = "search"
        render_timeline()
        started = time.perf_counter()
        with st.spinner("🔍 Search Agent is working…"):
            results["search"] = run_web_search(
                f"Find recent, reliable and detailed information about: {topic_val}"
            )
        st.session_state.stage_durations["search"] = time.perf_counter() - started
        st.session_state.results = dict(results)
        show_latest_output("Search Agent", results["search"])
        render_timeline()

        st.session_state.current_step = "reader"
        render_timeline()
        started = time.perf_counter()
        with st.spinner("📄 Reader Agent is scraping top resources…"):
            reader_agent = build_reader_agent()
            response = reader_agent.invoke({
                "messages": [HumanMessage(
                    content=(
                        f"Based on the following search results about '{topic_val}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{results['search'][:800]}"
                    )
                )]
            })
            results["reader"] = response["messages"][-1].content
        st.session_state.stage_durations["reader"] = time.perf_counter() - started
        st.session_state.results = dict(results)
        show_latest_output("Reader Agent", results["reader"])
        render_timeline()

        st.session_state.current_step = "writer"
        render_timeline()
        started = time.perf_counter()
        with st.spinner("✍️ Writer is drafting the report…"):
            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
            )
            results["writer"] = writer_chain.invoke({
                "topic": topic_val,
                "research": research_combined,
            })
        st.session_state.stage_durations["writer"] = time.perf_counter() - started
        st.session_state.results = dict(results)
        show_latest_output("Writer Chain", results["writer"])
        render_timeline()

        st.session_state.current_step = "critic"
        render_timeline()
        started = time.perf_counter()
        with st.spinner("🧐 Critic is reviewing the report…"):
            results["critic"] = critic_chain.invoke({"report": results["writer"]})
        st.session_state.stage_durations["critic"] = time.perf_counter() - started
        st.session_state.results = dict(results)
        show_latest_output("Critic Chain", results["critic"])

        st.session_state.running = False
        st.session_state.done = True
        st.session_state.current_step = None
        render_timeline()
        latest_output_placeholder.empty()
    except Exception as exc:
        st.session_state.running = False
        st.session_state.failed_step = st.session_state.current_step
        render_timeline()
        st.error(
            f"The {st.session_state.current_step or 'research'} stage failed: {exc}. "
            "Resolve the error and run the pipeline again."
        )


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="eyebrow">Completed research</p>'
        '<div class="section-heading">Your research brief</div>'
        '<p class="section-subtitle">A report built from verified agent outputs, with sources and activity kept close at hand.</p>',
        unsafe_allow_html=True
    )

    if "writer" in r:
        report = str(r["writer"])
        title = f"Research brief: {st.session_state.topic_input.strip() or 'Research'}"
        summary, takeaways = extract_overview(report)
        sources = extract_sources(r.get("search", ""), report)
        report_tabs = st.tabs(["Overview", "Full Report", "Sources", "Agent Activity"])

        with report_tabs[0]:
            with st.container(border=True):
                st.markdown('<div class="panel-label">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="overview-summary">{escape(summary) if summary else "The report is ready. Open Full Report to read the complete findings."}</div>',
                    unsafe_allow_html=True,
                )
            if takeaways:
                st.markdown("#### Key takeaways")
                takeaway_columns = st.columns(len(takeaways), gap="small")
                for column, (heading, detail) in zip(takeaway_columns, takeaways):
                    with column:
                        st.markdown(
                            f'<div class="takeaway-card"><strong>{escape(heading)}</strong>'
                            f'<p>{escape(detail)}</p></div>',
                            unsafe_allow_html=True,
                        )
            st.caption(f"{len(sources)} sources · {len(report.split())} words")

        with report_tabs[1]:
            with st.container(border=True):
                st.markdown('<div class="panel-label">FINAL RESEARCH REPORT</div>', unsafe_allow_html=True)
                st.markdown(report)
            st.markdown('<div class="report-toolbar"></div>', unsafe_allow_html=True)
            download_col, share_col = st.columns([1, 2], vertical_alignment="center")
            with download_col:
                st.download_button(
                    label="Download Markdown",
                    data=report,
                    file_name=f"research_report_{int(time.time())}.md",
                    mime="text/markdown",
                    key="download_markdown",
                )
                try:
                    pdf_data = create_report_pdf(report, title)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"research_report_{int(time.time())}.pdf",
                        mime="application/pdf",
                        key="download_pdf",
                    )
                except Exception as exc:
                    st.error(f"Could not generate the PDF report: {exc}")
            with share_col:
                # The model text is HTML-attribute escaped before JavaScript reads it as plain text.
                st.html(
                    '<div class="share-widget">'
                    f'<button id="copy-research-report" data-report="{escape(report, quote=True)}">Copy report</button>'
                    '<span id="copy-report-status" role="status" aria-live="polite"></span>'
                    '</div>'
                    '<script>'
                    'const copyButton = document.getElementById("copy-research-report");'
                    'const copyStatus = document.getElementById("copy-report-status");'
                    'copyButton.addEventListener("click", async () => {'
                    'try { await navigator.clipboard.writeText(copyButton.dataset.report);'
                    'copyStatus.textContent = "Copied to clipboard";'
                    '} catch (error) { copyStatus.textContent = "Clipboard access was denied"; }'
                    '});'
                    '</script>',
                    unsafe_allow_javascript=True,
                )

        with report_tabs[2]:
            st.markdown("#### Sources")
            if sources:
                for source in sources:
                    st.markdown(
                        f'<div class="report-source"><a href="{escape(source, quote=True)}" '
                        f'target="_blank" rel="noopener noreferrer">{escape(source)}</a></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No source URLs were found in the search output or report.")

        with report_tabs[3]:
            st.markdown("#### Agent activity")
            for index, (step, agent, description) in enumerate(pipeline_steps, start=1):
                if step in r:
                    status = "Complete"
                    duration = st.session_state.stage_durations.get(step)
                    if duration is not None:
                        status += f" · {duration:.1f}s"
                    with st.expander(f"{index:02} · {agent} · {status}"):
                        st.markdown(str(r[step]))
                elif step == st.session_state.failed_step:
                    st.markdown(f"**{agent} · Failed** — The stage stopped with an error.")
                elif st.session_state.running and step == st.session_state.current_step:
                    st.markdown(f"**{agent} · In progress** — {description}")
                else:
                    st.markdown(f"**{agent} · Queued** — {description}")

    elif "search" in r or "reader" in r:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-label">RESEARCH IN PROGRESS</div>',
                unsafe_allow_html=True,
            )
            st.info("The final report will appear here when the writer stage completes.")

    if "critic" in r:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-label">REVIEW NOTES</div>',
                unsafe_allow_html=True,
            )
            st.markdown(r["critic"])


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchAgent · Powered by LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)