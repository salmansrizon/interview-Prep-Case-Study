"""
RAG orchestration: embed question -> retrieve (optionally filtered by
document) -> similarity-threshold guardrail -> augment prompt -> generate
-> answer + citations.

This is the "Open-Book Exam" pipeline from Class 2 Lecture, section 3.1:

    User Question
         |
         v
    [1] EMBED   (src.embeddings.service.embed)
         |
         v
    [2] RETRIEVE top-K chunks from ChromaDB, optionally filtered by source
         |            (src.vectorstore.store.query)
         v
    [3] GUARDRAIL — best match's distance too high? -> "not found", skip
         |            LLM entirely (config.retrieval.max_distance_threshold)
         v
    [4] AUGMENT — build the Open-Book-Exam-style prompt with retrieved
         |          context injected
         v
    [5] GENERATE (src.llm.client.generate)
         |
         v
    Answer + Citations (source + page for every chunk actually used)

HIGHLIGHTS: এই module-ই একমাত্র জায়গা যেখানে embeddings, vectorstore, আর
llm — তিনটা layer-ই একসাথে coordinate হয়। app.py কখনো এই তিনটা module
সরাসরি ছোঁয় না — শুধু এই file-এর ``answer_question()`` call করে। এটাই
"orchestration একটা জায়গায়, presentation আরেকটা জায়গায়" boundary বজায়
রাখে — ঠিক app.py-র docstring-এ যেমন বলা আছে।
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import get_config
from src.embeddings.service import embed
from src.llm.client import OllamaConnectionError, generate
from src.utils.logger import get_logger
from src.vectorstore.store import RetrievedChunk, query

logger = get_logger("rag.pipeline")


# HIGHLIGHTS: এই exact string-টাই "not found" guardrail-এর USER-FACING
# output, আর এটা config/prompt-এ ছড়িয়ে না থেকে এক জায়গায় constant হিসেবে
# রাখা হয়েছে যাতে tests এবং app.py দুটোই ঠিক একই string-এর বিপরীতে assert/
# তুলনা করতে পারে — দুই জায়গায় হাতে-টাইপ করা duplicate string drift করে
# যেতে পারত।
NOT_FOUND_MESSAGE = (
    "This information was not found in the documents. Try rephrasing your "
    "question, uploading the relevant document, or removing the document "
    "filter if one is active."
)

# HIGHLIGHTS — GUARDRAIL, LAYER 2 OF 2 (LAYER 1 is the similarity-threshold
# check in answer_question() below, which skips calling the LLM entirely
# when nothing relevant was retrieved). This system prompt is the
# PROMPT-LEVEL half of the guardrail from Class 2 Lecture section 4.1: even
# when retrieval DOES return chunks above the threshold, the model must
# still be told explicitly to answer ONLY from them, and to say so plainly
# if the answer isn't actually in the given context (a distance just below
# the threshold doesn't guarantee the chunk actually answers the question —
# it only means it was topically close enough to be worth showing the
# model). Two layers because prompt wording ALONE is not a reliable
# guardrail (a model can still ignore instructions), and a distance
# threshold ALONE can't verify the retrieved text actually contains the
# answer — together they cover both failure modes.
SYSTEM_PROMPT = (
    "You are a careful research assistant answering questions using ONLY "
    "the CONTEXT provided below, taken from the user's own uploaded "
    "documents. This is an open-book exam: the context is the only book "
    "you're allowed to use.\n\n"
    "Rules:\n"
    "1. Answer strictly from the CONTEXT. Do not use outside knowledge, "
    "and do not guess or make anything up.\n"
    "2. If the CONTEXT does not contain enough information to answer the "
    "question, say clearly: "
    "\"This information was not found in the documents.\" "
    "Do not try to answer anyway.\n"
    "3. When you do answer, be concise and directly reference which part "
    "of the context supports your answer."
)


@dataclass(frozen=True)
class Citation:
    """One citation: which document and page a piece of the answer came
    from — the user-facing half of the metadata contract (README).
    """

    source: str
    page: int
    distance: float


@dataclass(frozen=True)
class RAGResult:
    """The full result of one question: the answer text, whether anything
    relevant was actually found, and the citations backing it up.

    HIGHLIGHTS: ``found`` field আলাদা করে exposed করা হয়েছে (শুধু
    ``answer == NOT_FOUND_MESSAGE`` স্ট্রিং তুলনা করার বদলে) যাতে app.py আর
    tests উভয়ই guardrail trigger হয়েছে কিনা তা একটা explicit boolean দিয়ে
    চেক করতে পারে — string-matching-এর ওপর ভরসা করলে ভবিষ্যতে message text
    বদলালে সেটা silently ভেঙে যেতে পারত।
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    found: bool = True
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered, citation-labeled context
    block for the augmented prompt.

    HIGHLIGHTS: প্রতিটা chunk-এর আগে ``[Source: filename, page N]`` লেবেল
    বসানো হয়েছে সরাসরি context-এর ভেতরে — শুধু structurally metadata আলাদা
    রাখা না, বরং মডেলকে নিজের উত্তরে REFERENCE করার জন্য (System prompt-এর
    rule ৩) একটা concrete anchor দেওয়া। এটাই lecture-এর section 5 টেবিলের
    "Citations" সারির বাস্তবায়ন।
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (Source: {chunk.source}, page {chunk.page})\n{chunk.text}")
    return "\n\n".join(lines)


def _build_augmented_prompt(question: str, context_block: str) -> str:
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer the question using only the CONTEXT above."
    )


def _dedupe_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Turn retrieved chunks into a citation list, preserving first-seen
    order but collapsing repeats of the exact same (source, page).

    HIGHLIGHTS: চাঙ্কিং-এর overlap-এর কারণে একই page থেকে একাধিক chunk
    retrieve হওয়া স্বাভাবিক — dedupe না করলে ইউজার একই "hr_policy.pdf,
    page 4" citation বারবার দেখতে পেত, যা আসল সোর্স বৈচিত্র্যের চেয়ে
    বিভ্রান্তিকর বেশি দেখাত।
    """
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(source=chunk.source, page=chunk.page, distance=chunk.distance))
    return citations


def format_citation(citation: Citation) -> str:
    """Human-readable citation string, e.g. ``"hr_policy.pdf, page 4"``.

    HIGHLIGHTS: এটাকে একটা আলাদা, ছোট function হিসেবে রাখা মানে app.py আর
    tests উভয়ই ঠিক একই formatting রুল ব্যবহার করে — citation ফরম্যাট
    app.py-তে আর tests/test_pipeline.py-তে আলাদাভাবে duplicate/drift হওয়ার
    সুযোগ থাকে না।
    """
    return f"{citation.source}, page {citation.page}"


def answer_question(
    question: str,
    source_filter: str | None = None,
    top_k: int | None = None,
) -> RAGResult:
    """Answer a question strictly from the indexed documents.

    Args:
        question: The user's natural-language question.
        source_filter: If given, restrict retrieval to only this document
            filename (the document-picker's "search only this PDF" mode).
            ``None`` (or "search all") means no filter.
        top_k: Override for how many chunks to retrieve. Defaults to
            ``config.retrieval.top_k``.

    Returns:
        A ``RAGResult`` — either a real answer with citations, or the
        honest "not found in documents" message (``found=False``) with no
        citations and the LLM never called.

    Raises:
        OllamaConnectionError: propagated from ``src.llm.client.generate``
            if the Ollama server can't be reached — callers (app.py) should
            catch this and show a friendly setup message.

    HIGHLIGHTS — GUARDRAIL, LAYER 1 OF 2: THE SIMILARITY-THRESHOLD CHECK.
    এটাই real, testable guardrail path — শুধু prompt wording-এর ভরসায় বসে
    না থেকে। ChromaDB retrieval সবসময়ই top_k টা result ফেরত দেয়, এমনকি যদি
    কোনোটাই আসলে প্রাসঙ্গিক না হয় (যেমন সম্পূর্ণ অসম্পর্কিত প্রশ্ন করা
    হলেও) — vector search "কিছু না পাওয়া" বলতে জানে না, শুধু "নিকটতম যা
    পেয়েছি" ফেরত দেয়। তাই যদি সবচেয়ে ভালো match-এর distance-ও
    ``config.retrieval.max_distance_threshold``-এর চেয়ে বড় হয়, আমরা LLM-কে
    call ই করি না — সরাসরি honest "not found" answer ফেরত দিই। এটা Class 2
    Lecture-এর section 4.1-এর ঠিক সেই দাবি বাস্তবায়ন করে: "শুধু prompt
    দিয়ে দিলেই যথেষ্ট না, LLM context উপেক্ষা করেও উত্তর দিতে পারে" —
    threshold check LLM-এর ওপর নির্ভর না করেই সেই ঝুঁকি এড়ায়।
    """
    cfg = get_config()

    question = question.strip()
    if not question:
        return RAGResult(answer="Please enter a question.", found=False)

    question_vector = embed([question])[0]
    retrieved = query(question_vector, top_k=top_k, source_filter=source_filter)

    if not retrieved or retrieved[0].distance > cfg.retrieval.max_distance_threshold:
        logger.info(
            "Guardrail triggered: best_distance=%s threshold=%s (question=%r)",
            retrieved[0].distance if retrieved else None,
            cfg.retrieval.max_distance_threshold,
            question,
        )
        return RAGResult(answer=NOT_FOUND_MESSAGE, found=False, retrieved_chunks=retrieved)

    context_block = _build_context_block(retrieved)
    augmented_prompt = _build_augmented_prompt(question, context_block)

    try:
        answer_text = generate(prompt=augmented_prompt, system_prompt=SYSTEM_PROMPT)
    except OllamaConnectionError:
        # Re-raise — app.py owns turning this into a friendly UI message
        # (see src/llm/client.py's OllamaConnectionError docstring).
        raise

    citations = _dedupe_citations(retrieved)
    return RAGResult(answer=answer_text, citations=citations, found=True, retrieved_chunks=retrieved)
