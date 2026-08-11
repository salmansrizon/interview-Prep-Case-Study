"""
ChromaDB wrapper: the ONLY module in this project that imports ``chromadb``
or knows its request/response shapes.

HIGHLIGHTS — কেন ChromaDB-কে এখানে isolate করা, app.py বা src/rag/pipeline.py
থেকে সরাসরি call না করে? Week 10/11-এর একই boundary যুক্তি: caller-রা কখনো
``chromadb.PersistentClient`` বা একটা raw ChromaDB result dict দেখে না।
তারা শুধু এই module-এর তিনটা function call করে — ``index_chunks()``,
``query()``, ``list_indexed_sources()`` — যেগুলো plain dataclass/dict/list
নেয় আর ফেরত দেয়। এর দুইটা concrete লাভ:
  ১. Swappability — ভবিষ্যতে FAISS/Pinecone/Weaviate-এ move করতে চাইলে
     (config.py-এর Vector Database Reference Table দেখুন) শুধু এই file
     বদলাতে হবে।
  ২. Testability — tests/test_pipeline.py একটা real temp ChromaDB
     PersistentClient (একটা tmp_path ডিরেক্টরিতে) ব্যবহার করতে পারে অথবা
     hand-crafted embedding দিয়ে — কোনো network call ছাড়াই, কারণ ChromaDB
     নিজেই ১০০% local (কোনো সার্ভার লাগে না)।

HIGHLIGHTS — module-এর top-এ না করে function-এর ভেতরে ``chromadb`` import
কেন? sentence-transformers/ollama-এর lazy-import প্যাটার্নের প্রতিফলন —
যদিও chromadb import করা নিজে ভারী না (network touch করে না), local করে
রাখাটা একটা visual signal: এই module import করা মানেই client তৈরি হয়ে যাওয়া
না, শুধু ``get_client()`` CALL করলেই।
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from config import get_config
from src.chunking.chunker import Chunk
from src.embeddings.service import embed
from src.utils.logger import get_logger

logger = get_logger("vectorstore")

_client = None
_collection = None


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieval result: the chunk text plus its citation metadata and
    how similar it was to the query.

    HIGHLIGHTS: ``distance`` কে raw ChromaDB output হিসেবেই রাখা হয়েছে
    (একটা normalized 0-1 "similarity score"-এ রূপান্তর না করে) কারণ
    src/rag/pipeline.py-এর "not found in documents" guardrail সরাসরি এই
    distance-কে ``config.retrieval.max_distance_threshold``-এর সাথে তুলনা
    করে (ChromaDB-তে ছোট distance = বেশি প্রাসঙ্গিক)। এই মান রূপান্তর না
    করাটাই সেই threshold-কে একটা single, unambiguous মিটারে (ChromaDB-র
    নিজস্ব distance স্কেল) রাখে।
    """

    text: str
    source: str
    page: int
    distance: float


def get_client():
    """Return the shared ChromaDB PersistentClient, creating it on first use.

    HIGHLIGHTS: ``PersistentClient(path=...)`` ডিস্কে ফাইল হিসেবে সেভ হয়
    (``config.vectorstore.persist_directory``, ডিফল্ট ``./data/chroma_db``)
    — কোনো সার্ভার প্রসেস চালানো লাগে না। Lazy singleton pattern এখানেও
    একই কারণে ব্যবহার করা হয়েছে: শুধু import করলে (যেমন কোনো test থেকে)
    কখনো ডিস্কে ডিরেক্টরি তৈরি হয়ে যাওয়া উচিত না, শুধু আসলে ব্যবহার করলেই।
    """
    global _client

    if _client is None:
        import chromadb

        cfg = get_config().vectorstore
        logger.info("Opening ChromaDB PersistentClient at %r", cfg.persist_directory)
        _client = chromadb.PersistentClient(path=cfg.persist_directory)

    return _client


def get_collection():
    """Return the shared ChromaDB collection, creating it on first use."""
    global _collection

    if _collection is None:
        cfg = get_config().vectorstore
        client = get_client()
        _collection = client.get_or_create_collection(name=cfg.collection_name)

    return _collection


def index_chunks(chunks: list[Chunk]) -> int:
    """Embed and index a list of chunks into ChromaDB, one entry per chunk.

    Args:
        chunks: ``Chunk`` objects from ``src.chunking.chunker`` — each
            already carries ``{source, page}`` metadata.

    Returns:
        The number of chunks actually indexed (0 if ``chunks`` is empty).

    HIGHLIGHTS: এখানেই প্রতিটা chunk-এর সাথে metadata (``source``, ``page``)
    সেভ করা হচ্ছে — কারণ পরে দুটো real feature এর জন্যই এটা লাগবে: (ক)
    প্রতিটা উত্তরের সাথে citation দেখানো (কোন ডকুমেন্ট, কোন পাতা), আর (খ)
    Ask ট্যাবের document-picker filter, যেটা এই একই ``source`` field-এর ওপর
    ChromaDB-র ``where`` clause চালায় (নিচে ``query()`` দেখুন)। এই দুটো ছাড়া
    metadata শুধু cosmetic display হয়ে থাকত।
    """
    if not chunks:
        return 0

    collection = get_collection()
    texts = [c.text for c in chunks]
    vectors = embed(texts)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": c.source, "page": c.page} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Indexed %d chunk(s) into collection %r", len(chunks), collection.name)
    return len(chunks)


def query(
    question_embedding,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-K most similar chunks to a question embedding.

    Args:
        question_embedding: A single embedding vector (list/np.ndarray) for
            the user's question — produced by ``src.embeddings.service.embed``.
        top_k: How many chunks to retrieve. Defaults to
            ``config.retrieval.top_k`` if not given.
        source_filter: If given, restricts retrieval to chunks whose
            ``source`` metadata exactly matches this filename — this is
            the "search only this document" feature, implemented as a REAL
            ChromaDB ``where`` clause, not a post-hoc filter on already-
            retrieved results (which would silently return fewer than
            top_k relevant chunks whenever other documents crowd them out).

    Returns:
        A list of ``RetrievedChunk``, best match first.

    HIGHLIGHTS: ``where={"source": source_filter}`` — Class 2 Lecture-এর
    section 3.3-এর ঠিক সেই কোড। এটা শুধু সুবিধার জন্য না — এটা ChromaDB
    engine-কেই বলে দেয় "IT-Guidelines.pdf-এর চাঙ্ক ছাড়া বাকি সব বাদ দাও",
    ফলে HR-Policy.pdf-এর কোনো chunk কখনো candidate result set-এই ঢোকে না —
    বনাম Python-এ post-filter করলে যেখানে HR-Policy-র chunk গুলো "প্রাসঙ্গিক"
    বলে top-K স্লট দখল করে ফেলতে পারত, IT-Guidelines-এর আসল প্রাসঙ্গিক
    চাঙ্ক গুলোকে সরিয়ে দিয়ে।
    """
    cfg = get_config().retrieval
    resolved_top_k = top_k or cfg.top_k
    collection = get_collection()

    if collection.count() == 0:
        return []

    where = {"source": source_filter} if source_filter else None

    vector = (
        question_embedding.tolist()
        if hasattr(question_embedding, "tolist")
        else list(question_embedding)
    )

    results = collection.query(
        query_embeddings=[vector],
        n_results=min(resolved_top_k, collection.count()),
        where=where,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = [
        RetrievedChunk(
            text=doc,
            source=meta.get("source", "unknown"),
            page=meta.get("page", -1),
            distance=float(dist),
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
    logger.info(
        "Retrieved %d chunk(s) (source_filter=%r, top_k=%d)",
        len(retrieved),
        source_filter,
        resolved_top_k,
    )
    return retrieved


def list_indexed_sources() -> list[str]:
    """Return the sorted, de-duplicated list of ``source`` filenames
    currently indexed — used to populate the Ask tab's document picker.

    HIGHLIGHTS: এই function ChromaDB collection-এর সব metadata স্ক্যান করে
    unique ``source`` value বের করে — এটাই সেই "reads the same metadata
    that indexing wrote" loop, যা document-picker-কে সবসময় সত্যিকার
    ইনডেক্স করা ডকুমেন্টের তালিকার সাথে sync রাখে, app.py-তে হাতে-মেইনটেইন
    করা কোনো তালিকার বদলে (যেটা drift করে যেতে পারত)।
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.get(include=["metadatas"])
    sources = {meta["source"] for meta in result.get("metadatas", []) if meta and "source" in meta}
    return sorted(sources)


def reset_collection() -> None:
    """Delete and recreate the collection — used by tests to start clean,
    and available for a future "clear index" UI action.

    HIGHLIGHTS: টেস্ট isolation-এর জন্য এই function দরকার — প্রতিটা টেস্ট
    একটা তাজা, খালি collection থেকে শুরু করা উচিত, আগের টেস্টের ইনডেক্স করা
    chunk বহন না করেই।
    """
    global _collection
    client = get_client()
    cfg = get_config().vectorstore
    try:
        client.delete_collection(name=cfg.collection_name)
    except Exception:  # noqa: BLE001 - fine if it didn't exist yet
        pass
    _collection = None
