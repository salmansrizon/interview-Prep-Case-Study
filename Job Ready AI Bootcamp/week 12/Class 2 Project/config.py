"""
Centralized configuration for the Local Document Q&A Bot (RAG Engine).

Follows the dataclass-based config style used throughout the course
(week 7's ``src/config.py``, week 10's ``config.py``, week 11's
``config.py``) — one place to change model choice, chunking parameters,
retrieval settings, and the "not found in documents" guardrail threshold
without hunting through app.py/src/ for magic numbers/strings.

HIGHLIGHTS: এই project দুইটা আলাদা local model-এর ওপর নির্ভর করে —
embedding (ChromaDB-তে ইনডেক্স করার জন্য, ADR 0004) আর generation (Ollama,
ADR 0006)। দুটোই এখানে *data* হিসেবে রাখা হয়েছে, src/embeddings/service.py
বা src/llm/client.py-এর ভেতরে পোঁতা string literal হিসেবে না — এই দুই
module-ই config lazily পড়ে (import time-এ না, call time-এ), ঠিক week 10/11-এর
মতোই একই lazy-loading যুক্তিতে।
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EmbeddingConfig:
    """Everything about *which* sentence-embedding model we use and how.

    HIGHLIGHTS: ``model_name`` reuses the exact same model as Week 10
    (ADR 0004: ``paraphrase-MiniLM-L3-v2``) — this project's own
    src/embeddings/service.py is a FRESH, self-contained implementation of
    the same lazy-singleton pattern (not an import from week 10's folder),
    per this course's rule that every portfolio project stands on its own.
    """

    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embedding_dim: int = 384
    encode_batch_size: int = 16


@dataclass(frozen=True)
class ChunkingConfig:
    """Recursive/paragraph-aware chunking parameters (Class 1 Lecture, 3.4).

    HIGHLIGHTS: chunk_size আর chunk_overlap_chars এখানে DATA হিসেবে রাখা
    হয়েছে যাতে src/chunking/chunker.py-এর কোড generic থাকে — Brain Teaser
    "chunk size experiment" (Class 1 Lecture) এখানে এক লাইন পরিবর্তনেই করা
    যায়, chunker.py-এর logic না ছুঁয়ে। ৩০০-৮০০ ক্যারেক্টার রেঞ্জ, ~১০-১৫%
    overlap — লেকচারের section 4.2-এর সুপারিশ অনুযায়ী।
    """

    chunk_size_chars: int = 700          # target chunk size (chars), mid-range of the 300-800 sweet spot
    chunk_overlap_chars: int = 100       # ~14% overlap — avoids losing sentences at chunk borders
    min_chunk_size_chars: int = 40       # discard leftover fragments smaller than this


@dataclass(frozen=True)
class VectorStoreConfig:
    """ChromaDB persistence + collection settings.

    HIGHLIGHTS: ChromaDB এই কোর্সের ভেক্টর ডেটাবেস স্ট্যান্ডার্ড (Class 1
    Lecture section 4.1) — সম্পূর্ণ local, disk-persistent, কোনো সার্ভার
    সেটআপ লাগে না, এই কোর্সের Sovereign AI দর্শনের সাথে মিলে যায়।

    ── Vector Database Reference Table (Class 1 Lecture, section 4.1) ──

    | Database | Use Case | Advantage | Limitation |
    |---|---|---|---|
    | **ChromaDB** (used here) | Local/small-to-medium scale projects | No server setup, Python-native | Limited performance at very large (10M+ vector) scale |
    | FAISS | Research, very large scale, in-memory | Extremely fast, GPU support | Doesn't manage metadata/persistence itself — extra code needed |
    | Pinecone | Production, managed cloud service | Scalable, managed infrastructure | Cloud-based — not local, subscription cost, data leaves the machine |
    | Weaviate | Production, built-in hybrid search | Self-host or cloud, powerful filtering | More complex setup than ChromaDB |

    আমরা ChromaDB বেছে নিয়েছি কারণ এটা সম্পূর্ণ local (ডিস্কে ফাইল হিসেবে
    সেভ হয়, কোনো সার্ভার/ক্লাউড অ্যাকাউন্ট লাগে না) — production-এ স্কেল
    বাড়লে FAISS/Pinecone/Weaviate-এর দিকে যাওয়া যুক্তিসঙ্গত, কিন্তু
    শেখার আর পোর্টফোলিও-স্কেল প্রজেক্টের জন্য ChromaDB-ই সেরা এন্ট্রি পয়েন্ট।
    """

    persist_directory: str = str(BASE_DIR / "data" / "chroma_db")
    collection_name: str = "document_qa_chunks"


@dataclass(frozen=True)
class LLMConfig:
    """Everything about *which* local LLM we call and how we sample from it.

    HIGHLIGHTS: model/fallback_model — এই দুটোই একমাত্র জিনিস যা এই
    project-কে একটা নির্দিষ্ট Ollama model-এর সাথে couple করে। src/llm/client.py
    এই value গুলো lazily পড়ে (import time-এ না, generate() call time-এ) —
    week 11-এর একই lazy-loading যুক্তি, শুধু downloaded model file-এর বদলে
    local LLM server-এর জন্য প্রয়োগ করা।

    ── LLM Model Reference Table (Week 11 lecture, section 4.1) ──

    | Model | Use Case | Advantage | Limitation |
    |---|---|---|---|
    | **`llama3.1:8b`** (used here, default) | General-purpose text generation, instruction-following | Strong output quality, runs comfortably on 16GB RAM | Tight on 8GB RAM machines, can be slow there |
    | `llama3.2:3b` (fallback) | Low-resource machines | Small, runs fine even on 8GB RAM | Less accurate than the 8B model, weaker on complex instructions |
    | `mistral:7b` | Fast inference, code/structured tasks | Similar size to llama3.1, faster on some benchmarks | Smaller community/fine-tune ecosystem than Llama |
    | OpenAI GPT-4 / Claude (Cloud API) | When you need best-in-class output quality | State-of-the-art benchmark performance | Not local — per-call cost, data leaves the machine, conflicts with Sovereign AI philosophy |

    RAG-এ generation quality খুবই গুরুত্বপূর্ণ কারণ মডেলকে নির্দেশ মেনে
    "শুধু context থেকে উত্তর দাও, না পেলে সততার সাথে স্বীকার করো" — এটা
    করার জন্য ভালো instruction-following লাগে, যা llama3.1:8b নির্ভরযোগ্যভাবে
    দেয়। 8GB-RAM মেশিনের student `llm.fallback_model`-এ switch করবে —
    config.py-তে এক লাইন বদল, কোথাও code বদলাতে হবে না।
    """

    model: str = "llama3.1:8b"
    fallback_model: str = "llama3.2:3b"
    default_temperature: float = 0.2   # low — RAG answers should be grounded/deterministic, not creative
    default_top_p: float = 0.9
    timeout_seconds: int = 120


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval + the "answer strictly from documents" guardrail.

    HIGHLIGHTS: ``similarity_threshold`` হলো Class 2 Lecture-এর guardrail
    (section 4.1)-এর CODE-LEVEL implementation — শুধু prompt wording না।
    ChromaDB `query()` প্রতিটা result-এর সাথে একটা distance ফেরত দেয়
    (ছোট distance = বেশি কাছাকাছি/প্রাসঙ্গিক)। src/rag/pipeline.py যদি সেরা
    match-এর distance এই threshold-এর চেয়ে বড় দেখে (মানে যথেষ্ট প্রাসঙ্গিক
    কিছুই পাওয়া যায়নি), তাহলে LLM-কে **call না করেই** সরাসরি "এই তথ্য
    ডকুমেন্টে পাওয়া যায়নি" ফেরত দেয় — এটাই real, testable guardrail path,
    prompt-এর ভরসায় বসে না থেকে।
    """

    top_k: int = 4                       # K=3-5 is the lecture's stated sweet spot (section 4.3)
    # ChromaDB's default distance metric here is L2 (squared Euclidean) on
    # UN-normalized paraphrase-MiniLM-L3-v2 embeddings, not cosine distance
    # — thresholds tuned empirically against this metric are NOT portable
    # to a cosine-distance collection, or to a different embedding model,
    # without re-tuning. A LOWER distance means a MORE similar chunk;
    # anything above this threshold is treated as "not similar enough to
    # trust." 22.0 was picked empirically against this project's own
    # ChromaDB collection (see README "Tuning the Guardrail Threshold"):
    # clearly-relevant questions scored ~10-19, clearly off-topic questions
    # scored ~30-37 — 22.0 sits in the gap. This is a heuristic, not a
    # guarantee (see the "Garbage In, Garbage Out" caveat, Class 2 Lecture
    # 4.2): a lightweight local model can still mis-score an oddly-worded
    # relevant question as "not found," which is the honest trade-off of
    # running fully local/offline instead of a larger cloud embedding model.
    max_distance_threshold: float = 22.0


@dataclass(frozen=True)
class AppConfig:
    """Streamlit page-level settings."""

    page_title: str = "Local Document Q&A Bot"
    page_icon: str = "📚"
    layout: str = "wide"


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating all sub-configs.

    HIGHLIGHTS: একটা বড় flat dataclass-এর বদলে কয়েকটা ছোট ছোট frozen
    dataclass compose করা — week 7/10/11-এর Config pattern-এরই প্রতিফলন।
    সবজায়গায় ``frozen=True`` মানে config পুরো process lifetime জুড়ে একটা
    constant — ``config.llm.model`` ভুলবশত mid-run mutate হলে সেটা একটা bug,
    আমরা চাই সেটা সাথে সাথে ধরা পড়ুক (FrozenInstanceError)।
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    app: AppConfig = field(default_factory=AppConfig)


_config_instance: Config | None = None


def get_config() -> Config:
    """Get or create the process-wide singleton Config instance.

    HIGHLIGHTS: week 7/10/11-এর ``get_config()``-এর মতোই একই singleton
    pattern — একটা module-level cache মানে প্রতিটা caller (app.py, প্রতিটা
    src/ module, tests) ঠিক একই config object দেখে।
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
