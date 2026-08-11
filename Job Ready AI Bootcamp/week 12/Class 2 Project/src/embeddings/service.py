"""
Sentence-embedding service: wraps ``sentence-transformers`` behind a lazy,
singleton-loaded ``get_model()`` and a small ``embed()`` convenience
function.

HIGHLIGHTS: এই file Week 10-এর src/embeddings/service.py-এর lazy-loading
singleton প্যাটার্নের রেফারেন্স নিয়ে লেখা, কিন্তু সম্পূর্ণ fresh,
self-contained কোড — week 10-এর ফোল্ডার থেকে import করা হয়নি
(self-contained-artifacts নিয়ম: প্রতিটা portfolio project নিজে নিজে
দাঁড়াতে পারে, অন্য week-এর ফোল্ডারের ওপর নির্ভর না করে)। `SentenceTransformer(...)`
এই module import করার সময় কখনো call হয় না — শুধু ``get_model()`` প্রথমবার
আসলে call হলে, যা এই app-এ প্রথমবার একটা PDF ইনডেক্স করার বা প্রশ্ন জিজ্ঞেস
করার সময় ঘটে। তিনটা concrete কারণ (Week 10-এর একই যুক্তির প্রতিফলন):

  ১. Cold import cost এড়ানো — মডেলটা ~61MB, প্রথমবার ব্যবহারের সময়
     HuggingFace Hub থেকে ডাউনলোড হয়। Import time-এ লোড হলে শুধু
     `pytest` চালানো বা এই ফাইল খোলাই একটা silent ৬১MB ডাউনলোড
     ট্রিগার করত।
  ২. "একবার ডাউনলোড, তারপর চিরকাল offline" — ADR 0004-এর সিদ্ধান্ত। প্রথম
     সফল লোডের পর sentence-transformers মডেলটা ডিস্কে cache করে রাখে;
     এরপর প্রতিটা call ১০০% local।
  ৩. Testability — tests/test_pipeline.py hand-crafted ভেক্টর দিয়ে
     vectorstore/pipeline logic টেস্ট করতে পারে, real মডেল লোড না করেই,
     internet ছাড়াই CI-তে চলার জন্য।
"""

from __future__ import annotations

import numpy as np

from config import get_config
from src.utils.logger import get_logger

logger = get_logger("embeddings")

# Module-level cache. Starts as None so import time stays cheap (see module
# docstring); becomes a real SentenceTransformer instance after the first
# get_model() call, and is reused for the lifetime of the process.
_model = None


def get_model():
    """Return the shared SentenceTransformer instance, loading it on first use.

    HIGHLIGHTS: এই function-এর ভেতরেই ``sentence_transformers`` import করা
    হয়েছে, module-এর top-এ না — Week 10-এর একই কারণে: এই module-এর top-level
    import গুলো দেখেই যে কেউ বুঝবে যে শুধু import করাটা কখনো ডাউনলোড
    ট্রিগার করতে পারে না, শুধু ``get_model()`` CALL করলেই পারে।
    """
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        cfg = get_config().embedding
        logger.info("Loading embedding model %r (first use — may download)...", cfg.model_name)
        _model = SentenceTransformer(cfg.model_name)
        logger.info("Embedding model loaded.")

    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into an (N, embedding_dim) array of vectors.

    Args:
        texts: Plain-text strings — either document chunks (indexing time)
            or a single user question (query time).

    Returns:
        A numpy array of shape (len(texts), embedding_dim), dtype float32.

    HIGHLIGHTS: এই module-ই ONLY জায়গা যা `model.encode(...)` call করে।
    app.py, src/rag/pipeline.py, src/vectorstore/store.py কখনো
    sentence_transformers সরাসরি import করে না — তারা শুধু এই
    ``embed()`` call করে। এটা "app.py-এর ভেতরে কোনো সরাসরি
    sentence-transformers call না" boundary বজায় রাখে (README/spec), আর
    ভবিষ্যতে embedding backend বদলানো সহজ করে তোলে (একটা মডেল-নাম বদল,
    কোথাও কোনো caller বদলাতে হবে না)।
    """
    if not texts:
        return np.empty((0, get_config().embedding.embedding_dim), dtype=np.float32)

    model = get_model()
    cfg = get_config().embedding
    vectors = model.encode(
        texts,
        batch_size=cfg.encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
