"""
Thin wrapper around the ``ollama`` Python package: the ONLY module in this
project that imports ``ollama`` or knows Ollama's request/response shape.

HIGHLIGHTS — কেন Ollama-কে এখানে isolate করা, app.py বা src/rag/pipeline.py
থেকে সরাসরি call না করে? Week 11-এর src/llm/client.py-এর একই boundary
যুক্তি (এই ফাইলটা সেই ফাইলের STYLE রেফারেন্স নিয়ে লেখা, কিন্তু fresh,
self-contained কোড — week 11-এর ফোল্ডার থেকে import করা হয়নি): app.py আর
src/rag/pipeline.py কখনো ``ollama`` import করে না, কখনো একটা raw Ollama
response dict দেখে না — তারা শুধু এই module-এর ``generate()`` call করে,
যেটা plain string নেয় আর plain string ফেরত দেয়। তিনটা concrete লাভ:
  ১. Swappability — ভবিষ্যতে অন্য local runtime (llama.cpp server, বা
     তুলনার জন্য cloud API) ব্যবহার করতে চাইলে শুধু এই file বদলাতে হবে।
  ২. Testability — tests/test_pipeline.py ঠিক ``ollama.chat``-কে mock
     করে (``sys.modules['ollama']`` monkeypatch করে) — RAG pipeline-এর
     বাকি প্রতিটা layer (retrieval, threshold guardrail, prompt
     construction, citation formatting) সেই একটা seam-এর বিপরীতে test
     হয়, কোথাও real network/server call ছাড়াই (এই পরিবেশের standing
     constraint — ADR 0006)।
  ৩. Failure নিয়ে যুক্তি করার এক জায়গা — Ollama server চালু না থাকা, model
     pull না করা, timeout — সবকিছু ONE try/except-এ funnel হয়।

HIGHLIGHTS — module-এর top-এ না করে function-এর ভেতরে ``ollama`` import কেন?
src/embeddings/service.py-এর "get_model()-এর ভেতরে sentence_transformers
import করা" সিদ্ধান্তের প্রতিফলন, একই কারণে: এই module import করা (যেমন
কোনো test থেকে) কখনোই চালু Ollama server দাবি করা উচিত না — শুধু
``generate()`` CALL করলেই পারে।
"""

from __future__ import annotations

from config import get_config
from src.utils.logger import get_logger

logger = get_logger("llm.client")


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server can't be reached or the
    requested model isn't available.

    HIGHLIGHTS: ``ollama`` যা raise করে সেটা raw propagate হতে না দিয়ে একটা
    dedicated exception type থাকার মানে app.py শুধু ONE জিনিস catch করে একটা
    friendly "Ollama চালু আছে তো? `ollama pull llama3.1:8b` করেছ?" মেসেজ
    দেখাতে পারে — Streamlit UI-তে একটা low-level connection-refused
    traceback leak হওয়ার বদলে।
    """


def generate(
    prompt: str,
    system_prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    model: str | None = None,
) -> str:
    """Call the local Ollama model and return its text response.

    Args:
        prompt: The user-turn content — already assembled by
            src/rag/pipeline.py as the "augmented prompt" (retrieved
            context + question, Open-Book-Exam style).
        system_prompt: The system-turn content — the "answer strictly from
            the given context, and admit when you can't" guardrail
            instruction (see src/rag/pipeline.py's SYSTEM_PROMPT).
        temperature: Sampling temperature. Defaults to
            ``config.llm.default_temperature`` if not given.
        top_p: Nucleus-sampling threshold. Defaults to
            ``config.llm.default_top_p`` if not given.
        model: Ollama model tag to call. Defaults to ``config.llm.model``
            (llama3.1:8b) if not given — pass ``config.llm.fallback_model``
            explicitly for low-RAM machines.

    Returns:
        The model's response text (stripped of leading/trailing whitespace).

    Raises:
        OllamaConnectionError: if the Ollama server isn't running, the
            model isn't pulled, or the request otherwise fails.

    HIGHLIGHTS — কেন temperature ডিফল্ট এত LOW (config.py-তে 0.2)?
    Week 11-এর brand-copy generator "creative" আউটপুট চেয়েছিল
    (temperature=0.8) — কিন্তু RAG-এ generation-এর কাজ ভিন্ন: retrieved
    context-এর প্রতি বিশ্বস্ত থাকা, নতুন কিছু "কল্পনা" করা না (lecture-এর
    section 4.1-এর guardrail যুক্তি)। নিচু temperature মডেলকে বেশি
    deterministic/grounded রাখে, high-probability (context-সমর্থিত) টোকেনের
    দিকে ঝুঁকিয়ে — hallucination কমানোর একটা extra, prompt-independent লেয়ার।
    """
    cfg = get_config().llm
    resolved_model = model or cfg.model
    resolved_temperature = cfg.default_temperature if temperature is None else temperature
    resolved_top_p = cfg.default_top_p if top_p is None else top_p

    import ollama  # local import — see module HIGHLIGHTS above

    logger.info(
        "Calling Ollama model=%r temperature=%.2f top_p=%.2f",
        resolved_model,
        resolved_temperature,
        resolved_top_p,
    )

    try:
        response = ollama.chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": resolved_temperature,
                "top_p": resolved_top_p,
            },
        )
    except Exception as exc:  # noqa: BLE001
        # HIGHLIGHTS: ollama কয়েক ধরনের exception raise করে (connection
        # error, missing-model-এর জন্য ResponseError, ইত্যাদি) — আমরা
        # ইচ্ছাকৃতভাবে সবগুলোকে ONE OllamaConnectionError-এ funnel করছি,
        # যাতে caller শুধু একটা জিনিস handle করলেই চলে (উপরে class
        # docstring দেখুন)।
        logger.error("Ollama call failed: %s", exc)
        raise OllamaConnectionError(
            f"Could not reach Ollama or run model {resolved_model!r}. "
            f"Is the Ollama server running (`ollama serve`), and has the "
            f"model been pulled (`ollama pull {resolved_model}`)? "
            f"Original error: {exc}"
        ) from exc

    content = response["message"]["content"]
    return content.strip()
