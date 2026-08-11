"""
Thin wrapper around the ``ollama`` Python package: the ONLY module in this
project that imports ``ollama`` or knows Ollama's request/response shape.

HIGHLIGHTS — কেন Ollama-কে এখানে isolate করা, app.py থেকে সরাসরি call না করে?
Week 10-এর src/embeddings/service.py sentence-transformers wrap করার মতোই
একই boundary যুক্তি: app.py, src/prompting/templates.py, আর
src/security/injection_guard.py কখনো ``ollama`` import করে না, কখনো একটা
Ollama response dict দেখে না। তারা শুধু এই module-এর ``generate()`` function
call করে, যেটা plain string/float নেয় আর plain string ফেরত দেয়। তিনটা
concrete লাভ:
  ১. Swappability — ভবিষ্যতের কোনো class (বা student-এর নিজের project)
     Ollama-র বদলে অন্য local runtime (যেমন llama.cpp-র server, বা তুলনার
     জন্য cloud API) ব্যবহার করতে চাইলে শুধু এই file বদলাতে হবে — app.py,
     prompting logic, injection-defense logic — সব অপরিবর্তিত থাকে।
  ২. Testability — tests/test_pipeline.py ঠিক একটা জিনিসই mock করে —
     ``ollama.chat`` (বা এই module-এর সেটার call) — বাকি প্রতিটা layer
     (prompt construction, few-shot templating, injection delimiting,
     output validation) সেই একটা seam-এর বিপরীতে test হয়, পুরো suite-এ
     কোথাও real network/server call ছাড়াই।
  ৩. Failure নিয়ে যুক্তি করার এক জায়গা — Ollama server চালু না থাকা, model
     pull না করা, timeout — এগুলো Ollama-specific failure mode, যেগুলো
     প্রতিটা call site-এ ছড়িয়ে না থেকে ONE try/except-এ থাকা উচিত।

HIGHLIGHTS — module-এর top-এ না করে function-এর ভেতরে ``ollama`` import কেন?
Week 10-এর "get_model()-এর ভেতরে sentence_transformers import করা" সিদ্ধান্তেরই
প্রতিফলন, একই কারণে: এই module import করা (যেমন কোনো test থেকে, বা
`python -c "import src.llm.client"` থেকে) কখনোই চালু Ollama server দাবি করা
উচিত না। ``ollama`` package নিজে একটা lightweight HTTP client (import করলেই
কোথাও connect হয়ে যায় না), কিন্তু import-টা generate()-এর ভেতরে local রাখা
একটা visual signal — এই file-এর top-level import গুলো দেখলেই কেউ বুঝবে কোথাও
``ollama`` import নেই, তাই এই module import করলে network touch হওয়া অসম্ভব —
শুধু generate() CALL করলেই হতে পারে।
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
    দেখাতে পারে — student-এর Streamlit UI-তে একটা low-level connection-
    refused traceback leak হওয়ার বদলে।
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
        prompt: The user-turn content (already assembled by
            src/prompting/templates.py — zero-shot/few-shot/CoT-wrapped,
            and if it contains untrusted user text, already delimited by
            src/security/injection_guard.py).
        system_prompt: The system-turn content, setting the assistant's
            role/instructions (also usually built with
            instruction-hierarchy framing by injection_guard.py).
        temperature: Sampling temperature (0-2). Defaults to
            ``config.llm.default_temperature`` if not given.
        top_p: Nucleus-sampling threshold (0-1). Defaults to
            ``config.llm.default_top_p`` if not given.
        model: Ollama model tag to call. Defaults to
            ``config.llm.model`` (llama3.1:8b) if not given — pass
            ``config.llm.fallback_model`` explicitly for low-RAM machines.

    Returns:
        The model's response text (stripped of leading/trailing
        whitespace).

    Raises:
        OllamaConnectionError: if the Ollama server isn't running, the
            model isn't pulled, or the request otherwise fails.

    HIGHLIGHTS — MODEL REFERENCE TABLE (Week 11 lecture, section 4.1 —
    "কোন লোকাল LLM কখন ব্যবহার করবেন" থেকে adapt করা):

    | Model                              | Use Case                                        | Advantage                                              | Limitation                                                          |
    |--------------------------------------|--------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------------------|
    | `llama3.1:8b` (used here, default)   | General-purpose text generation, instruction-following | Strong output quality, runs comfortably on 16GB RAM        | Tight on 8GB RAM machines, can be slow there                          |
    | `llama3.2:3b` (fallback)              | Low-resource machines                             | Small, runs fine even on 8GB RAM                            | Less accurate than the 8B model, weaker on complex instructions        |
    | `mistral:7b`                          | Fast inference, code/structured tasks             | Similar size to llama3.1, faster on some benchmarks         | Smaller community/fine-tune ecosystem than Llama                       |
    | OpenAI GPT-4 / Claude (Cloud API)     | When you need best-in-class output quality        | State-of-the-art benchmark performance                     | Not local — per-call cost, data leaves the machine, conflicts with this course's Sovereign AI philosophy |

    আমরা ডিফল্ট হিসেবে `llama3.1:8b` বেছে নিয়েছি (config.py-এর
    ``LLMConfig.model``), কারণ এটা একটা portfolio project যেটা PROMPTING
    technique (zero/few-shot, CoT, injection defense) দেখানোর জন্য, raw
    model capability দেখানোর জন্য না — llama3.1:8b-এর instruction-following
    এই technique গুলো স্পষ্টভাবে দেখানোর জন্য যথেষ্টের চেয়ে বেশি শক্তিশালী,
    সম্পূর্ণ offline, কোনো API key ছাড়া, প্রতি-request খরচ ছাড়া। 8GB-RAM
    মেশিনের কোনো student-এর ``config.llm.fallback_model`` (llama3.2:3b)-এ
    switch করা উচিত — config.py-তে এক লাইন বদল, আর কোথাও code বদলাতে হবে না।

    HIGHLIGHTS — ``temperature`` আর ``top_p``-কে একটা "creativity" স্লাইডারের
    বদলে দুটো আলাদা parameter হিসেবে কেন expose করা হলো?
    এরা আসলে ভিন্ন ভিন্ন জিনিস control করে (lecture-এর section 3.4 দেখুন):
    temperature পুরো probability distribution reshape করে (flatten করে
    দেয়, ফলে কম-probability token গুলোও likely হয়ে ওঠে), আর top_p শুধু
    sample করা candidate token set-এর SIZE বদলায়। দুটোকে একটা "creativity:
    low/medium/high" knob-এ মিশিয়ে দিলে এই পার্থক্যটা লুকিয়ে যেত, আর
    student কখনো দেখতে পেত না যে "low temperature + wide top_p" কীভাবে
    "high temperature + narrow top_p"-এর চেয়ে আলাদা আচরণ করে — দুটোর জন্য
    সরাসরি slider থাকাটাই app.py-এর Generate tab-এ lecture-এর এই পার্থক্যকে
    observable করে তোলে।
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
        # error, missing model-এর জন্য ResponseError, ইত্যাদি); আমরা
        # ইচ্ছাকৃতভাবে সবগুলোকে ONE OllamaConnectionError-এ funnel করছি,
        # যাতে caller শুধু একটা জিনিস handle করলেই চলে। কারণ উপরে class
        # docstring-এ আছে।
        logger.error("Ollama call failed: %s", exc)
        raise OllamaConnectionError(
            f"Could not reach Ollama or run model {resolved_model!r}. "
            f"Is the Ollama server running (`ollama serve`), and has the "
            f"model been pulled (`ollama pull {resolved_model}`)? "
            f"Original error: {exc}"
        ) from exc

    # ollama.chat() returns a dict-like object with response["message"]["content"].
    content = response["message"]["content"]
    return content.strip()
