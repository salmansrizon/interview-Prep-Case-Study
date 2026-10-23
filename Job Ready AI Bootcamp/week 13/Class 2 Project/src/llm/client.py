"""
Ollama chat client — the only module that imports ``ollama``.

HIGHLIGHTS: Agent শুধু ``chat(messages, tools=..., stop=...) -> dict`` callable
চায়, তাই tests-এ real Ollama-র বদলে scripted fake LLM বসানো যায় (ADR 0006)।
Ollama বন্ধ থাকলে বা model pull না করা থাকলে সবকিছু একটাই ``LLMUnavailable``
exception-এ আসে। Return dict-এ token count-ও থাকে (``prompt_eval_count``),
যাতে প্রতি step-এ context কত বড় হচ্ছে trace-এ দেখা যায়।
"""
from __future__ import annotations

from config import get_config


class LLMUnavailable(RuntimeError):
    pass


def chat(messages: list[dict], tools: list[dict] | None = None, stop: list[str] | None = None,
         model: str | None = None) -> dict:
    """One chat turn -> {"content", "tool_calls": [{"name", "arguments"}], "prompt_tokens", "output_tokens"}.

    ``tools`` = native tool calling. ``stop`` = text-ReAct mode: generation halts
    before the model can write its own fake ``Observation:``.
    """
    cfg = get_config().llm
    name = model or cfg.model
    options = {"temperature": cfg.temperature, "seed": cfg.seed,
               "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict}
    if stop:
        options["stop"] = stop
    try:
        import ollama

        resp = ollama.Client(host=cfg.host, timeout=cfg.timeout_seconds).chat(
            model=name, messages=messages, tools=tools or None, options=options)
    except Exception as e:  # noqa: BLE001 - connection/model errors come in many types
        raise LLMUnavailable(
            f"Ollama call failed ({e}). Is `ollama serve` running and `ollama pull {name}` done?") from e
    msg = resp.message
    return {
        "content": msg.content or "",
        "tool_calls": [{"name": c.function.name, "arguments": dict(c.function.arguments or {})}
                       for c in (msg.tool_calls or [])],
        "prompt_tokens": getattr(resp, "prompt_eval_count", None),
        "output_tokens": getattr(resp, "eval_count", None),
    }
