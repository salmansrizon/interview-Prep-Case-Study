"""Ollama token count surface করা — Week 12 কোড না ছুঁয়ে।

Week 12 ``generate()`` শুধু text ফেরত দেয়, কিন্তু Ollama response-এ
``prompt_eval_count`` (input token) আর ``eval_count`` (output token) থাকে।

HIGHLIGHTS: Week 12 ``generate()`` প্রতি call-এ ``ollama.chat`` attribute lookup করে,
তাই ``ollama.chat``-কে wrap করলেই response পথে count ধরা যায়। Request আলাদা
threadpool thread-এ চলে, তাই ``threading.local`` ব্যবহার — এক request-এর count
অন্যটায় মিশে না। Alternative: Week 12 ``generate()`` বদলানো (সহজ, কিন্তু Week 12
ছোঁয়া), অথবা Ollama HTTP proxy (কোনো code বদলায় না, কিন্তু আলাদা process)।
"""

import threading

_local = threading.local()


def record_usage(response) -> None:
    get = response.get if hasattr(response, "get") else lambda k, d=None: getattr(response, k, d)
    _local.usage = {
        "prompt_tokens": int(get("prompt_eval_count", 0) or 0),
        "completion_tokens": int(get("eval_count", 0) or 0),
    }


def pop_usage() -> dict | None:
    u, _local.usage = getattr(_local, "usage", None), None
    return u


def install_ollama_usage_capture() -> None:
    import ollama

    if getattr(ollama.chat, "_usage_wrapped", False):
        return
    orig = ollama.chat

    def chat(*a, **kw):
        resp = orig(*a, **kw)
        record_usage(resp)
        return resp

    chat._usage_wrapped = True
    ollama.chat = chat
