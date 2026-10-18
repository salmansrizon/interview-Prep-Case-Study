"""Langfuse sink (optional) — env key/host না থাকলে বা SDK না থাকলে ``None``, app স্বাভাবিক চলে।

Env: ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_HOST`` (বা ``LANGFUSE_BASE_URL``)।

HIGHLIGHTS — LLM observability tool reference:
| Tool              | Use case                       | Advantage                      | Limitation                            |
| Langfuse (chosen) | self-host, open source tracing | data নিজের কাছে, cost+score UI   | compose-এ ৫-৬ service (ভারী)            |
| LangSmith         | LangChain ecosystem            | LangChain-এর সাথে গভীর          | closed SaaS, self-host enterprise-only |
| Arize Phoenix     | local eval + OpenTelemetry     | হালকা, notebook-এ চলে           | dashboard/team feature কম             |
| Helicone          | proxy-based logging            | code বদলায় না (URL বদল)        | proxy-র ওপর নির্ভর, deep trace কম       |
| SQLite (fallback) | কোনো infra ছাড়া                | zero setup, এই project-এ আছে   | UI/alert/multi-user নেই               |

HIGHLIGHTS: observer চলে request শেষে, তাই span-এর নিজস্ব duration প্রায় শূন্য; আসল
latency ``metadata.latency_s``-এ রাখা হয়। সঠিক duration চাইলে ``_run()`` এর আগে span খুলতে হয়
(seam বদলাতে হবে)। ``cost_details`` এখানে illustrative cloud-equivalent।
"""

import os


def make_langfuse_sink(client=None):
    if client is None:
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
                and (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL"))):
            return None
        try:
            from langfuse import Langfuse
            client = Langfuse()
        except Exception:
            return None

    def sink(m: dict, rec) -> None:
        if m["op"] != "ask" or not m["ok"]:
            return
        gen = client.start_observation(
            name="rag-ask", as_type="generation", model=m["model"], input=m["question"],
            output=rec.result.answer,
            usage_details={"input": m["prompt_tokens"] or 0, "output": m["completion_tokens"] or 0},
            cost_details={"total": m["cost_usd"] or 0.0},
            metadata={"latency_s": m["latency_s"], "request_id": m["request_id"], "found": m["found"]},
        )
        gen.score_trace(name="grounding", value=m["quality"], comment=m["quality_label"])
        gen.end()

    return sink
