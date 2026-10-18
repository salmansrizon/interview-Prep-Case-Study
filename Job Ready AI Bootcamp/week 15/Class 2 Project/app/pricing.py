"""Equivalent cloud cost — local model free, তাই "যদি cloud API হতো তাহলে কত লাগত" হিসাব।

ILLUSTRATIVE ONLY: নিচের দামগুলো শেখার জন্য গোল সংখ্যা, আসল দাম নয়। আসল price
provider-এর pricing page থেকে নিয়ে ``RAG_PRICE_TABLE_JSON`` env দিয়ে override করুন।

HIGHLIGHTS — Price table reference (USD / 1M tokens, illustrative):
| Key (equivalent)     | Use case                | Advantage                 | Limitation                      |
| small-cloud (chosen) | সস্তা RAG answer তুলনা    | কম খরচ, দ্রুত              | reasoning মাঝারি                 |
| large-cloud          | কঠিন reasoning           | সর্বোচ্চ quality            | ১০-২০x দামি                      |
| local (Ollama)       | আসল চলমান model          | খরচ ০, data বাইরে যায় না   | নিজের hardware/electricity লাগে  |
"""

import json
import os

# (input, output) USD per 1M tokens — ILLUSTRATIVE
DEFAULT_TABLE: dict[str, tuple[float, float]] = {
    "small-cloud": (0.15, 0.60),
    "large-cloud": (3.00, 15.00),
    "local": (0.0, 0.0),
}


def price_table() -> dict[str, tuple[float, float]]:
    raw = os.environ.get("RAG_PRICE_TABLE_JSON")
    return {**DEFAULT_TABLE, **{k: tuple(v) for k, v in json.loads(raw).items()}} if raw else DEFAULT_TABLE


def equivalent_cost(price_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = price_table().get(price_key, DEFAULT_TABLE["small-cloud"])
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
