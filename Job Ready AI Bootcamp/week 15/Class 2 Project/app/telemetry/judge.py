"""Offline, sampled LLM-as-judge (tier 3): is every claim in the answer supported by its passages?

Run:  python -m app.telemetry.judge --limit 20        # judges the newest un-judged answers, writes back to SQLite

HIGHLIGHTS:
* **Offline, not in the request path:** Ollama একসাথে একটাই request চালায় — judge request-এর ভেতরে
  চললে user-এর latency দ্বিগুণ। তাই log থেকে পরে, sample করে, batch-এ।
* **Exact passages:** log-এ ``chunk_ids`` থাকে; judge ঠিক সেই passage-গুলো Chroma থেকে পড়ে।
* **Structured output:** Ollama ``format=<JSON schema>`` → সবসময় parse-যোগ্য JSON।
* **Claim-by-claim, evidence first:** answer-কে আলাদা claim-এ ভাঙা, প্রতিটার জন্য passage থেকে হুবহু
  evidence quote, তারপর রায়। Code-এ আরেকটা deterministic check: "supported" claim-এর quote সত্যিই
  passage-এ আছে কিনা (judge-এর বানানো সমর্থন ধরে)।
  Live-এ যা শেখা গেছে (llama3.1:8b, ১১টা উত্তর, ২টা ইচ্ছাকৃত ভুল): শুধু ``supported: bool`` চাইলে
  judge একবার ``true`` বলে একই উত্তরে unsupported claim লিখেছিল; "unsupported claim-এর তালিকা দাও"
  চাইলে সব claim-কেই তালিকায় দিয়েছিল (১১টার ১১টাই fail)। Claim-by-claim + evidence: ১১/১১ ঠিক।
* **Bias:** এখানে judge আর answerer একই model (llama3.1:8b) → self-preference bias এড়ানো যায় না।
  Score-কে absolute সত্য না, *trend* হিসেবে পড়ুন, আর মাঝে মাঝে মানুষের label-এর সাথে মিলিয়ে দেখুন।
"""
from __future__ import annotations

import argparse
import json
import re

from pydantic import BaseModel

JUDGE_SYSTEM = (
    "You are a strict fact-checker. Split the ANSWER into its factual claims (ignore citation markers like [1]). "
    "For each claim, copy the exact sentence from the PASSAGES that supports it into `evidence` (empty string if "
    "none), then set `supported`: true only if the evidence states the same fact (a number must have the same "
    "value, e.g. twelve = 12).")


class Claim(BaseModel):
    claim: str
    evidence: str                    # field order = generation order: evidence BEFORE the verdict
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim]

    def problems(self, passages: list[str]) -> list[str]:
        """Unsupported claims, plus "supported" claims whose quote is not really in the passages."""
        if not self.claims:
            return ["judge returned no claims"]
        text = _norm(" ".join(passages))
        out = [f"unsupported: {c.claim}" for c in self.claims if not c.supported]
        out += [f"quote not in passages: {c.claim}" for c in self.claims
                if c.supported and (not c.evidence.strip() or _norm(c.evidence) not in text)]
        return out


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def judge_one(client, model: str, question: str, answer: str, passages: list[str]) -> list[str]:
    """Returns the problems found ([] = every claim supported)."""
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    r = client.chat(model=model, format=Verdict.model_json_schema(),
                    options={"temperature": 0, "seed": 42, "num_ctx": 8192},
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": f"PASSAGES:\n{ctx}\n\nQUESTION: {question}\n\nANSWER: {answer}"}])
    return Verdict.model_validate_json(r.message.content).problems(passages)


def run(log, store, client, model: str, limit: int) -> list[dict]:
    out = []
    for row in log.unjudged(limit):
        ids = [i for i in (row["chunk_ids"] or "").split(",") if i]
        texts = store.texts(ids)
        if not texts or (row["question"] or "").startswith("sha256:"):
            log.set_judge(row["id"], None, "skipped: passages or text not available")
            continue
        try:
            problems = judge_one(client, model, row["question"], row["answer"], [texts[i] for i in ids if i in texts])
            log.set_judge(row["id"], not problems, "; ".join(problems)[:300])
            out.append({"id": row["id"], "question": row["question"], "supported": not problems, "problems": problems})
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the batch
            out.append({"id": row["id"], "question": row["question"], "supported": None, "error": str(exc)[:120]})
    return out


def main() -> None:
    import ollama

    from ..rag.components import ChromaStore
    from ..settings import get_settings
    from .store import RequestLog

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()
    s = get_settings()
    results = run(RequestLog(s.monitor_db), ChromaStore(str(s.chroma_dir), s.collection),
                  ollama.Client(host=s.ollama_host, timeout=s.llm_timeout_s), s.model, a.limit)
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print(f"judged {sum(r.get('supported') is not None for r in results)} of {len(results)}")


if __name__ == "__main__":
    main()
