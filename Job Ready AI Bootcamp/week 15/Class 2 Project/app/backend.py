"""Backend seam #1: Week 12 pipeline-কে একটা ছোট sync interface-এর পেছনে রাখা।

Approach (README-তেও লেখা): Week 12 কোড COPY করা হয়নি, import করা হয়েছে। Week 12
`config` আর `src.*` নামে top-level import করে, তাই তার root-কে sys.path-এর
**শেষে** যোগ করি (আমাদের `app` package আগে resolve হয়, Week 12-এর streamlit
`app.py` ছায়ায় থাকে)। Chroma path override করা হয় যাতে week 12-এর data না বদলায়।

Backend interface (duck typing):
    ask(question, source, top_k) -> AskResponse   # BLOCKING
    ingest(filename, data: bytes) -> int          # BLOCKING
    sources() -> list[str]                        # BLOCKING

HIGHLIGHTS — কেন এগুলো plain ``def``, ``async def`` না? Ollama, Chroma,
sentence-transformers সবই blocking। service layer এগুলো threadpool-এ চালায়
(service.py)। Backend নিজে async হওয়ার ভান করলে event loop আটকে যেত।

HIGHLIGHTS — ASGI server reference (কোনটা কখন):
| Server             | Use case                     | Advantage                   | Limitation                     |
| Uvicorn (chosen)   | dev + single-node production | সহজ, দ্রুত (uvloop)         | multi-process নিজে manage করে না |
| Gunicorn + Uvicorn | production multi-worker      | process supervision, restart | Unix only; Windows-এ চলে না     |
| Hypercorn          | HTTP/2, HTTP/3 দরকার হলে     | trio support                | ecosystem ছোট                  |
| Granian            | Rust-based, কম overhead      | নতুন, দ্রুত                 | কম battle-tested               |

HIGHLIGHTS — LLM reference (Week 12 config থেকে; এখানে বদলানো হয়নি):
| Model            | Use case              | Advantage              | Limitation                    |
| llama3.1:8b (chosen) | সাধারণ RAG answer | ভালো quality, local   | ~8GB RAM, CPU-তে ধীর          |
| llama3.2:3b      | low-RAM machine       | দ্রুত, ছোট            | reasoning দুর্বল               |
| qwen2.5:7b       | multilingual (Bangla) | বহুভাষিক ভালো          | llama-র মতোই RAM লাগে         |
| Cloud API (GPT/Claude) | সর্বোচ্চ quality | কোনো local compute নেই | cost, data বাইরে যায়          |
"""

import sys
import time
from dataclasses import replace

from .errors import BadDocument, LLMUnavailable
from .schemas import AskResponse, CitationOut
from .settings import Settings
from .usage import install_ollama_usage_capture, pop_usage


class RealBackend:
    def __init__(self, settings: Settings):
        week12 = str(settings.week12_dir)
        if week12 not in sys.path:
            sys.path.append(week12)  # append (prepend না) — ওপরের docstring দেখুন
        import config as w12cfg  # Week 12 config

        cfg = w12cfg.get_config()
        vs = replace(cfg.vectorstore, persist_directory=str(settings.chroma_dir))
        # Week 12 singleton override: data এখন rag-api/data-তে যায়
        w12cfg._config_instance = replace(cfg, vectorstore=vs)

        from src.chunking.chunker import chunk_page_texts
        from src.extraction.pdf_extractor import ExtractionError, extract_pages
        from src.llm.client import OllamaConnectionError
        from src.rag.pipeline import answer_question
        from src.vectorstore.store import index_chunks, list_indexed_sources

        install_ollama_usage_capture()  # Week 12 না ছুঁয়ে token count ধরা (usage.py)
        self._chunk, self._extract = chunk_page_texts, extract_pages
        self._answer, self._index, self._list = answer_question, index_chunks, list_indexed_sources
        self._ExtractionError, self._OllamaErr = ExtractionError, OllamaConnectionError

    def ask(self, question, source=None, top_k=None) -> AskResponse:
        pop_usage()  # আগের call-এর stale count সাফ
        try:
            r = self._answer(question, source_filter=source, top_k=top_k)
        except self._OllamaErr as exc:
            raise LLMUnavailable(str(exc)) from exc
        cites = [CitationOut(source=c.source, page=c.page, distance=c.distance) for c in r.citations]
        return AskResponse(answer=r.answer, found=r.found, citations=cites, usage=pop_usage(),
                           context=[c.text for c in r.retrieved_chunks], question=question)

    def ingest(self, filename, data: bytes) -> int:
        try:
            pages = self._extract(data, filename=filename)
        except self._ExtractionError as exc:
            raise BadDocument(str(exc)) from exc
        return self._index(self._chunk(pages, source=filename))

    def sources(self) -> list[str]:
        return self._list()


class FakeBackend:
    """Ollama/Chroma ছাড়া চলে — tests আর benchmark-এর জন্য।

    ``delay`` সেকেন্ড ``time.sleep`` করে LLM latency নকল করে (blocking, ঠিক
    আসল Ollama call-এর মতো)।
    """

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self._docs: dict[str, int] = {}
        self.context_override: list[str] | None = None  # tests: grounding নিয়ন্ত্রণ
        self.answer_override: str | None = None

    def ask(self, question, source=None, top_k=None) -> AskResponse:
        time.sleep(self.delay)
        if not self._docs or (source and source not in self._docs):
            return AskResponse(answer="This information was not found in the documents.", found=False, question=question)
        src = source or sorted(self._docs)[0]
        answer = self.answer_override or f"Fake answer to: {question}"
        return AskResponse(
            answer=answer, found=True, citations=[CitationOut(source=src, page=1, distance=0.5)],
            usage={"prompt_tokens": 40 + len(question.split()), "completion_tokens": len(answer.split())},
            context=self.context_override if self.context_override is not None else [f"context about {question}"],
            question=question,
        )

    def ingest(self, filename, data: bytes) -> int:
        if not data.startswith(b"%PDF"):
            raise BadDocument(f"{filename} is not a readable PDF")
        self._docs[filename] = 3
        return 3

    def sources(self) -> list[str]:
        return sorted(self._docs)


def make_backend(settings: Settings):
    return FakeBackend(settings.fake_delay_s) if settings.backend == "fake" else RealBackend(settings)
