"""Pydantic v2 request/response schemas — the contract shown at /docs.

HIGHLIGHTS: Safety schema-র অংশ: ``route`` চারটার একটা (Literal), ``rule`` বলে কোন নিয়মে সিদ্ধান্ত হলো,
``redaction_count`` বলে কতগুলো PII মুছে ফেলা হয়েছে। Response-এ raw প্রশ্ন ফেরত যায় না — শুধু
``redacted_question``, যাতে user দেখে কী সরানো হলো।
"""
from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["answered", "abstained", "escalated", "refused"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["How do I prepare oral rehydration solution?"])


class Citation(BaseModel):
    n: int = Field(description="Passage number as used in the answer, e.g. [1]")
    id: str = Field(examples=["kb:dehydration:3"])
    page: str
    section: str
    distance: float = Field(description="Cosine distance to the question (0 = closest)")
    snippet: str


class AskResponse(BaseModel):
    route: Route
    answer: str
    rule: str | None = Field(default=None, description="Rule that decided the route, e.g. chest_pain, below_threshold")
    citations: list[Citation] = []
    redaction_count: int = 0
    redactions: dict[str, int] = {}
    redacted_question: str
    flags: list[str] = []
    disclaimer: str
    timings_ms: dict[str, float] = {}
    request_id: str | None = None


class PageInfo(BaseModel):
    file: str
    page: str
    sections: int


class DocumentsResponse(BaseModel):
    pages: list[PageInfo]


class IngestResponse(BaseModel):
    pages: int
    sections: int
    removed: int = Field(description="Stale sections deleted because they are no longer in data/kb")


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    backend: str
    llm: str
    indexed_sections: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
