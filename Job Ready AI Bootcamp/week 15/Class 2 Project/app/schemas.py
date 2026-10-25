"""Pydantic v2 request/response schemas — the contract that /docs (Swagger) shows.

HIGHLIGHTS: Grounding schema-র অংশ — ``found``, ``degraded``, ``citations`` field
হিসেবে আছে, তাই client prose পার্স না করেই জানে উত্তরটা document-ভিত্তিক কিনা।
"""
from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["How many sick days do employees get?"])
    source: str | None = Field(default=None, description="Search only this document (null = all)")
    top_k: int | None = Field(default=None, ge=1, le=10)


class Citation(BaseModel):
    n: int = Field(description="Passage number as used in the answer, e.g. [1]")
    source: str
    page: int
    chunk_id: str
    distance: float
    snippet: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_s: float = 0.0
    ttft_ms: float | None = None
    load_ms: float = 0.0          # > 0 = model was (re)loaded: cold start
    prefill_ms: float = 0.0       # prompt processing; small when Ollama reuses a cached prompt prefix


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    vote: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=500)


class AskResponse(BaseModel):
    answer: str
    found: bool = Field(description="false = nothing relevant was retrieved; the LLM was not called")
    degraded: bool = Field(default=False, description="true = LLM unavailable; citations are the best passages")
    cached: bool = False
    citations: list[Citation] = []
    timings_ms: dict[str, float] = {}
    usage: Usage | None = None
    request_id: str | None = None


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    new_chunks: int = Field(description="0 = this exact document was already indexed (idempotent upload)")
    request_id: str | None = None


class SourcesResponse(BaseModel):
    sources: list[str]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    backend: str
    llm: str
    indexed_documents: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
