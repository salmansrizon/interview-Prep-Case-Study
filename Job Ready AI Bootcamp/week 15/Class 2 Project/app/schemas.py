"""Pydantic v2 request/response schemas — এগুলোই /docs (Swagger)-এর contract।"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What is the leave policy?"])
    source: str | None = Field(default=None, description="শুধু এই PDF-এর ভেতরে খোঁজো (null = সব)")
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    source: str
    page: int
    distance: float


class AskResponse(BaseModel):
    answer: str
    found: bool = Field(description="false = guardrail trigger করেছে, LLM call হয়নি")
    citations: list[CitationOut] = []
    request_id: str | None = None
    latency_ms: float | None = None
    # HIGHLIGHTS: নিচের ৪টা field শুধু monitoring observer-এর জন্য; exclude=True তাই API response-এ যায় না।
    usage: dict | None = Field(default=None, exclude=True)  # prompt_tokens / completion_tokens
    context: list[str] = Field(default_factory=list, exclude=True)  # retrieved chunk text (grounding score)
    question: str | None = Field(default=None, exclude=True)


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    request_id: str | None = None


class SourcesResponse(BaseModel):
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    backend: str
    indexed_documents: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
