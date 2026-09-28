"""Shared domain and API models."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    title: str
    source: str
    text: str


class Chunk(BaseModel):
    id: str
    document_id: str
    title: str
    source: str
    text: str
    position: int


class SearchResult(BaseModel):
    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    REJECTED = "rejected"


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str
    quote: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class AskResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
