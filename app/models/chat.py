from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the chatbot")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation context")


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceDocument] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatStreamChunk(BaseModel):
    content: str
    is_final: bool = False
    session_id: Optional[str] = None
    sources: Optional[list[SourceDocument]] = None
