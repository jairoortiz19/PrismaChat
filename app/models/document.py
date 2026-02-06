from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    chunk_count: int
    ingested_at: datetime
    metadata: dict = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    chunks_created: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    success: bool
    document_id: str
    message: str


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(
        default=None, description="Directory path to ingest documents from. If None, uses default documents directory"
    )


class IngestResponse(BaseModel):
    success: bool
    documents_processed: int
    total_chunks: int
    errors: list[str] = Field(default_factory=list)
