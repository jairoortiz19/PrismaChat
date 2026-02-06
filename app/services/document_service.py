from pathlib import Path
from typing import Optional
from datetime import datetime
import hashlib

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.models.document import (
    DocumentInfo,
    DocumentUploadResponse,
    IngestResponse,
)
from app.repositories.vector_store import get_vector_store_instance
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import DocumentProcessingError


class DocumentService:
    """Service for document processing and ingestion"""

    def __init__(self):
        self._vector_store = get_vector_store_instance()
        self._settings = get_settings()
        self._logger = get_logger()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=len,
        )

    def _get_loader(self, file_path: Path):
        """Get appropriate loader based on file extension"""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        elif suffix in [".txt", ".md"]:
            return TextLoader(str(file_path), encoding="utf-8")
        elif suffix in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader(str(file_path))
        else:
            raise DocumentProcessingError(f"Unsupported file type: {suffix}")

    def _generate_source_id(self, file_path: Path) -> str:
        """Generate unique ID for document"""
        content_hash = hashlib.md5(file_path.name.encode()).hexdigest()[:8]
        return f"{file_path.stem}_{content_hash}"

    async def process_file(self, file_path: Path) -> DocumentUploadResponse:
        """Process a single file and add to vector store"""
        try:
            self._logger.info(f"Processing file: {file_path}")

            if not file_path.exists():
                raise DocumentProcessingError(f"File not found: {file_path}")

            # Load document
            loader = self._get_loader(file_path)
            documents = loader.load()

            # Split into chunks
            chunks = self._text_splitter.split_documents(documents)

            # Add metadata
            source_id = self._generate_source_id(file_path)
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "source": file_path.name,
                        "file_type": file_path.suffix,
                        "ingested_at": datetime.utcnow().isoformat(),
                    }
                )

            # Add to vector store
            chunk_count = await self._vector_store.add_documents(chunks, source_id)

            self._logger.info(f"Successfully processed {file_path.name}: {chunk_count} chunks")

            return DocumentUploadResponse(
                success=True,
                document_id=source_id,
                filename=file_path.name,
                chunks_created=chunk_count,
                message=f"Successfully processed {file_path.name}",
            )
        except Exception as e:
            self._logger.error(f"Error processing {file_path}: {e}")
            raise DocumentProcessingError(f"Failed to process {file_path.name}: {str(e)}")

    async def ingest_directory(self, directory: Optional[str] = None) -> IngestResponse:
        """Ingest all documents from a directory"""
        dir_path = Path(directory) if directory else Path(self._settings.documents_dir)

        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Created documents directory: {dir_path}")

        supported_extensions = {".pdf", ".txt", ".md", ".docx", ".doc"}
        files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

        if not files:
            return IngestResponse(
                success=True,
                documents_processed=0,
                total_chunks=0,
                errors=[f"No supported documents found in {dir_path}"],
            )

        documents_processed = 0
        total_chunks = 0
        errors = []

        for file_path in files:
            try:
                result = await self.process_file(file_path)
                documents_processed += 1
                total_chunks += result.chunks_created
            except DocumentProcessingError as e:
                errors.append(str(e))
                self._logger.error(f"Failed to process {file_path}: {e}")

        return IngestResponse(
            success=documents_processed > 0,
            documents_processed=documents_processed,
            total_chunks=total_chunks,
            errors=errors,
        )

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from vector store"""
        try:
            result = await self._vector_store.delete_by_source(document_id)
            if result:
                self._logger.info(f"Deleted document: {document_id}")
            return result
        except Exception as e:
            self._logger.error(f"Error deleting document {document_id}: {e}")
            raise DocumentProcessingError(f"Failed to delete document: {str(e)}")

    async def list_documents(self) -> list[DocumentInfo]:
        """List all indexed documents"""
        try:
            sources = await self._vector_store.get_all_sources()
            return [
                DocumentInfo(
                    id=s["source_id"],
                    filename=s["filename"],
                    file_type=Path(s["filename"]).suffix if "." in s["filename"] else "unknown",
                    size_bytes=0,  # Not tracked currently
                    chunk_count=s["chunk_count"],
                    ingested_at=datetime.utcnow(),  # Simplified
                )
                for s in sources
            ]
        except Exception as e:
            self._logger.error(f"Error listing documents: {e}")
            return []


# Singleton instance
_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
