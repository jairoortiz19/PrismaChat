from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
import aiofiles
import os

from app.models.document import (
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentUploadResponse,
    IngestRequest,
    IngestResponse,
)
from app.services.document_service import get_document_service
from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError, document_processing_error
from app.core.logging import get_logger

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
logger = get_logger()


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """
    List all indexed documents.

    Returns information about all documents that have been processed
    and added to the vector store.
    """
    document_service = get_document_service()
    documents = await document_service.list_documents()

    return DocumentListResponse(
        documents=documents,
        total_count=len(documents),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest = None):
    """
    Ingest all documents from the documents directory.

    Processes all supported files (PDF, TXT, MD, DOCX) from the
    configured documents directory and adds them to the vector store.
    """
    try:
        document_service = get_document_service()
        directory = request.directory if request else None
        result = await document_service.ingest_directory(directory)

        if not result.success and not result.documents_processed:
            logger.warning(f"No documents ingested: {result.errors}")

        return result
    except DocumentProcessingError as e:
        logger.error(f"Document ingestion error: {e}")
        raise document_processing_error(str(e))
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a single document.

    Supported formats: PDF, TXT, MD, DOCX
    """
    settings = get_settings()
    document_service = get_document_service()

    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".doc"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save file to documents directory
    documents_dir = Path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)

    file_path = documents_dir / file.filename

    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"File saved: {file_path}")

        # Process the file
        result = await document_service.process_file(file_path)
        return result

    except DocumentProcessingError as e:
        # Clean up file if processing failed
        if file_path.exists():
            os.remove(file_path)
        logger.error(f"Failed to process uploaded file: {e}")
        raise document_processing_error(str(e))
    except Exception as e:
        if file_path.exists():
            os.remove(file_path)
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document from the vector store.

    Note: This removes the document from the search index but
    does not delete the original file from disk.
    """
    try:
        document_service = get_document_service()
        success = await document_service.delete_document(document_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found",
            )

        return DocumentDeleteResponse(
            success=True,
            document_id=document_id,
            message=f"Document '{document_id}' deleted successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
