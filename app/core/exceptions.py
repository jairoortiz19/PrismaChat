from fastapi import HTTPException, status


class ChatbotException(Exception):
    """Base exception for chatbot application"""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class LLMConnectionError(ChatbotException):
    """Raised when connection to Ollama fails"""

    pass


class DocumentProcessingError(ChatbotException):
    """Raised when document processing fails"""

    pass


class VectorStoreError(ChatbotException):
    """Raised when vector store operations fail"""

    pass


class SessionNotFoundError(ChatbotException):
    """Raised when session is not found"""

    pass


class SessionExpiredError(ChatbotException):
    """Raised when session has expired"""

    pass


# HTTP Exception helpers
def llm_connection_error(detail: str = "Could not connect to LLM service") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
    )


def document_processing_error(detail: str = "Error processing document") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=detail,
    )


def session_not_found_error(session_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Session '{session_id}' not found",
    )


def internal_server_error(detail: str = "Internal server error") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    )
