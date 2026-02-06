from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import json

from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import get_chat_service
from app.core.exceptions import LLMConnectionError, llm_connection_error
from app.core.logging import get_logger
from app.core.rate_limiter import get_chat_rate_limiter

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])
logger = get_logger()


@router.post("", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request):
    """
    Send a question to the chatbot and receive a response.

    The chatbot uses RAG (Retrieval Augmented Generation) to find relevant
    information from indexed documents and generate a contextual response.

    - **question**: The question to ask
    - **session_id**: Optional session ID for conversation context.

    Rate limited: 10 requests burst, refills at 0.5/s per client.
    """
    # Rate limiting
    limiter = get_chat_rate_limiter()
    if not limiter.check(request, cost=1):
        retry_after = limiter.get_retry_after(request)
        raise HTTPException(
            status_code=429,
            detail=f"Demasiadas solicitudes. Intenta de nuevo en {retry_after:.0f} segundos.",
            headers={"Retry-After": str(int(retry_after))},
        )

    try:
        chat_service = get_chat_service()
        response = await chat_service.chat(request_body)
        return response
    except LLMConnectionError as e:
        logger.error(f"LLM connection error: {e}")
        raise llm_connection_error(str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request_body: ChatRequest, request: Request):
    """
    Send a question and receive a streaming response.

    Returns Server-Sent Events (SSE) with response chunks.
    The final chunk includes sources and session_id.

    Rate limited: 10 requests burst, refills at 0.5/s per client.
    """
    # Rate limiting
    limiter = get_chat_rate_limiter()
    if not limiter.check(request, cost=1):
        retry_after = limiter.get_retry_after(request)
        raise HTTPException(
            status_code=429,
            detail=f"Demasiadas solicitudes. Intenta de nuevo en {retry_after:.0f} segundos.",
            headers={"Retry-After": str(int(retry_after))},
        )

    chat_service = get_chat_service()

    async def generate():
        async for chunk in chat_service.chat_stream(request_body):
            data = {
                "content": chunk.content,
                "is_final": chunk.is_final,
            }
            if chunk.is_final:
                data["session_id"] = chunk.session_id
                if chunk.sources:
                    data["sources"] = [s.model_dump() for s in chunk.sources]

            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
