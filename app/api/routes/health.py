from fastapi import APIRouter

from app.services.chat_service import get_chat_service
from app.repositories.vector_store import get_vector_store_instance
from app.core.config import get_settings
from app.core.rate_limiter import get_chat_rate_limiter, get_upload_rate_limiter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Check system health, service connectivity, cache and queue stats"""
    settings = get_settings()
    chat_service = get_chat_service()
    vector_store = get_vector_store_instance()

    # Check LLM
    llm_status = "unknown"
    try:
        llm_ok = await chat_service.check_llm_connection()
        llm_status = "connected" if llm_ok else "disconnected"
    except Exception:
        llm_status = "error"

    # Check vector store
    vector_store_status = "unknown"
    try:
        stats = await vector_store.get_collection_stats()
        vector_store_status = "connected" if "error" not in stats else "error"
        document_count = stats.get("total_documents", 0)
    except Exception:
        vector_store_status = "error"
        document_count = 0

    overall_status = "healthy" if llm_status == "connected" and vector_store_status == "connected" else "degraded"

    return {
        "status": overall_status,
        "services": {
            "llm": {
                "status": llm_status,
                "model": settings.llm_model,
                "url": settings.ollama_base_url,
            },
            "vector_store": {
                "status": vector_store_status,
                "document_count": document_count,
            },
        },
        "cache": {
            "search": vector_store.get_cache_stats(),
            "response": chat_service.get_cache_stats(),
        },
        "queue": chat_service.get_queue_stats(),
        "rate_limiting": {
            "chat": get_chat_rate_limiter().get_stats(),
            "upload": get_upload_rate_limiter().get_stats(),
        },
        "config": {
            "session_backend": settings.session_backend,
            "session_ttl_hours": settings.session_ttl_hours,
            "workers": settings.workers,
        },
    }


@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe"""
    chat_service = get_chat_service()

    try:
        llm_ok = await chat_service.check_llm_connection()
        if llm_ok:
            return {"status": "ready"}
        return {"status": "not ready", "reason": "LLM not connected"}, 503
    except Exception as e:
        return {"status": "not ready", "reason": str(e)}, 503
