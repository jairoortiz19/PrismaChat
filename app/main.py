from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import ChatbotException
from app.core.queue import get_inference_queue

from app.api.routes import health, chat, documents, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    setup_logging()
    logger = get_logger()
    logger.info("Starting Chatbot RAG API...")

    settings = get_settings()
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Session Backend: {settings.session_backend}")
    logger.info(f"Queue: {settings.queue_max_concurrent} workers, max {settings.queue_max_size} queued")
    logger.info(f"Cache: search TTL={settings.cache_search_ttl}s, response TTL={settings.cache_response_ttl}s")

    # Start inference queue
    queue = get_inference_queue(
        max_concurrent=settings.queue_max_concurrent,
        max_queue_size=settings.queue_max_size,
    )
    await queue.start()
    logger.info("Inference queue started")

    yield

    # Shutdown
    logger.info("Shutting down Chatbot RAG API...")
    await queue.stop()
    logger.info("Inference queue stopped")


def create_app() -> FastAPI:
    """Application factory"""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
## Chatbot RAG API

API REST para chatbot con RAG (Retrieval Augmented Generation) que permite:

- **Chat**: Hacer preguntas sobre documentos de la empresa
- **Documentos**: Subir y gestionar documentos para indexar
- **Sesiones**: Mantener contexto de conversación

### Flujo de uso típico:

1. Subir documentos: `POST /api/v1/documents/upload`
2. Ingestar documentos: `POST /api/v1/documents/ingest`
3. Crear sesión: `POST /api/v1/sessions`
4. Hacer preguntas: `POST /api/v1/chat` con el session_id
        """,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        session_id = request.headers.get("X-Session-ID", "no-session")
        logger = get_logger(session_id)

        logger.info(f"Request: {request.method} {request.url.path}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )

        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Exception handlers
    @app.exception_handler(ChatbotException)
    async def chatbot_exception_handler(request: Request, exc: ChatbotException):
        logger = get_logger()
        logger.error(f"ChatbotException: {exc.message}")
        return JSONResponse(
            status_code=500,
            content={"detail": exc.message, "type": exc.__class__.__name__},
        )

    # Register routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(sessions.router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.workers,
        reload=True,
    )
