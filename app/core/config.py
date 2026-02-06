from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    # LLM Configuration
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="llama3.1:8b")
    embedding_model: str = Field(default="nomic-embed-text")

    # Storage Configuration
    chroma_persist_dir: str = Field(default="./data/vectorstore")
    documents_dir: str = Field(default="./documents")

    # Session Configuration
    session_ttl_hours: int = Field(default=24)
    session_backend: Literal["memory", "redis"] = Field(default="memory")
    redis_url: str = Field(default="redis://localhost:6379")

    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_title: str = Field(default="Chatbot RAG API")
    api_version: str = Field(default="1.0.0")

    # RAG Configuration
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    retriever_k: int = Field(default=4)

    # Cache Configuration
    cache_search_ttl: int = Field(default=1800, description="Search cache TTL in seconds")
    cache_search_max_size: int = Field(default=512)
    cache_response_ttl: int = Field(default=3600, description="Response cache TTL in seconds")
    cache_response_max_size: int = Field(default=256)

    # Queue Configuration
    queue_max_concurrent: int = Field(default=2, description="Max concurrent LLM inferences")
    queue_max_size: int = Field(default=50, description="Max queue size before rejecting")

    # Rate Limiting
    rate_limit_chat_tokens: int = Field(default=10, description="Max chat burst requests")
    rate_limit_chat_refill: float = Field(default=0.5, description="Chat tokens refill per second")
    rate_limit_upload_tokens: int = Field(default=5)
    rate_limit_upload_refill: float = Field(default=0.1)

    # Workers
    workers: int = Field(default=1, description="Number of Uvicorn workers")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
