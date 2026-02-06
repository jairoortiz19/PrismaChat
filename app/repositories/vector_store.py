from pathlib import Path
from typing import Optional
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import VectorStoreError
from app.core.cache import get_search_cache


class VectorStoreRepository:
    """Repository for ChromaDB vector store operations"""

    def __init__(self):
        self._settings = get_settings()
        self._logger = get_logger()
        self._vectorstore: Optional[Chroma] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._search_cache = get_search_cache()

    def _get_embeddings(self) -> OllamaEmbeddings:
        """Lazy initialization of embeddings"""
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self._settings.embedding_model,
                base_url=self._settings.ollama_base_url,
            )
        return self._embeddings

    def _get_vectorstore(self) -> Chroma:
        """Lazy initialization of vector store"""
        if self._vectorstore is None:
            persist_dir = Path(self._settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._vectorstore = Chroma(
                collection_name="documents",
                embedding_function=self._get_embeddings(),
                persist_directory=str(persist_dir),
            )
            self._logger.info(f"Vector store initialized at {persist_dir}")
        return self._vectorstore

    def _make_search_key(self, query: str, k: int, filter_dict: Optional[dict]) -> str:
        """Generate cache key for search"""
        key_str = f"search:{query}:{k}:{filter_dict}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def add_documents(self, documents: list[Document], source_id: str) -> int:
        """Add documents to vector store"""
        try:
            vectorstore = self._get_vectorstore()

            # Add source_id to metadata for tracking
            for doc in documents:
                doc.metadata["source_id"] = source_id
                # Create unique ID for each chunk
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                doc.metadata["chunk_id"] = f"{source_id}_{content_hash}"

            # Add to vector store
            ids = [doc.metadata["chunk_id"] for doc in documents]
            vectorstore.add_documents(documents, ids=ids)

            # Invalidate search cache since documents changed
            self._search_cache.clear()
            self._logger.info(f"Added {len(documents)} chunks for source {source_id} (cache cleared)")

            return len(documents)
        except Exception as e:
            self._logger.error(f"Error adding documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")

    async def similarity_search(
        self, query: str, k: int = 4, filter_dict: Optional[dict] = None
    ) -> list[Document]:
        """Search for similar documents"""
        try:
            vectorstore = self._get_vectorstore()
            results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
            return results
        except Exception as e:
            self._logger.error(f"Error in similarity search: {e}")
            raise VectorStoreError(f"Search failed: {e}")

    async def similarity_search_with_score(
        self, query: str, k: int = 4, filter_dict: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores (cached)"""
        cache_key = self._make_search_key(query, k, filter_dict)

        # Check cache
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            self._logger.debug(f"Search cache HIT for query: {query[:30]}...")
            return cached

        try:
            vectorstore = self._get_vectorstore()
            results = vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)

            # Store in cache
            self._search_cache.set(cache_key, results)
            self._logger.debug(f"Search cache MISS for query: {query[:30]}... (cached)")

            return results
        except Exception as e:
            self._logger.error(f"Error in similarity search with score: {e}")
            raise VectorStoreError(f"Search failed: {e}")

    async def delete_by_source(self, source_id: str) -> bool:
        """Delete all documents from a specific source"""
        try:
            vectorstore = self._get_vectorstore()
            collection = vectorstore._collection
            results = collection.get(where={"source_id": source_id})

            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                # Invalidate cache
                self._search_cache.clear()
                self._logger.info(f"Deleted {len(results['ids'])} chunks for source {source_id} (cache cleared)")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error deleting source {source_id}: {e}")
            raise VectorStoreError(f"Delete failed: {e}")

    async def get_all_sources(self) -> list[dict]:
        """Get list of all indexed sources"""
        try:
            vectorstore = self._get_vectorstore()
            collection = vectorstore._collection

            results = collection.get(include=["metadatas"])

            sources = {}
            for metadata in results.get("metadatas", []):
                source_id = metadata.get("source_id")
                if source_id and source_id not in sources:
                    sources[source_id] = {
                        "source_id": source_id,
                        "filename": metadata.get("source", "unknown"),
                        "chunk_count": 0,
                    }
                if source_id:
                    sources[source_id]["chunk_count"] += 1

            return list(sources.values())
        except Exception as e:
            self._logger.error(f"Error getting sources: {e}")
            raise VectorStoreError(f"Failed to get sources: {e}")

    async def get_collection_stats(self) -> dict:
        """Get statistics about the vector store"""
        try:
            vectorstore = self._get_vectorstore()
            collection = vectorstore._collection
            count = collection.count()

            return {
                "total_documents": count,
                "collection_name": "documents",
                "persist_directory": self._settings.chroma_persist_dir,
            }
        except Exception as e:
            self._logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def get_retriever(self, k: int = None):
        """Get a retriever for RAG chain"""
        vectorstore = self._get_vectorstore()
        return vectorstore.as_retriever(
            search_kwargs={"k": k or self._settings.retriever_k}
        )

    def get_cache_stats(self) -> dict:
        """Get search cache statistics"""
        return self._search_cache.get_stats()


# Singleton instance
_vector_store: Optional[VectorStoreRepository] = None


def get_vector_store_instance() -> VectorStoreRepository:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreRepository()
    return _vector_store
