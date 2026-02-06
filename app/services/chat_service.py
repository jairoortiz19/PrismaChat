import asyncio
import hashlib
import uuid
from typing import Optional, AsyncGenerator

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from app.models.chat import ChatRequest, ChatResponse, SourceDocument, ChatStreamChunk
from app.models.session import SessionData
from app.repositories.vector_store import get_vector_store_instance
from app.services.session_service import get_session_service
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import LLMConnectionError
from app.core.cache import get_response_cache
from app.core.queue import get_inference_queue


class ChatService:
    """Service for RAG-based chat functionality"""

    def __init__(self):
        self._settings = get_settings()
        self._vector_store = get_vector_store_instance()
        self._session_service = get_session_service()
        self._logger = get_logger()
        self._llm: Optional[ChatOllama] = None
        self._response_cache = get_response_cache()
        self._queue = get_inference_queue(
            max_concurrent=self._settings.queue_max_concurrent,
            max_queue_size=self._settings.queue_max_size,
        )

    def _get_llm(self) -> ChatOllama:
        """Lazy initialization of LLM"""
        if self._llm is None:
            self._llm = ChatOllama(
                model=self._settings.llm_model,
                base_url=self._settings.ollama_base_url,
                temperature=0.7,
            )
        return self._llm

    def _get_prompt_template(self) -> ChatPromptTemplate:
        """Get the RAG prompt template"""
        system_template = """Eres un asistente útil de la empresa que responde preguntas basándose en el contexto proporcionado.

Usa SOLO la información del contexto para responder. Si la información no está en el contexto, di que no tienes esa información.

Contexto relevante:
{context}

Responde de manera clara, concisa y profesional en español."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    def _format_docs(self, docs) -> str:
        """Format retrieved documents for context"""
        return "\n\n---\n\n".join(
            f"Fuente: {doc.metadata.get('source', 'Desconocida')}\n{doc.page_content}"
            for doc in docs
        )

    def _get_chat_history(self, session: SessionData) -> list:
        """Convert session messages to LangChain format"""
        messages = []
        for msg in session.messages[-10:]:  # Last 10 messages for context
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        return messages

    def _make_response_cache_key(self, question: str, context: str) -> str:
        """Generate cache key from question + context"""
        key_str = f"response:{question}:{context}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def _run_inference(self, context: str, chat_history: list, question: str) -> str:
        """Run LLM inference (used by queue)"""
        prompt = self._get_prompt_template()
        llm = self._get_llm()
        chain = prompt | llm | StrOutputParser()

        return await chain.ainvoke(
            {
                "context": context,
                "chat_history": chat_history,
                "question": question,
            }
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request with RAG (with cache and queue)"""
        try:
            # Get or create session
            session = await self._session_service.get_or_create_session(request.session_id)
            logger = self._logger.bind(session_id=session.session_id)

            logger.info(f"Processing question: {request.question[:50]}...")

            # Retrieve relevant documents (cached in vector_store)
            docs_with_scores = await self._vector_store.similarity_search_with_score(
                request.question, k=self._settings.retriever_k
            )

            # Format context
            docs = [doc for doc, score in docs_with_scores]
            context = self._format_docs(docs)

            # Build sources
            sources = [
                SourceDocument(
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page"),
                    score=score,
                )
                for doc, score in docs_with_scores
            ]

            # Check response cache (only for questions without chat history)
            cache_key = self._make_response_cache_key(request.question, context)
            chat_history = self._get_chat_history(session)

            if not chat_history:
                cached_answer = self._response_cache.get(cache_key)
                if cached_answer is not None:
                    logger.info("Response cache HIT")
                    # Save messages to session even for cached responses
                    await self._session_service.add_message(session.session_id, "user", request.question)
                    await self._session_service.add_message(session.session_id, "assistant", cached_answer)

                    return ChatResponse(
                        answer=cached_answer,
                        session_id=session.session_id,
                        sources=sources,
                    )

            # Submit inference to queue for backpressure control
            task_id = f"chat-{uuid.uuid4().hex[:8]}"

            answer = await self._queue.submit(
                task_id=task_id,
                coroutine_factory=lambda ctx=context, hist=chat_history, q=request.question: self._run_inference(ctx, hist, q),
            )

            # Cache the response (only for fresh conversations)
            if not chat_history:
                self._response_cache.set(cache_key, answer)

            # Save messages to session
            await self._session_service.add_message(session.session_id, "user", request.question)
            await self._session_service.add_message(session.session_id, "assistant", answer)

            logger.info(f"Response generated, {len(sources)} sources used")

            return ChatResponse(
                answer=answer,
                session_id=session.session_id,
                sources=sources,
            )

        except asyncio.QueueFull:
            self._logger.warning("Inference queue full, rejecting request")
            raise LLMConnectionError(
                "El sistema está procesando demasiadas solicitudes. Intenta de nuevo en unos segundos."
            )
        except Exception as e:
            self._logger.error(f"Chat error: {e}")
            raise LLMConnectionError(f"Error processing chat: {str(e)}")

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[ChatStreamChunk, None]:
        """Stream chat response (queue managed, no response cache for streaming)"""
        try:
            # Get or create session
            session = await self._session_service.get_or_create_session(request.session_id)
            logger = self._logger.bind(session_id=session.session_id)

            logger.info(f"Processing streaming question: {request.question[:50]}...")

            # Retrieve relevant documents (cached in vector_store)
            docs_with_scores = await self._vector_store.similarity_search_with_score(
                request.question, k=self._settings.retriever_k
            )

            docs = [doc for doc, score in docs_with_scores]
            context = self._format_docs(docs)

            sources = [
                SourceDocument(
                    content=doc.page_content[:200] + "...",
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page"),
                    score=score,
                )
                for doc, score in docs_with_scores
            ]

            chat_history = self._get_chat_history(session)

            prompt = self._get_prompt_template()
            llm = self._get_llm()

            chain = prompt | llm | StrOutputParser()

            full_response = ""
            async for chunk in chain.astream(
                {
                    "context": context,
                    "chat_history": chat_history,
                    "question": request.question,
                }
            ):
                full_response += chunk
                yield ChatStreamChunk(content=chunk, is_final=False)

            # Save to session after streaming complete
            await self._session_service.add_message(session.session_id, "user", request.question)
            await self._session_service.add_message(session.session_id, "assistant", full_response)

            # Final chunk with sources
            yield ChatStreamChunk(
                content="",
                is_final=True,
                session_id=session.session_id,
                sources=sources,
            )

            logger.info(f"Streaming response completed")

        except Exception as e:
            self._logger.error(f"Stream error: {e}")
            yield ChatStreamChunk(content=f"Error: {str(e)}", is_final=True)

    async def check_llm_connection(self) -> bool:
        """Check if LLM is accessible"""
        try:
            llm = self._get_llm()
            await llm.ainvoke("test")
            return True
        except Exception as e:
            self._logger.error(f"LLM connection check failed: {e}")
            return False

    def get_cache_stats(self) -> dict:
        """Get response cache statistics"""
        return self._response_cache.get_stats()

    def get_queue_stats(self) -> dict:
        """Get inference queue statistics"""
        return self._queue.get_stats()


# Singleton instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
