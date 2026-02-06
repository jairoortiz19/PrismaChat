from typing import Optional

from app.models.session import SessionData, SessionCreate, SessionResponse
from app.repositories.session_store import get_session_store_instance
from app.core.logging import get_logger
from app.core.exceptions import SessionNotFoundError


class SessionService:
    """Service for managing chat sessions"""

    def __init__(self):
        self._store = get_session_store_instance()
        self._logger = get_logger()

    async def create_session(self, request: Optional[SessionCreate] = None) -> SessionResponse:
        """Create a new session"""
        session = SessionData(metadata=request.metadata if request else {})
        await self._store.create(session)

        self._logger.bind(session_id=session.session_id).info("New session created")

        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            last_activity=session.last_activity,
            message_count=len(session.messages),
            metadata=session.metadata,
        )

    async def get_session(self, session_id: str) -> SessionData:
        """Get session by ID"""
        session = await self._store.get(session_id)
        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return session

    async def get_or_create_session(self, session_id: Optional[str] = None) -> SessionData:
        """Get existing session or create new one"""
        if session_id:
            session = await self._store.get(session_id)
            if session:
                return session

        # Create new session
        session = SessionData()
        await self._store.create(session)
        self._logger.bind(session_id=session.session_id).info("New session created (auto)")
        return session

    async def add_message(
        self, session_id: str, role: str, content: str
    ) -> SessionData:
        """Add a message to session"""
        session = await self.get_session(session_id)
        session.add_message(role, content)
        await self._store.update(session)
        return session

    async def get_session_response(self, session_id: str) -> SessionResponse:
        """Get session as response model"""
        session = await self.get_session(session_id)
        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            last_activity=session.last_activity,
            message_count=len(session.messages),
            metadata=session.metadata,
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        result = await self._store.delete(session_id)
        if result:
            self._logger.bind(session_id=session_id).info("Session deleted")
        return result

    async def list_sessions(self) -> list[SessionResponse]:
        """List all active sessions"""
        sessions = await self._store.list_all()
        return [
            SessionResponse(
                session_id=s.session_id,
                created_at=s.created_at,
                last_activity=s.last_activity,
                message_count=len(s.messages),
                metadata=s.metadata,
            )
            for s in sessions
        ]

    async def cleanup_expired(self) -> int:
        """Clean up expired sessions"""
        count = await self._store.cleanup_expired()
        if count:
            self._logger.info(f"Cleaned up {count} expired sessions")
        return count


# Singleton instance
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
