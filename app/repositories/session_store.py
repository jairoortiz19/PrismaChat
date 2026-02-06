from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional
import json
from pathlib import Path

from app.models.session import SessionData
from app.core.config import get_settings
from app.core.logging import get_logger


class SessionStoreBase(ABC):
    """Abstract base class for session storage"""

    @abstractmethod
    async def create(self, session: SessionData) -> SessionData:
        pass

    @abstractmethod
    async def get(self, session_id: str) -> Optional[SessionData]:
        pass

    @abstractmethod
    async def update(self, session: SessionData) -> SessionData:
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        pass

    @abstractmethod
    async def list_all(self) -> list[SessionData]:
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        pass


class InMemorySessionStore(SessionStoreBase):
    """In-memory session storage with TTL support"""

    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._settings = get_settings()
        self._logger = get_logger()

    async def create(self, session: SessionData) -> SessionData:
        self._sessions[session.session_id] = session
        self._logger.info(f"Session created: {session.session_id}")
        return session

    async def get(self, session_id: str) -> Optional[SessionData]:
        session = self._sessions.get(session_id)
        if session:
            # Check if expired
            ttl = timedelta(hours=self._settings.session_ttl_hours)
            if datetime.utcnow() - session.last_activity > ttl:
                await self.delete(session_id)
                return None
        return session

    async def update(self, session: SessionData) -> SessionData:
        session.last_activity = datetime.utcnow()
        self._sessions[session.session_id] = session
        return session

    async def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._logger.info(f"Session deleted: {session_id}")
            return True
        return False

    async def list_all(self) -> list[SessionData]:
        await self.cleanup_expired()
        return list(self._sessions.values())

    async def cleanup_expired(self) -> int:
        ttl = timedelta(hours=self._settings.session_ttl_hours)
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self._sessions.items() if now - session.last_activity > ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            self._logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)


class FileSessionStore(SessionStoreBase):
    """File-based session storage for persistence without Redis"""

    def __init__(self):
        self._settings = get_settings()
        self._logger = get_logger()
        self._session_dir = Path(self._settings.chroma_persist_dir).parent / "sessions"
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        return self._session_dir / f"{session_id}.json"

    async def create(self, session: SessionData) -> SessionData:
        path = self._get_session_path(session.session_id)
        path.write_text(session.model_dump_json())
        self._logger.info(f"Session created: {session.session_id}")
        return session

    async def get(self, session_id: str) -> Optional[SessionData]:
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            session = SessionData(**data)

            # Check if expired
            ttl = timedelta(hours=self._settings.session_ttl_hours)
            if datetime.utcnow() - session.last_activity > ttl:
                await self.delete(session_id)
                return None

            return session
        except Exception as e:
            self._logger.error(f"Error reading session {session_id}: {e}")
            return None

    async def update(self, session: SessionData) -> SessionData:
        session.last_activity = datetime.utcnow()
        path = self._get_session_path(session.session_id)
        path.write_text(session.model_dump_json())
        return session

    async def delete(self, session_id: str) -> bool:
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
            self._logger.info(f"Session deleted: {session_id}")
            return True
        return False

    async def list_all(self) -> list[SessionData]:
        sessions = []
        for path in self._session_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                session = SessionData(**data)
                sessions.append(session)
            except Exception:
                continue
        return sessions

    async def cleanup_expired(self) -> int:
        ttl = timedelta(hours=self._settings.session_ttl_hours)
        now = datetime.utcnow()
        count = 0

        for path in self._session_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                session = SessionData(**data)
                if now - session.last_activity > ttl:
                    path.unlink()
                    count += 1
            except Exception:
                continue

        if count:
            self._logger.info(f"Cleaned up {count} expired sessions")
        return count


def get_session_store() -> SessionStoreBase:
    """Factory function to get appropriate session store"""
    settings = get_settings()
    if settings.session_backend == "memory":
        return InMemorySessionStore()
    else:
        # File-based as fallback (Redis can be added later)
        return FileSessionStore()


# Singleton instance
_session_store: Optional[SessionStoreBase] = None


def get_session_store_instance() -> SessionStoreBase:
    global _session_store
    if _session_store is None:
        _session_store = get_session_store()
    return _session_store
