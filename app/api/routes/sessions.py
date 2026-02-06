from fastapi import APIRouter, HTTPException

from app.models.session import SessionCreate, SessionResponse, SessionData
from app.services.session_service import get_session_service
from app.core.exceptions import SessionNotFoundError, session_not_found_error
from app.core.logging import get_logger

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])
logger = get_logger()


@router.post("", response_model=SessionResponse)
async def create_session(request: SessionCreate = None):
    """
    Create a new chat session.

    Sessions maintain conversation history and context.
    Returns a session_id to use in subsequent chat requests.
    """
    session_service = get_session_service()
    session = await session_service.create_session(request)
    return session


@router.get("", response_model=list[SessionResponse])
async def list_sessions():
    """
    List all active sessions.
    """
    session_service = get_session_service()
    sessions = await session_service.list_sessions()
    return sessions


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get session information by ID.
    """
    try:
        session_service = get_session_service()
        session = await session_service.get_session_response(session_id)
        return session
    except SessionNotFoundError:
        raise session_not_found_error(session_id)


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str):
    """
    Get all messages from a session.
    """
    try:
        session_service = get_session_service()
        session = await session_service.get_session(session_id)
        return {
            "session_id": session.session_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in session.messages
            ],
            "total_messages": len(session.messages),
        }
    except SessionNotFoundError:
        raise session_not_found_error(session_id)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its conversation history.
    """
    session_service = get_session_service()
    success = await session_service.delete_session(session_id)

    if not success:
        raise session_not_found_error(session_id)

    return {
        "success": True,
        "message": f"Session '{session_id}' deleted successfully",
    }


@router.post("/cleanup")
async def cleanup_expired_sessions():
    """
    Clean up expired sessions.

    Removes sessions that have exceeded the configured TTL.
    """
    session_service = get_session_service()
    count = await session_service.cleanup_expired()

    return {
        "success": True,
        "sessions_cleaned": count,
    }
