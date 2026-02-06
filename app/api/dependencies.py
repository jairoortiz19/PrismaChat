from fastapi import Request
from app.core.logging import get_logger


async def get_request_logger(request: Request):
    """Get a logger bound to the current request"""
    session_id = request.headers.get("X-Session-ID", "no-session")
    return get_logger(session_id)
