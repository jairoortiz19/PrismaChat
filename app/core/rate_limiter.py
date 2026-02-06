import time
from collections import defaultdict
from typing import Optional
import threading

from fastapi import Request, HTTPException, status

from app.core.logging import get_logger


class RateLimiter:
    """
    Token bucket rate limiter.

    Each client gets a bucket with max_tokens capacity that refills
    at refill_rate tokens per second.
    """

    def __init__(
        self,
        max_tokens: int = 10,
        refill_rate: float = 1.0,  # tokens per second
        window_seconds: int = 60,
    ):
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._window_seconds = window_seconds
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": max_tokens, "last_refill": time.time()}
        )
        self._lock = threading.Lock()
        self._logger = get_logger()

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Use X-Forwarded-For if behind proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _refill(self, bucket: dict) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - bucket["last_refill"]
        new_tokens = elapsed * self._refill_rate
        bucket["tokens"] = min(self._max_tokens, bucket["tokens"] + new_tokens)
        bucket["last_refill"] = now

    def check(self, request: Request, cost: int = 1) -> bool:
        """
        Check if request is allowed.

        Args:
            request: FastAPI request
            cost: Token cost for this request (default 1)

        Returns:
            True if allowed, False if rate limited
        """
        client_key = self._get_client_key(request)

        with self._lock:
            bucket = self._buckets[client_key]
            self._refill(bucket)

            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                return True

            return False

    def get_retry_after(self, request: Request, cost: int = 1) -> float:
        """Get seconds until enough tokens are available"""
        client_key = self._get_client_key(request)

        with self._lock:
            bucket = self._buckets[client_key]
            needed = cost - bucket["tokens"]
            if needed <= 0:
                return 0
            return needed / self._refill_rate

    def get_remaining(self, request: Request) -> int:
        """Get remaining tokens for a client"""
        client_key = self._get_client_key(request)

        with self._lock:
            bucket = self._buckets[client_key]
            self._refill(bucket)
            return int(bucket["tokens"])

    def cleanup(self) -> int:
        """Remove stale entries"""
        with self._lock:
            now = time.time()
            stale = [
                k for k, v in self._buckets.items()
                if now - v["last_refill"] > self._window_seconds * 10
            ]
            for key in stale:
                del self._buckets[key]
            return len(stale)

    def get_stats(self) -> dict:
        """Get rate limiter stats"""
        with self._lock:
            return {
                "active_clients": len(self._buckets),
                "max_tokens": self._max_tokens,
                "refill_rate_per_second": self._refill_rate,
            }


# --- Rate limiter instances ---

_chat_limiter: Optional[RateLimiter] = None
_upload_limiter: Optional[RateLimiter] = None
_general_limiter: Optional[RateLimiter] = None


def get_chat_rate_limiter() -> RateLimiter:
    """Rate limiter for chat endpoints (more restrictive because of LLM cost)"""
    global _chat_limiter
    if _chat_limiter is None:
        _chat_limiter = RateLimiter(
            max_tokens=10,       # 10 requests burst
            refill_rate=0.5,     # 1 request every 2 seconds
            window_seconds=60,
        )
    return _chat_limiter


def get_upload_rate_limiter() -> RateLimiter:
    """Rate limiter for document uploads"""
    global _upload_limiter
    if _upload_limiter is None:
        _upload_limiter = RateLimiter(
            max_tokens=5,        # 5 uploads burst
            refill_rate=0.1,     # 1 upload every 10 seconds
            window_seconds=60,
        )
    return _upload_limiter


def get_general_rate_limiter() -> RateLimiter:
    """Rate limiter for general endpoints"""
    global _general_limiter
    if _general_limiter is None:
        _general_limiter = RateLimiter(
            max_tokens=30,       # 30 requests burst
            refill_rate=2.0,     # 2 requests per second
            window_seconds=60,
        )
    return _general_limiter
