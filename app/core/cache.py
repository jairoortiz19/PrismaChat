import hashlib
import time
from typing import Any, Optional
from collections import OrderedDict
import threading

from app.core.logging import get_logger


class TTLCache:
    """Thread-safe in-memory cache with TTL (Time To Live) and LRU eviction"""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._logger = get_logger()

    def _make_key(self, *args, **kwargs) -> str:
        """Generate a hash key from arguments"""
        key_str = f"{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, returns None if expired or not found"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry["timestamp"] > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = {
                "value": value,
                "timestamp": time.time(),
            }

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove all expired entries"""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._cache.items()
                if now - v["timestamp"] > self._ttl_seconds
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
            }


# --- Singleton cache instances ---

_search_cache: Optional[TTLCache] = None
_response_cache: Optional[TTLCache] = None


def get_search_cache() -> TTLCache:
    """Cache for similarity search results (shorter TTL, invalidated on ingest)"""
    global _search_cache
    if _search_cache is None:
        _search_cache = TTLCache(max_size=512, ttl_seconds=1800)  # 30 min
    return _search_cache


def get_response_cache() -> TTLCache:
    """Cache for full LLM responses (longer TTL for identical questions)"""
    global _response_cache
    if _response_cache is None:
        _response_cache = TTLCache(max_size=256, ttl_seconds=3600)  # 1 hour
    return _response_cache
