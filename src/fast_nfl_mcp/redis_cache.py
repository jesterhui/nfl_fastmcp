"""Redis caching utilities for NFL data.

This module provides Redis-based caching for expensive data operations,
specifically the player_ids dataset used by lookup_player. The cache
supports configurable TTL and graceful degradation when Redis is unavailable.

Connection failures are cached with a cooldown period to avoid blocking
every request when Redis is known to be down.
"""

import logging
import os
import pickle
import time
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Redis connection instance (lazy initialization)
_redis_client: Any = None

# Connection failure tracking
_last_failure_time: float = 0.0
_redis_permanently_unavailable: bool = False

# Cooldown period before retrying after a connection failure (seconds)
CONNECTION_RETRY_COOLDOWN_SECONDS: int = 30

# Cache key constants
PLAYER_IDS_CACHE_KEY = "fast_nfl_mcp:player_ids"


def _get_redis_url() -> str:
    """Get the Redis URL from environment variable or use default.

    Returns:
        The Redis URL to connect to.
    """
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def _should_skip_connection_attempt() -> bool:
    """Check if we should skip attempting to connect to Redis.

    Returns True if:
    - Redis is permanently unavailable (e.g., package not installed)
    - We're within the cooldown period after a recent failure

    Returns:
        True if connection attempt should be skipped, False otherwise.
    """
    global _last_failure_time, _redis_permanently_unavailable

    if _redis_permanently_unavailable:
        return True

    if _last_failure_time > 0:
        elapsed = time.monotonic() - _last_failure_time
        if elapsed < CONNECTION_RETRY_COOLDOWN_SECONDS:
            return True

    return False


def _record_connection_failure(permanent: bool = False) -> None:
    """Record a connection failure for cooldown tracking.

    Args:
        permanent: If True, mark Redis as permanently unavailable
                   (e.g., package not installed).
    """
    global _last_failure_time, _redis_permanently_unavailable

    if permanent:
        _redis_permanently_unavailable = True
    else:
        _last_failure_time = time.monotonic()


def _try_connect() -> Any:
    """Attempt to connect to Redis once.

    Returns:
        The connected Redis client, or None if connection failed.
    """
    try:
        import redis

        redis_url = _get_redis_url()
        logger.info(f"Connecting to Redis at {redis_url}")
        client = redis.from_url(
            redis_url,
            decode_responses=False,  # We need bytes for pickle
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection
        client.ping()
        logger.info("Redis connection established")
        return client
    except ImportError:
        logger.warning("redis package not installed, caching permanently disabled")
        _record_connection_failure(permanent=True)
        return None
    except Exception as e:
        logger.warning(
            f"Redis connection failed: {e}. "
            f"Will retry in {CONNECTION_RETRY_COOLDOWN_SECONDS}s."
        )
        _record_connection_failure(permanent=False)
        return None


def _get_redis_client() -> Any:
    """Get or create the Redis client connection.

    If Redis is known to be unavailable (recent failure or package missing),
    returns None immediately without blocking.

    Returns:
        The Redis client instance, or None if connection failed/unavailable.
    """
    global _redis_client

    # If already connected, verify connection is still alive
    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            logger.warning("Redis connection lost, will retry on next request")
            _redis_client = None
            _record_connection_failure(permanent=False)
            return None

    # Skip connection attempt if within cooldown or permanently unavailable
    if _should_skip_connection_attempt():
        return None

    # Attempt to connect (single attempt, no retries)
    _redis_client = _try_connect()
    return _redis_client


def is_redis_available() -> bool:
    """Check if Redis is available for caching.

    Returns:
        True if Redis is connected and available, False otherwise.
    """
    return _get_redis_client() is not None


def get_cached_dataframe(key: str) -> pd.DataFrame | None:
    """Retrieve a cached DataFrame from Redis.

    Args:
        key: The cache key to retrieve.

    Returns:
        The cached DataFrame if found and valid, None otherwise.
    """
    client = _get_redis_client()
    if client is None:
        return None

    try:
        data = client.get(key)
        if data is None:
            logger.debug(f"Cache miss for key: {key}")
            return None

        df = pickle.loads(data)
        if isinstance(df, pd.DataFrame):
            logger.debug(f"Cache hit for key: {key} ({len(df)} rows)")
            return df
        else:
            logger.warning(f"Cached data for key {key} is not a DataFrame")
            return None
    except Exception as e:
        logger.warning(f"Error retrieving from cache: {e}")
        return None


def set_cached_dataframe(key: str, df: pd.DataFrame, ttl_seconds: int) -> bool:
    """Store a DataFrame in the Redis cache.

    Args:
        key: The cache key to store under.
        df: The DataFrame to cache.
        ttl_seconds: Time-to-live in seconds for the cache entry.

    Returns:
        True if caching succeeded, False otherwise.
    """
    client = _get_redis_client()
    if client is None:
        return False

    try:
        data = pickle.dumps(df)
        client.setex(key, ttl_seconds, data)
        logger.debug(f"Cached DataFrame under key: {key} (TTL: {ttl_seconds}s)")
        return True
    except Exception as e:
        logger.warning(f"Error storing in cache: {e}")
        return False


def invalidate_cache(key: str) -> bool:
    """Remove a key from the cache.

    Args:
        key: The cache key to remove.

    Returns:
        True if the key was removed, False otherwise.
    """
    client = _get_redis_client()
    if client is None:
        return False

    try:
        client.delete(key)
        logger.debug(f"Invalidated cache key: {key}")
        return True
    except Exception as e:
        logger.warning(f"Error invalidating cache: {e}")
        return False


def reset_redis_connection() -> None:
    """Reset the Redis connection state for testing purposes.

    This allows re-attempting a connection after a previous failure.
    """
    global _redis_client, _last_failure_time, _redis_permanently_unavailable
    _redis_client = None
    _last_failure_time = 0.0
    _redis_permanently_unavailable = False
