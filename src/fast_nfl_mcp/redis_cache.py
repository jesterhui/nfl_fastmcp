"""Redis caching utilities for NFL data.

This module provides Redis-based caching for expensive data operations,
specifically the player_ids dataset used by lookup_player. The cache
supports configurable TTL and automatic retry with backoff when Redis
is temporarily unavailable.
"""

import logging
import os
import pickle
from typing import Any

import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Redis connection instance (lazy initialization)
_redis_client: Any = None

# Cache key constants
PLAYER_IDS_CACHE_KEY = "fast_nfl_mcp:player_ids"


def _get_redis_url() -> str:
    """Get the Redis URL from environment variable or use default.

    Returns:
        The Redis URL to connect to.
    """
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


class RedisConnectionError(Exception):
    """Raised when Redis connection fails."""

    pass


@retry(
    retry=retry_if_exception_type(RedisConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _connect_to_redis() -> Any:
    """Attempt to connect to Redis with retry logic.

    Uses tenacity to retry connection attempts with exponential backoff.
    Retries up to 3 times with waits of 2s, 4s, 8s between attempts.

    Returns:
        The connected Redis client.

    Raises:
        RedisConnectionError: If connection fails after all retries.
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
        logger.warning("redis package not installed, caching disabled")
        raise RedisConnectionError("redis package not installed") from None
    except Exception as e:
        logger.warning(f"Redis connection attempt failed: {e}")
        raise RedisConnectionError(str(e)) from e


def _get_redis_client() -> Any:
    """Get or create the Redis client connection.

    Returns:
        The Redis client instance, or None if connection failed.
    """
    global _redis_client

    # If already connected, verify connection is still alive
    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            logger.warning("Redis connection lost, attempting to reconnect")
            _redis_client = None

    # Attempt to connect with retry
    try:
        _redis_client = _connect_to_redis()
        return _redis_client
    except RedisConnectionError as e:
        logger.warning(f"Redis connection failed after retries: {e}")
        return None


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
    global _redis_client
    _redis_client = None
