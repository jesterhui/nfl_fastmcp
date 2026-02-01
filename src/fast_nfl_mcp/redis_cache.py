"""Redis caching utilities for NFL data.

This module provides Redis-based caching for expensive data operations,
specifically the player_ids dataset used by lookup_player. The cache
supports configurable TTL and graceful degradation when Redis is unavailable.
"""

import logging
import os
import pickle
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Redis connection instance (lazy initialization)
_redis_client: Any = None
_redis_connection_attempted: bool = False
_redis_available: bool = False


def _get_redis_url() -> str:
    """Get the Redis URL from environment variable or use default.

    Returns:
        The Redis URL to connect to.
    """
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def _get_redis_client() -> Any:
    """Get or create the Redis client connection.

    Returns:
        The Redis client instance, or None if connection failed.
    """
    global _redis_client, _redis_connection_attempted, _redis_available

    # Only attempt connection once
    if _redis_connection_attempted:
        return _redis_client if _redis_available else None

    _redis_connection_attempted = True

    try:
        import redis

        redis_url = _get_redis_url()
        logger.info(f"Connecting to Redis at {redis_url}")
        _redis_client = redis.from_url(
            redis_url,
            decode_responses=False,  # We need bytes for pickle
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis connection established")
        return _redis_client
    except ImportError:
        logger.warning("redis package not installed, caching disabled")
        _redis_available = False
        return None
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching disabled.")
        _redis_available = False
        return None


def is_redis_available() -> bool:
    """Check if Redis is available for caching.

    Returns:
        True if Redis is connected and available, False otherwise.
    """
    global _redis_connection_attempted, _redis_available

    if not _redis_connection_attempted:
        _get_redis_client()

    return _redis_available


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
    global _redis_client, _redis_connection_attempted, _redis_available
    _redis_client = None
    _redis_connection_attempted = False
    _redis_available = False


# Cache key constants
PLAYER_IDS_CACHE_KEY = "fast_nfl_mcp:player_ids"
