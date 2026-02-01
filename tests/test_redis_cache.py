"""Tests for the Redis caching utilities.

This module tests the Redis-based caching functionality for NFL data,
including connection handling, DataFrame caching, and graceful fallback.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import fast_nfl_mcp.utils.cache as redis_cache_module
from fast_nfl_mcp.utils.cache import (
    CONNECTION_RETRY_COOLDOWN_SECONDS,
    PLAYER_IDS_CACHE_KEY,
    get_cached_dataframe,
    invalidate_cache,
    is_redis_available,
    reset_redis_connection,
    set_cached_dataframe,
)


@pytest.fixture(autouse=True)
def reset_redis_state() -> None:
    """Reset Redis connection state before each test."""
    reset_redis_connection()
    yield
    reset_redis_connection()


class TestRedisConnection:
    """Tests for Redis connection handling."""

    def test_is_redis_available_false_when_connection_fails(self) -> None:
        """Test that is_redis_available returns False when connection fails."""
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection refused")
            result = is_redis_available()
            assert result is False

    def test_is_redis_available_true_when_connected(self) -> None:
        """Test that is_redis_available returns True when connected."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch("redis.from_url", return_value=mock_client):
            result = is_redis_available()
            assert result is True

    def test_connection_reused_when_successful(self) -> None:
        """Test that successful connection is reused on subsequent calls."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch("redis.from_url", return_value=mock_client) as mock_from_url:
            # Call multiple times
            is_redis_available()
            is_redis_available()
            is_redis_available()

            # Should only call from_url once (connection is reused)
            mock_from_url.assert_called_once()

    def test_connection_failure_cached_with_cooldown(self) -> None:
        """Test that failure is cached and not retried immediately."""
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection refused")

            # First call - attempts connection
            result1 = is_redis_available()
            assert result1 is False
            assert mock_from_url.call_count == 1

            # Second call within cooldown - should skip connection attempt
            result2 = is_redis_available()
            assert result2 is False
            assert mock_from_url.call_count == 1  # No additional attempt

    def test_connection_retried_after_cooldown(self) -> None:
        """Test that connection is retried after cooldown period expires."""
        # Manually manipulate the module's failure tracking state
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection refused")

            # First call - attempts connection, fails
            is_redis_available()
            assert mock_from_url.call_count == 1

            # Second call within cooldown - should skip
            # (failure was just recorded, cooldown hasn't expired)
            is_redis_available()
            assert mock_from_url.call_count == 1  # No retry yet

            # Manually expire the cooldown by setting last_failure_time to past
            redis_cache_module._last_failure_time = (
                time.monotonic() - CONNECTION_RETRY_COOLDOWN_SECONDS - 1
            )

            # Third call after cooldown expires
            is_redis_available()
            assert mock_from_url.call_count == 2  # Retry occurred

    def test_import_error_permanently_disables_redis(self) -> None:
        """Test that ImportError permanently disables Redis."""
        with patch.dict("sys.modules", {"redis": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                # First call - import fails
                result1 = is_redis_available()
                assert result1 is False

                # Reset time tracking to simulate cooldown passed
                # (but shouldn't matter for ImportError)
                monotonic_path = "fast_nfl_mcp.utils.cache.time.monotonic"
                with patch(monotonic_path, return_value=1000.0):
                    result2 = is_redis_available()
                    assert result2 is False

    def test_redis_url_from_environment(self) -> None:
        """Test that Redis URL is read from environment variable."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        custom_url = "redis://custom-host:6380"

        with (
            patch.dict(os.environ, {"REDIS_URL": custom_url}),
            patch("redis.from_url", return_value=mock_client) as mock_from_url,
        ):
            is_redis_available()

            mock_from_url.assert_called_once()
            call_args = mock_from_url.call_args
            assert call_args[0][0] == custom_url

    def test_default_redis_url(self) -> None:
        """Test that default Redis URL is used when env var not set."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        # Ensure REDIS_URL is not set
        env = {k: v for k, v in os.environ.items() if k != "REDIS_URL"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("redis.from_url", return_value=mock_client) as mock_from_url,
        ):
            is_redis_available()

            call_args = mock_from_url.call_args
            assert "localhost:6379" in call_args[0][0]

    def test_reset_allows_reconnection(self) -> None:
        """Test that reset_redis_connection allows reconnection."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch("redis.from_url", return_value=mock_client) as mock_from_url:
            # First connection
            is_redis_available()
            assert mock_from_url.call_count == 1

            # Reset and try again
            reset_redis_connection()
            is_redis_available()
            assert mock_from_url.call_count == 2

    def test_connection_lost_triggers_cooldown(self) -> None:
        """Test that connection loss triggers cooldown."""
        mock_client = MagicMock()
        # First ping succeeds, second fails (connection lost)
        mock_client.ping.side_effect = [True, Exception("Connection lost")]

        with patch("redis.from_url", return_value=mock_client) as mock_from_url:
            # First call - establishes connection
            assert is_redis_available() is True
            assert mock_from_url.call_count == 1

            # Second call - ping fails, enters cooldown
            assert is_redis_available() is False
            # No reconnection attempt due to cooldown
            assert mock_from_url.call_count == 1


class TestDataFrameCachingWithMockedRedis:
    """Tests for DataFrame caching operations using mocked Redis."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "gsis_id": ["00-0033873", "00-0036945"],
                "name": ["Patrick Mahomes", "Josh Allen"],
                "team": ["KC", "BUF"],
            }
        )

    @pytest.fixture
    def mock_redis_client(self) -> MagicMock:
        """Create a mock Redis client that stores data in a dict."""
        client = MagicMock()
        client.ping.return_value = True
        storage: dict[str, bytes] = {}

        def mock_get(key: str) -> bytes | None:
            return storage.get(key)

        def mock_setex(key: str, ttl: int, value: bytes) -> None:
            storage[key] = value

        def mock_delete(key: str) -> None:
            storage.pop(key, None)

        client.get.side_effect = mock_get
        client.setex.side_effect = mock_setex
        client.delete.side_effect = mock_delete

        return client

    def test_set_and_get_cached_dataframe(
        self, sample_df: pd.DataFrame, mock_redis_client: MagicMock
    ) -> None:
        """Test storing and retrieving a DataFrame from cache."""
        with patch("redis.from_url", return_value=mock_redis_client):
            # Set the cached data
            result = set_cached_dataframe("test_key", sample_df, 3600)
            assert result is True

            # Get it back
            retrieved_df = get_cached_dataframe("test_key")
            assert retrieved_df is not None
            # check_dtype=False: Parquet may convert StringDtype to object
            pd.testing.assert_frame_equal(retrieved_df, sample_df, check_dtype=False)

    def test_get_cached_dataframe_returns_none_on_miss(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Test that cache miss returns None."""
        with patch("redis.from_url", return_value=mock_redis_client):
            result = get_cached_dataframe("nonexistent_key")
            assert result is None

    def test_invalidate_cache_removes_key(
        self, sample_df: pd.DataFrame, mock_redis_client: MagicMock
    ) -> None:
        """Test that invalidate_cache removes the key."""
        with patch("redis.from_url", return_value=mock_redis_client):
            # Set a value
            set_cached_dataframe("test_key", sample_df, 3600)
            assert get_cached_dataframe("test_key") is not None

            # Invalidate it
            result = invalidate_cache("test_key")
            assert result is True

            # Should be gone
            assert get_cached_dataframe("test_key") is None


class TestDataFrameCachingEdgeCases:
    """Tests for edge cases in DataFrame caching."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "gsis_id": ["00-0033873"],
                "name": ["Patrick Mahomes"],
            }
        )

    def test_get_cached_dataframe_handles_corruption(self) -> None:
        """Test that corrupted cache data is handled gracefully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = b"corrupted_data"

        with patch("redis.from_url", return_value=mock_client):
            result = get_cached_dataframe("test_key")
            # Should return None on Parquet parsing error
            assert result is None

    def test_set_cached_dataframe_returns_false_when_unavailable(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test that set returns False when Redis is unavailable."""
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection failed")
            result = set_cached_dataframe("test_key", sample_df, 3600)
            assert result is False

    def test_get_cached_dataframe_returns_none_when_unavailable(self) -> None:
        """Test that get returns None when Redis is unavailable."""
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection failed")
            result = get_cached_dataframe("test_key")
            assert result is None


class TestCacheKeyConstants:
    """Tests for cache key constants."""

    def test_player_ids_cache_key(self) -> None:
        """Test that player_ids cache key is properly defined."""
        assert PLAYER_IDS_CACHE_KEY == "fast_nfl_mcp:player_ids"

    def test_cooldown_constant(self) -> None:
        """Test that cooldown constant is properly defined."""
        assert CONNECTION_RETRY_COOLDOWN_SECONDS == 30


class TestGracefulDegradation:
    """Tests for graceful degradation when Redis is unavailable."""

    def test_operations_work_without_redis_connection(self) -> None:
        """Test that operations work when Redis connection fails."""
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection failed")

            # These should not raise
            result_available = is_redis_available()
            result_get = get_cached_dataframe("test")
            result_set = set_cached_dataframe("test", pd.DataFrame(), 3600)
            result_invalidate = invalidate_cache("test")

            assert result_available is False
            assert result_get is None
            assert result_set is False
            assert result_invalidate is False

    def test_redis_error_during_get_handled(self) -> None:
        """Test that Redis errors during get are handled gracefully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.side_effect = Exception("Redis error")

        with patch("redis.from_url", return_value=mock_client):
            result = get_cached_dataframe("test_key")
            assert result is None

    def test_redis_error_during_set_handled(self) -> None:
        """Test that Redis errors during set are handled gracefully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.setex.side_effect = Exception("Redis error")

        with patch("redis.from_url", return_value=mock_client):
            result = set_cached_dataframe("test_key", pd.DataFrame(), 3600)
            assert result is False

    def test_redis_error_during_invalidate_handled(self) -> None:
        """Test that Redis errors during invalidate are handled gracefully."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.delete.side_effect = Exception("Redis error")

        with patch("redis.from_url", return_value=mock_client):
            result = invalidate_cache("test_key")
            assert result is False

    def test_no_blocking_on_repeated_failures(self) -> None:
        """Test that repeated calls don't block when Redis is down."""
        with patch("redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection refused")

            # Record start time
            start = time.time()

            # Make multiple calls
            for _ in range(10):
                is_redis_available()
                get_cached_dataframe("test")
                set_cached_dataframe("test", pd.DataFrame(), 3600)

            # Should complete quickly (1 attempt due to cooldown)
            elapsed = time.time() - start
            assert elapsed < 1.0  # Should be nearly instant

            # Only one connection attempt should have been made
            assert mock_from_url.call_count == 1
