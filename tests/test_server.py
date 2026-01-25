"""Tests for the FastMCP server initialization."""

import pytest

from fast_nfl_mcp.server import lifespan, main, mcp


class TestServerInitialization:
    """Tests for server initialization."""

    def test_mcp_instance_exists(self) -> None:
        """Test that the MCP instance is created."""
        assert mcp is not None

    def test_mcp_has_correct_name(self) -> None:
        """Test that the MCP server has the correct name."""
        assert mcp.name == "fast-nfl-mcp"

    def test_main_is_callable(self) -> None:
        """Test that main() entry point exists and is callable."""
        assert callable(main)

    def test_lifespan_is_async_context_manager(self) -> None:
        """Test that lifespan is an async context manager."""
        # The lifespan should be an async context manager function
        assert callable(lifespan)
        # It should have __aenter__ and __aexit__ when called
        context = lifespan(mcp)
        assert hasattr(context, "__aenter__")
        assert hasattr(context, "__aexit__")


class TestLifespan:
    """Tests for the server lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_yields_empty_dict(self) -> None:
        """Test that lifespan yields an empty dict (placeholder behavior)."""
        async with lifespan(mcp) as context:
            assert context == {}
            assert isinstance(context, dict)

    @pytest.mark.asyncio
    async def test_lifespan_completes_without_error(self) -> None:
        """Test that the lifespan context manager completes without error."""
        async with lifespan(mcp):
            pass  # Just verify it runs without error
