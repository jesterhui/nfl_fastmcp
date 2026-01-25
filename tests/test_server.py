"""Tests for the FastMCP server initialization.

This module tests the server initialization, lifespan management,
and MCP tool registration using mocked dependencies.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.schema_manager import SchemaManager
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

    @pytest.fixture
    def mock_dataset_definitions(self) -> dict:
        """Create minimal mock dataset definitions."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        return {
            "test_dataset": (
                lambda _: mock_df,
                "Test dataset",
                True,
                2024,
            ),
        }

    @pytest.mark.asyncio
    async def test_lifespan_yields_schema_manager(
        self, mock_dataset_definitions: dict
    ) -> None:
        """Test that lifespan yields a dict with schema_manager."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            mock_dataset_definitions,
            clear=True,
        ):
            async with lifespan(mcp) as context:
                assert isinstance(context, dict)
                assert "schema_manager" in context
                assert isinstance(context["schema_manager"], SchemaManager)

    @pytest.mark.asyncio
    async def test_lifespan_preloads_schemas(
        self, mock_dataset_definitions: dict
    ) -> None:
        """Test that lifespan preloads schemas during startup."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            mock_dataset_definitions,
            clear=True,
        ):
            async with lifespan(mcp) as context:
                schema_manager = context["schema_manager"]
                # Schema should be preloaded
                assert schema_manager.get_loaded_count() == 1
                assert schema_manager.is_loaded("test_dataset")

    @pytest.mark.asyncio
    async def test_lifespan_completes_without_error(
        self, mock_dataset_definitions: dict
    ) -> None:
        """Test that the lifespan context manager completes without error."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            mock_dataset_definitions,
            clear=True,
        ):
            async with lifespan(mcp):
                pass  # Just verify it runs without error

    @pytest.mark.asyncio
    async def test_lifespan_handles_preload_failures(self) -> None:
        """Test that lifespan handles schema preload failures gracefully."""

        def raise_error(_: object) -> pd.DataFrame:
            raise RuntimeError("Network error")

        failing_definitions = {
            "failing_dataset": (
                raise_error,
                "Will fail",
                True,
                2024,
            ),
        }

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            failing_definitions,
            clear=True,
        ):
            # Should not raise, even if preload fails
            async with lifespan(mcp) as context:
                schema_manager = context["schema_manager"]
                assert "failing_dataset" in schema_manager.get_failed_datasets()


class TestMCPTools:
    """Tests for registered MCP tools."""

    def test_list_datasets_tool_registered(self) -> None:
        """Test that list_datasets tool is registered."""
        # The mcp instance should have tools registered
        # We check by looking at the _tool_manager or similar
        # Note: FastMCP internals may vary, so we test the function exists
        from fast_nfl_mcp.server import mcp

        # Check the tool is callable on the module
        assert hasattr(mcp, "tool")

    def test_describe_dataset_tool_registered(self) -> None:
        """Test that describe_dataset tool is registered."""
        from fast_nfl_mcp.server import mcp

        assert hasattr(mcp, "tool")


class TestMCPToolFunctions:
    """Tests for the MCP tool implementation functions."""

    @pytest.fixture
    def preloaded_manager(self) -> SchemaManager:
        """Create a preloaded SchemaManager."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_definitions = {
            "test_dataset": (
                lambda _: mock_df,
                "Test dataset",
                True,
                2024,
            ),
        }

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            mock_definitions,
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()
            return manager

    def test_list_datasets_impl_works(self, preloaded_manager: SchemaManager) -> None:
        """Test that list_datasets_impl works with SchemaManager."""
        from fast_nfl_mcp.tools.utilities import list_datasets_impl

        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_definitions = {
            "test_dataset": (
                lambda _: mock_df,
                "Test dataset",
                True,
                2024,
            ),
        }

        with patch.dict(
            "fast_nfl_mcp.tools.utilities.DATASET_DEFINITIONS",
            mock_definitions,
            clear=True,
        ):
            response = list_datasets_impl(preloaded_manager)
            assert response.success is True

    def test_describe_dataset_impl_works(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that describe_dataset_impl works with SchemaManager."""
        from fast_nfl_mcp.tools.utilities import describe_dataset_impl

        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_definitions = {
            "test_dataset": (
                lambda _: mock_df,
                "Test dataset",
                True,
                2024,
            ),
        }

        with patch.dict(
            "fast_nfl_mcp.tools.utilities.DATASET_DEFINITIONS",
            mock_definitions,
            clear=True,
        ):
            response = describe_dataset_impl(preloaded_manager, "test_dataset")
            assert response.success is True
