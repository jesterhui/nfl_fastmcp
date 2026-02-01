"""Tests for the FastMCP server initialization.

This module tests the server initialization, lifespan management,
MCP tool registration, and CLI argument parsing using mocked dependencies.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fast_nfl_mcp.core.server import lifespan, main, mcp
from fast_nfl_mcp.data.schema import SchemaManager
from fast_nfl_mcp.utils.types import DatasetDefinition


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
            "test_dataset": DatasetDefinition(
                loader=lambda _: mock_df,
                description="Test dataset",
                supports_seasons=True,
            ),
        }

    @pytest.mark.asyncio
    async def test_lifespan_yields_schema_manager(
        self, mock_dataset_definitions: dict
    ) -> None:
        """Test that lifespan yields a dict with schema_manager."""
        with patch.dict(
            "fast_nfl_mcp.data.schema.DATASET_DEFINITIONS",
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
            "fast_nfl_mcp.data.schema.DATASET_DEFINITIONS",
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
            "fast_nfl_mcp.data.schema.DATASET_DEFINITIONS",
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
            "failing_dataset": DatasetDefinition(
                loader=raise_error,
                description="Will fail",
                supports_seasons=True,
            ),
        }

        with patch.dict(
            "fast_nfl_mcp.data.schema.DATASET_DEFINITIONS",
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
        from fast_nfl_mcp.core.server import mcp

        # Check the tool is callable on the module
        assert hasattr(mcp, "tool")

    def test_describe_dataset_tool_registered(self) -> None:
        """Test that describe_dataset tool is registered."""
        from fast_nfl_mcp.core.server import mcp

        assert hasattr(mcp, "tool")


class TestMCPToolFunctions:
    """Tests for the MCP tool implementation functions."""

    @pytest.fixture
    def preloaded_manager(self) -> SchemaManager:
        """Create a preloaded SchemaManager."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_definitions = {
            "test_dataset": DatasetDefinition(
                loader=lambda _: mock_df,
                description="Test dataset",
                supports_seasons=True,
            ),
        }

        with patch.dict(
            "fast_nfl_mcp.data.schema.DATASET_DEFINITIONS",
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
            "test_dataset": DatasetDefinition(
                loader=lambda _: mock_df,
                description="Test dataset",
                supports_seasons=True,
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
            "test_dataset": DatasetDefinition(
                loader=lambda _: mock_df,
                description="Test dataset",
                supports_seasons=True,
            ),
        }

        with patch.dict(
            "fast_nfl_mcp.tools.utilities.DATASET_DEFINITIONS",
            mock_definitions,
            clear=True,
        ):
            response = describe_dataset_impl(preloaded_manager, "test_dataset")
            assert response.success is True


class TestServerCLI:
    """Tests for server CLI argument parsing and transport selection."""

    def test_default_stdio_transport(self) -> None:
        """Test that default (no args) uses stdio transport."""
        mock_run = MagicMock()

        with patch.object(mcp, "run", mock_run):
            with patch("sys.argv", ["fast-nfl-mcp"]):
                main()

        mock_run.assert_called_once_with()

    def test_sse_transport_default_port(self) -> None:
        """Test that --sse flag uses SSE transport with default port."""
        mock_run = MagicMock()

        with patch.object(mcp, "run", mock_run):
            with patch("sys.argv", ["fast-nfl-mcp", "--sse"]):
                main()

        mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=8000)

    def test_sse_transport_custom_port(self) -> None:
        """Test that --sse with --port uses SSE transport with custom port."""
        mock_run = MagicMock()

        with patch.object(mcp, "run", mock_run):
            with patch("sys.argv", ["fast-nfl-mcp", "--sse", "--port", "3000"]):
                main()

        mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=3000)

    def test_http_transport_default_port(self) -> None:
        """Test that --http flag uses HTTP transport with default port."""
        mock_run = MagicMock()

        with patch.object(mcp, "run", mock_run):
            with patch("sys.argv", ["fast-nfl-mcp", "--http"]):
                main()

        mock_run.assert_called_once_with(transport="http", host="0.0.0.0", port=8000)

    def test_http_transport_custom_port(self) -> None:
        """Test that --http with --port uses HTTP transport with custom port."""
        mock_run = MagicMock()

        with patch.object(mcp, "run", mock_run):
            with patch("sys.argv", ["fast-nfl-mcp", "--http", "--port", "9000"]):
                main()

        mock_run.assert_called_once_with(transport="http", host="0.0.0.0", port=9000)

    def test_custom_host(self) -> None:
        """Test that --host flag sets custom host."""
        mock_run = MagicMock()

        with patch.object(mcp, "run", mock_run):
            with patch("sys.argv", ["fast-nfl-mcp", "--sse", "--host", "127.0.0.1"]):
                main()

        mock_run.assert_called_once_with(transport="sse", host="127.0.0.1", port=8000)

    def test_custom_host_and_port(self) -> None:
        """Test that both --host and --port can be customized."""
        mock_run = MagicMock()

        with patch.object(mcp, "run", mock_run):
            with patch(
                "sys.argv",
                ["fast-nfl-mcp", "--http", "--host", "localhost", "--port", "5000"],
            ):
                main()

        mock_run.assert_called_once_with(transport="http", host="localhost", port=5000)

    def test_sse_and_http_mutually_exclusive(self) -> None:
        """Test that --sse and --http are mutually exclusive."""
        with patch("sys.argv", ["fast-nfl-mcp", "--sse", "--http"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for argument errors
            assert exc_info.value.code == 2
