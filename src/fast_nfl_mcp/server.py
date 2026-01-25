"""FastMCP server for NFL data access.

This module initializes the FastMCP application and provides the entry point
for running the MCP server via stdio transport.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.schema_manager import SchemaManager
from fast_nfl_mcp.tools.utilities import describe_dataset_impl, list_datasets_impl


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage server lifecycle.

    This context manager handles startup and shutdown operations,
    including preloading dataset schemas for fast retrieval.

    Args:
        server: The FastMCP server instance.

    Yields:
        A dictionary containing shared resources including the SchemaManager.
    """
    # Initialize and preload the schema manager
    schema_manager = SchemaManager()
    schema_manager.preload_all()

    yield {"schema_manager": schema_manager}


# Initialize the FastMCP application
mcp = FastMCP(
    name="fast-nfl-mcp",
    lifespan=lifespan,
)


@mcp.tool()
def list_datasets(ctx: Any) -> SuccessResponse:
    """List all available NFL datasets.

    Returns a list of all available datasets that can be queried,
    including their names, descriptions, and whether they support
    season-based filtering. Use this to discover what data is available.

    Returns:
        A list of datasets with name, description, supports_seasons, and status.
    """
    schema_manager: SchemaManager = ctx.request_context.lifespan_context[
        "schema_manager"
    ]
    return list_datasets_impl(schema_manager)


@mcp.tool()
def describe_dataset(ctx: Any, dataset: str) -> SuccessResponse | ErrorResponse:
    """Get detailed schema information for a specific NFL dataset.

    Returns column names, data types, sample values, and other metadata
    for the specified dataset. Use this to understand what data is available
    and how to query it before making data requests.

    Args:
        dataset: The name of the dataset to describe (e.g., "play_by_play",
                "weekly_stats", "rosters"). Use list_datasets to see all options.

    Returns:
        Schema information including columns, types, and sample values,
        or an error if the dataset is not found.
    """
    schema_manager: SchemaManager = ctx.request_context.lifespan_context[
        "schema_manager"
    ]
    return describe_dataset_impl(schema_manager, dataset)


def main() -> None:
    """Run the MCP server.

    This is the entry point for the `fast-nfl-mcp` command defined in pyproject.toml.
    It starts the server using stdio transport for MCP communication.
    """
    mcp.run()


if __name__ == "__main__":
    main()
