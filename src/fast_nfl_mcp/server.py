"""FastMCP server for NFL data access.

This module initializes the FastMCP application and provides the entry point
for running the MCP server via stdio transport.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage server lifecycle.

    This context manager handles startup and shutdown operations.
    Currently a placeholder for future schema preloading functionality.

    Args:
        server: The FastMCP server instance.

    Yields:
        A dictionary containing shared resources (currently empty).
    """
    # TODO: Preload dataset schemas on startup
    # This will be implemented in a future task to cache schema information
    # for fast retrieval during tool calls.
    yield {}


# Initialize the FastMCP application
mcp = FastMCP(
    name="fast-nfl-mcp",
    lifespan=lifespan,
)


def main() -> None:
    """Run the MCP server.

    This is the entry point for the `fast-nfl-mcp` command defined in pyproject.toml.
    It starts the server using stdio transport for MCP communication.
    """
    mcp.run()


if __name__ == "__main__":
    main()
