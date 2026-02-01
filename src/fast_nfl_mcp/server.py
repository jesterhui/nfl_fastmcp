"""FastMCP server for NFL data access.

This module initializes the FastMCP application and provides the entry point
for running the MCP server via stdio or HTTP transport.
"""

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from fast_nfl_mcp.schema_manager import SchemaManager
from fast_nfl_mcp.tools.registry import register_tools


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

# Register all MCP tools
register_tools(mcp)


def main() -> None:
    """Run the MCP server.

    This is the entry point for the `fast-nfl-mcp` command defined in pyproject.toml.
    Supports stdio (default), SSE, and HTTP transport modes.

    Usage:
        fast-nfl-mcp                    # stdio mode (default)
        fast-nfl-mcp --sse              # SSE mode on port 8000
        fast-nfl-mcp --http             # HTTP mode on port 8000
        fast-nfl-mcp --sse --port 3000  # SSE mode on custom port
    """
    parser = argparse.ArgumentParser(
        description="Fast NFL MCP Server - NFL data access via MCP protocol"
    )
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--sse",
        action="store_true",
        help="Run in SSE mode (simpler HTTP transport for persistent server)",
    )
    transport_group.add_argument(
        "--http",
        action="store_true",
        help="Run in HTTP mode (streamable HTTP transport with sessions)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP/SSE server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind server (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    if args.sse:
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.http:
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
