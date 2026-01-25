"""MCP tool definitions for NFL data access.

This package contains all MCP tools exposed by the Fast NFL MCP server.
Tools are organized by category:
- utilities: Dataset discovery and schema inspection tools
"""

from fast_nfl_mcp.tools.utilities import describe_dataset_impl, list_datasets_impl

__all__ = [
    "list_datasets_impl",
    "describe_dataset_impl",
]
