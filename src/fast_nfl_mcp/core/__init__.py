"""Core components for the Fast NFL MCP server.

This package contains the main server implementation and response models.
"""

from fast_nfl_mcp.core.models import (
    ColumnSchema,
    DatasetSchema,
    ErrorResponse,
    ResponseMetadata,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.core.server import main, mcp

__all__ = [
    # Server
    "main",
    "mcp",
    # Models
    "ColumnSchema",
    "DatasetSchema",
    "ErrorResponse",
    "ResponseMetadata",
    "SuccessResponse",
    "create_error_response",
    "create_success_response",
]
