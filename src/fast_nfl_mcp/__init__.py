"""Fast NFL MCP Server.

An MCP server that exposes NFL data from nfl_data_py to AI assistants
like Claude Code for data exploration during model building.

Package Structure:
- core/: Server implementation and response models
- data/: Data fetching and schema management
- utils/: Constants, validation, serialization, and caching utilities
- tools/: MCP tool implementations organized by data type
"""

__version__ = "0.1.0"

# Only expose the main entry point at the top level
from fast_nfl_mcp.core.server import main

__all__ = [
    "__version__",
    "main",
]
