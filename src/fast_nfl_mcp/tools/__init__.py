"""MCP tool definitions for NFL data access.

This package contains all MCP tools exposed by the Fast NFL MCP server.
Tools are organized by category:
- utilities: Dataset discovery and schema inspection tools
- play_by_play: Play-by-play data retrieval tool
- player_stats: Weekly and seasonal player statistics tools
- rosters: Team roster data tools
- reference: Reference data (player IDs, teams, officials, contracts)
- draft: Draft picks data tool
- schedules: Game schedule data tool
- bdb: Big Data Bowl tracking data tools
- registry: Tool registration with the MCP server
"""

from fast_nfl_mcp.tools.play_by_play import get_play_by_play_impl
from fast_nfl_mcp.tools.registry import register_tools
from fast_nfl_mcp.tools.utilities import describe_dataset_impl, list_datasets_impl

__all__ = [
    "describe_dataset_impl",
    "get_play_by_play_impl",
    "list_datasets_impl",
    "register_tools",
]
