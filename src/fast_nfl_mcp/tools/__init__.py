"""MCP tool definitions for NFL data access.

This package contains all MCP tools exposed by the Fast NFL MCP server.
Tools are organized by category:

- registry: Centralized tool registration (register_tools function)
- utilities: Dataset discovery and schema inspection tools
- play_by_play: Play-by-play data retrieval
- player_stats: Weekly and seasonal player statistics
- rosters: Team roster data
- schedules: Game schedules and results
- draft: Historical draft picks
- reference: Player IDs, team descriptions, officials, contracts
- bdb: Big Data Bowl tracking data
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
