"""Utility modules for the Fast NFL MCP server.

This package contains constants, enums, type definitions, validation
functions, serialization utilities, and caching infrastructure.
"""

from fast_nfl_mcp.utils.cache import (
    PLAYER_IDS_CACHE_KEY,
    get_cached_dataframe,
    invalidate_cache,
    is_redis_available,
    reset_redis_connection,
    set_cached_dataframe,
)
from fast_nfl_mcp.utils.constants import (
    BDB_AVAILABLE_WEEKS,
    BDB_COMPETITION_HANDLE,
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_ROWS_TRACKING,
    DEFAULT_PLAYER_IDS_CACHE_TTL_SECONDS,
    LOOKUP_PLAYER_COLUMNS,
    LOOKUP_PLAYER_DEFAULT_LIMIT,
    LOOKUP_PLAYER_MAX_LIMIT,
    MAX_ROSTERS_SEASONS,
    MAX_ROWS_TRACKING,
    MAX_SEASONS,
    MAX_SEASONS_COMBINE,
    MAX_SEASONS_DEPTH_CHARTS,
    MAX_SEASONS_DRAFT,
    MAX_SEASONS_INJURIES,
    MAX_SEASONS_NGS,
    MAX_SEASONS_QBR,
    MAX_SEASONS_SCHEDULES,
    MAX_SEASONS_SEASONAL,
    MAX_SEASONS_SNAP_COUNTS,
    MAX_SEASONS_WEEKLY,
    MAX_WEEK,
    MIN_SEASON,
    MIN_SEASON_NGS,
    MIN_WEEK,
    VALID_TEAM_ABBREVIATIONS,
    get_current_season_year,
)
from fast_nfl_mcp.utils.enums import DatasetName, DatasetStatus
from fast_nfl_mcp.utils.helpers import add_warnings_to_response, merge_warnings
from fast_nfl_mcp.utils.serialization import (
    convert_dataframe_to_records,
    convert_value,
    extract_sample_values,
)
from fast_nfl_mcp.utils.types import DatasetDefinition
from fast_nfl_mcp.utils.validation import (
    normalize_filters,
    validate_seasons,
    validate_teams,
    validate_weeks,
)

__all__ = [
    # Cache
    "PLAYER_IDS_CACHE_KEY",
    "get_cached_dataframe",
    "invalidate_cache",
    "is_redis_available",
    "reset_redis_connection",
    "set_cached_dataframe",
    # Constants
    "BDB_AVAILABLE_WEEKS",
    "BDB_COMPETITION_HANDLE",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MAX_ROWS_TRACKING",
    "DEFAULT_PLAYER_IDS_CACHE_TTL_SECONDS",
    "LOOKUP_PLAYER_COLUMNS",
    "LOOKUP_PLAYER_DEFAULT_LIMIT",
    "LOOKUP_PLAYER_MAX_LIMIT",
    "MAX_ROSTERS_SEASONS",
    "MAX_ROWS_TRACKING",
    "MAX_SEASONS",
    "MAX_SEASONS_COMBINE",
    "MAX_SEASONS_DEPTH_CHARTS",
    "MAX_SEASONS_DRAFT",
    "MAX_SEASONS_INJURIES",
    "MAX_SEASONS_NGS",
    "MAX_SEASONS_QBR",
    "MAX_SEASONS_SCHEDULES",
    "MAX_SEASONS_SEASONAL",
    "MAX_SEASONS_SNAP_COUNTS",
    "MAX_SEASONS_WEEKLY",
    "MAX_WEEK",
    "MIN_SEASON",
    "MIN_SEASON_NGS",
    "MIN_WEEK",
    "VALID_TEAM_ABBREVIATIONS",
    "get_current_season_year",
    # Enums
    "DatasetName",
    "DatasetStatus",
    # Helpers
    "add_warnings_to_response",
    "merge_warnings",
    # Serialization
    "convert_dataframe_to_records",
    "convert_value",
    "extract_sample_values",
    # Types
    "DatasetDefinition",
    # Validation
    "normalize_filters",
    "validate_seasons",
    "validate_teams",
    "validate_weeks",
]
