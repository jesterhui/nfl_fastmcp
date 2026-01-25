"""Constants for the Fast NFL MCP server.

This module centralizes configuration constants used throughout the package.
Update values here to change behavior across the codebase.
"""

# Maximum number of rows to return by default in data fetches
DEFAULT_MAX_ROWS: int = 100

# Maximum number of seasons allowed per request for play-by-play data
MAX_SEASONS: int = 3

# Valid week range for NFL regular season
MIN_WEEK: int = 1
MAX_WEEK: int = 18

# Earliest season with reliable play-by-play data
MIN_SEASON: int = 1999
