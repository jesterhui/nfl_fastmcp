"""Constants for the Fast NFL MCP server.

This module centralizes configuration constants used throughout the package.
Update values here to change behavior across the codebase.
"""

from datetime import date

# Maximum number of rows to return by default in data fetches
DEFAULT_MAX_ROWS: int = 100

# Maximum number of seasons allowed per request for play-by-play data
MAX_SEASONS: int = 3

# Maximum number of seasons allowed per request for rosters data
MAX_ROSTERS_SEASONS: int = 5

# Maximum seasons for player stats tools
MAX_SEASONS_WEEKLY: int = 5
MAX_SEASONS_SEASONAL: int = 10

# Maximum seasons for Next Gen Stats tools
MAX_SEASONS_NGS: int = 5

# Maximum seasons for schedules and draft picks
MAX_SEASONS_SCHEDULES: int = 10
MAX_SEASONS_DRAFT: int = 20

# Maximum seasons for misc tools
MAX_SEASONS_SNAP_COUNTS: int = 5
MAX_SEASONS_INJURIES: int = 5
MAX_SEASONS_DEPTH_CHARTS: int = 5
MAX_SEASONS_COMBINE: int = 10
MAX_SEASONS_QBR: int = 5

# Earliest season with Next Gen Stats data (NGS tracking started in 2016)
MIN_SEASON_NGS: int = 2016

# Valid week range for NFL regular season
MIN_WEEK: int = 1
MAX_WEEK: int = 18

# Earliest season with reliable play-by-play data
MIN_SEASON: int = 1999

# Big Data Bowl (BDB) competition settings
BDB_COMPETITION_HANDLE: str = "nfl-big-data-bowl-2026-analytics"
BDB_AVAILABLE_WEEKS: tuple[int, ...] = tuple(range(1, 19))  # Weeks 1-18
DEFAULT_MAX_ROWS_TRACKING: int = 50
MAX_ROWS_TRACKING: int = 100

# Lookup player tool settings (from tools/reference.py)
LOOKUP_PLAYER_DEFAULT_LIMIT: int = 10
LOOKUP_PLAYER_MAX_LIMIT: int = 100

# Cache TTL for player_ids dataset used by lookup_player (in seconds)
# Default: 1 hour (3600 seconds)
# Can be overridden via PLAYER_IDS_CACHE_TTL_SECONDS environment variable
DEFAULT_PLAYER_IDS_CACHE_TTL_SECONDS: int = 3600


def get_player_ids_cache_ttl() -> int:
    """Get the cache TTL for player_ids dataset.

    Reads from PLAYER_IDS_CACHE_TTL_SECONDS environment variable if set,
    otherwise uses the default value of 3600 seconds (1 hour).

    Returns:
        The cache TTL in seconds.
    """
    import os

    env_value = os.environ.get("PLAYER_IDS_CACHE_TTL_SECONDS")
    if env_value is not None:
        try:
            ttl = int(env_value)
            if ttl > 0:
                return ttl
        except ValueError:
            pass
    return DEFAULT_PLAYER_IDS_CACHE_TTL_SECONDS


# Columns returned by lookup_player
LOOKUP_PLAYER_COLUMNS: list[str] = [
    "gsis_id",
    "name",
    "team",
    "position",
    "merge_name",
]


def get_current_season_year() -> int:
    """Calculate the current NFL season year based on today's date.

    The NFL season typically starts in early September and ends in February.
    This function determines which season year should be considered "current":
    - January through August: Returns the previous calendar year (e.g., Feb 2026 -> 2025)
    - September through December: Returns the current calendar year (e.g., Oct 2026 -> 2026)

    Returns:
        The current NFL season year as an integer.

    Examples:
        >>> # If today is February 15, 2026
        >>> get_current_season_year()  # Returns 2025
        >>> # If today is October 15, 2026
        >>> get_current_season_year()  # Returns 2026
    """
    today = date.today()
    # NFL season starts in September (month 9)
    # If we're in January-August, we're still in the previous season
    if today.month < 9:
        return today.year - 1
    return today.year


# Valid NFL team abbreviations (current 32 teams plus historical)
VALID_TEAM_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # Current 32 teams
        "ARI",  # Arizona Cardinals
        "ATL",  # Atlanta Falcons
        "BAL",  # Baltimore Ravens
        "BUF",  # Buffalo Bills
        "CAR",  # Carolina Panthers
        "CHI",  # Chicago Bears
        "CIN",  # Cincinnati Bengals
        "CLE",  # Cleveland Browns
        "DAL",  # Dallas Cowboys
        "DEN",  # Denver Broncos
        "DET",  # Detroit Lions
        "GB",  # Green Bay Packers
        "HOU",  # Houston Texans
        "IND",  # Indianapolis Colts
        "JAX",  # Jacksonville Jaguars
        "KC",  # Kansas City Chiefs
        "LA",  # Los Angeles Rams
        "LAC",  # Los Angeles Chargers
        "LV",  # Las Vegas Raiders
        "MIA",  # Miami Dolphins
        "MIN",  # Minnesota Vikings
        "NE",  # New England Patriots
        "NO",  # New Orleans Saints
        "NYG",  # New York Giants
        "NYJ",  # New York Jets
        "PHI",  # Philadelphia Eagles
        "PIT",  # Pittsburgh Steelers
        "SEA",  # Seattle Seahawks
        "SF",  # San Francisco 49ers
        "TB",  # Tampa Bay Buccaneers
        "TEN",  # Tennessee Titans
        "WAS",  # Washington Commanders
        # Historical abbreviations that may appear in older data
        "OAK",  # Oakland Raiders (now LV)
        "SD",  # San Diego Chargers (now LAC)
        "STL",  # St. Louis Rams (now LA)
    }
)
