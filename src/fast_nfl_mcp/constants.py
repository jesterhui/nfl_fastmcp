"""Constants for the Fast NFL MCP server.

This module centralizes configuration constants used throughout the package.
Update values here to change behavior across the codebase.
"""

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

# Valid week range for NFL regular season
MIN_WEEK: int = 1
MAX_WEEK: int = 18

# Earliest season with reliable play-by-play data
MIN_SEASON: int = 1999

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
