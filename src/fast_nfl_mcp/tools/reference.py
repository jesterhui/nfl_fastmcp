"""Reference data retrieval tools.

This module provides tools for accessing NFL reference data that doesn't
require season parameters. These datasets contain metadata about players,
teams, officials, and contracts.
"""

from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse


def get_player_ids_impl(
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get cross-platform player ID mappings.

    Returns player identifiers across multiple platforms (ESPN, Yahoo, etc.)
    to help correlate player data from different sources.

    Args:
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with player ID mappings (up to 100 rows by default),
        or ErrorResponse on failure.
    """
    fetcher = DataFetcher()
    return fetcher.fetch(
        "player_ids",
        offset=offset,
        limit=limit,
    )


def get_team_descriptions_impl(
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL team metadata and information.

    Returns team details including abbreviations, full names, divisions,
    conferences, and other organizational information for all NFL teams.

    Args:
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with team descriptions (up to 100 rows by default),
        or ErrorResponse on failure.
    """
    fetcher = DataFetcher()
    return fetcher.fetch(
        "team_descriptions",
        offset=offset,
        limit=limit,
    )


def get_officials_impl(
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL game officials data.

    Returns information about game officials including their names and
    positions (referee, umpire, line judge, etc.).

    Args:
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with officials data (up to 100 rows by default),
        or ErrorResponse on failure.
    """
    fetcher = DataFetcher()
    return fetcher.fetch(
        "officials",
        offset=offset,
        limit=limit,
    )


def get_contracts_impl(
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL player contract data.

    Returns player contract information including salary details,
    contract length, guarantees, and signing bonuses.

    Args:
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with contract data (up to 100 rows by default),
        or ErrorResponse on failure.
    """
    fetcher = DataFetcher()
    return fetcher.fetch(
        "contracts",
        offset=offset,
        limit=limit,
    )
