"""Rosters data retrieval tool.

This module provides the get_rosters MCP tool for retrieving
NFL team roster data with filtering by seasons, weeks, and teams.
"""

from typing import Any

from fast_nfl_mcp.constants import MAX_ROSTERS_SEASONS
from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)
from fast_nfl_mcp.tools.validation import (
    normalize_filters,
    validate_seasons,
    validate_teams,
    validate_weeks,
)
from fast_nfl_mcp.utils import add_warnings_to_response


def get_rosters_impl(
    seasons: list[int],
    weeks: list[int] | None = None,
    teams: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
    columns: list[str] | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL team roster data with filtering options.

    This function retrieves roster data for the specified seasons, with
    optional filtering by weeks and teams. Roster data includes:
    - Player identification (player_id, player_name)
    - Team and position information
    - Physical attributes (height, weight)
    - Draft and college information
    - Contract and status details

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
        weeks: Optional list of weeks to filter (1-18). All weeks if not specified.
        teams: Optional list of team abbreviations to filter (e.g., ["KC", "SF"]).
               Use standard NFL team abbreviations.
        filters: Optional dict mapping column names to filter values.
                 Values can be single items or lists of acceptable values.
                 Example: {"position": "QB", "status": ["ACT", "RES"]}
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).
        columns: List of column names to include in output (required).

    Returns:
        SuccessResponse with roster data (up to 100 rows by default),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(
        seasons, MAX_ROSTERS_SEASONS, "Roster data"
    )
    if season_warning:
        warnings.append(season_warning)

    if not valid_seasons:
        return create_success_response(
            data=[],
            total_available=0,
            truncated=False,
            warning=" ".join(warnings) if warnings else "No valid seasons provided.",
        )

    # Validate weeks
    valid_weeks, week_warning = validate_weeks(weeks)
    if week_warning:
        warnings.append(week_warning)

    # Validate teams
    valid_teams, team_warning = validate_teams(teams)
    if team_warning:
        warnings.append(team_warning)

    # If teams were provided but all are invalid, return empty result
    # rather than silently returning all teams
    if teams and not valid_teams:
        return create_success_response(
            data=[],
            total_available=0,
            truncated=False,
            warning=(
                " ".join(warnings)
                if warnings
                else "No valid team abbreviations provided."
            ),
        )

    # Build filters (normalize user filters and add week/team filters)
    combined_filters = normalize_filters(filters)
    if valid_weeks:
        combined_filters["week"] = valid_weeks
    if valid_teams:
        combined_filters["team"] = valid_teams

    # Fetch the data using generic fetch with filters and pagination
    fetcher = DataFetcher()
    result = fetcher.fetch(
        "rosters",
        {"seasons": valid_seasons},
        combined_filters,
        offset=offset,
        limit=limit,
        columns=columns,
    )

    # Add any validation warnings to the result
    if warnings and isinstance(result, SuccessResponse):
        return add_warnings_to_response(result, warnings)

    return result
