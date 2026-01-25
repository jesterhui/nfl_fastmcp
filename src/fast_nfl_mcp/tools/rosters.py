"""Rosters data retrieval tool.

This module provides the get_rosters MCP tool for retrieving
NFL team roster data with filtering by seasons, weeks, and teams.
"""

from typing import Any

from fast_nfl_mcp.constants import (
    MAX_ROSTERS_SEASONS,
    MAX_WEEK,
    MIN_SEASON,
    MIN_WEEK,
    VALID_TEAM_ABBREVIATIONS,
)
from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)


def validate_seasons(seasons: list[int]) -> tuple[list[int], str | None]:
    """Validate the seasons parameter for rosters.

    Args:
        seasons: List of season years to validate.

    Returns:
        A tuple of (valid_seasons, warning_message).
        warning_message is None if all seasons are valid.
    """
    if not seasons:
        return [], "No seasons provided. Please specify at least one season."

    if len(seasons) > MAX_ROSTERS_SEASONS:
        return (
            seasons[:MAX_ROSTERS_SEASONS],
            f"Too many seasons requested ({len(seasons)}). "
            f"Limited to {MAX_ROSTERS_SEASONS} seasons: {seasons[:MAX_ROSTERS_SEASONS]}",
        )

    # Filter out invalid seasons
    valid_seasons = []
    invalid_seasons = []
    for season in seasons:
        if season >= MIN_SEASON:
            valid_seasons.append(season)
        else:
            invalid_seasons.append(season)

    warning = None
    if invalid_seasons:
        warning = (
            f"Invalid seasons removed: {invalid_seasons}. "
            f"Roster data is available from {MIN_SEASON} onwards."
        )

    return valid_seasons, warning


def validate_weeks(weeks: list[int] | None) -> tuple[list[int] | None, str | None]:
    """Validate the weeks parameter.

    Args:
        weeks: Optional list of week numbers to validate.

    Returns:
        A tuple of (valid_weeks, warning_message).
        warning_message is None if all weeks are valid.
    """
    if weeks is None:
        return None, None

    if not weeks:
        return None, None

    # Filter out invalid weeks
    valid_weeks = []
    invalid_weeks = []
    for week in weeks:
        if MIN_WEEK <= week <= MAX_WEEK:
            valid_weeks.append(week)
        else:
            invalid_weeks.append(week)

    warning = None
    if invalid_weeks:
        warning = (
            f"Invalid weeks removed: {invalid_weeks}. "
            f"Valid week range is {MIN_WEEK}-{MAX_WEEK}."
        )

    return valid_weeks if valid_weeks else None, warning


def validate_teams(teams: list[str] | None) -> tuple[list[str] | None, str | None]:
    """Validate team abbreviations.

    Args:
        teams: Optional list of team abbreviations to validate.

    Returns:
        A tuple of (valid_teams, warning_message).
        warning_message is None if all teams are valid.
    """
    if teams is None:
        return None, None

    if not teams:
        return None, None

    # Normalize to uppercase and validate
    valid_teams = []
    invalid_teams = []
    for team in teams:
        team_upper = team.upper()
        if team_upper in VALID_TEAM_ABBREVIATIONS:
            valid_teams.append(team_upper)
        else:
            invalid_teams.append(team)

    warning = None
    if invalid_teams:
        warning = (
            f"Invalid team abbreviations removed: {invalid_teams}. "
            f"Use standard NFL team abbreviations (e.g., KC, SF, TB)."
        )

    return valid_teams if valid_teams else None, warning


def normalize_filters(
    filters: dict[str, Any] | None,
) -> dict[str, list[Any]]:
    """Normalize filter values to lists.

    Args:
        filters: Dict mapping column names to filter values.
                 Values can be single items or lists.

    Returns:
        Dict with all values normalized to lists.
    """
    if filters is None:
        return {}

    normalized: dict[str, list[Any]] = {}
    for column, value in filters.items():
        if isinstance(value, list):
            normalized[column] = value
        else:
            normalized[column] = [value]
    return normalized


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
    valid_seasons, season_warning = validate_seasons(seasons)
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
        existing_warning = result.warning or ""
        combined_warning = " ".join(filter(None, [existing_warning, *warnings]))
        # Create new response with combined warning
        return create_success_response(
            data=result.data,
            total_available=result.metadata.total_available,
            truncated=result.metadata.truncated,
            columns=result.metadata.columns,
            warning=combined_warning if combined_warning else None,
        )

    return result
