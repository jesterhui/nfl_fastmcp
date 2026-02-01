"""Play-by-play data retrieval tool.

This module provides the get_play_by_play MCP tool for retrieving
NFL play-by-play data with EPA, WPA, and detailed play outcomes.
"""

from typing import Any

from fast_nfl_mcp.core.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)
from fast_nfl_mcp.data.fetcher import NFLDataPyFetcher
from fast_nfl_mcp.utils.constants import MAX_SEASONS
from fast_nfl_mcp.utils.helpers import add_warnings_to_response
from fast_nfl_mcp.utils.validation import (
    normalize_filters,
    validate_seasons,
    validate_weeks,
)


def get_play_by_play_impl(
    seasons: list[int],
    weeks: list[int] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
    columns: list[str] | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL play-by-play data including EPA, WPA, and detailed play outcomes.

    This function retrieves play-level data for the specified seasons and weeks.
    Play-by-play data includes:
    - Play identifiers (game_id, play_id)
    - Team and player information
    - Play outcomes (yards_gained, touchdown, interception, etc.)
    - Advanced metrics (EPA, WPA, CPOE)
    - Situational data (down, distance, field position)

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 3 seasons allowed.
        weeks: Optional list of weeks to filter (1-18). All weeks if not specified.
        filters: Optional dict mapping column names to filter values.
                 Values can be single items or lists of acceptable values.
                 Example: {"home_team": "TB", "play_type": ["pass", "run"]}
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).
        columns: List of column names to include in output (required).

    Returns:
        SuccessResponse with play-by-play data (up to 100 rows by default),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(
        seasons, MAX_SEASONS, "Play-by-play data"
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

    # Build filters (normalize user filters and add week filter)
    combined_filters = normalize_filters(filters)
    if valid_weeks:
        combined_filters["week"] = valid_weeks

    # Fetch the data using generic fetch with filters and pagination
    fetcher = NFLDataPyFetcher()
    result = fetcher.fetch(
        "play_by_play",
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
