"""Play-by-play data retrieval tool.

This module provides the get_play_by_play MCP tool for retrieving
NFL play-by-play data with EPA, WPA, and detailed play outcomes.
"""

from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)

# Maximum number of seasons allowed per request
MAX_SEASONS = 3

# Valid week range for NFL regular season
MIN_WEEK = 1
MAX_WEEK = 18

# Earliest season with reliable play-by-play data
MIN_SEASON = 1999


def validate_seasons(seasons: list[int]) -> tuple[list[int], str | None]:
    """Validate the seasons parameter.

    Args:
        seasons: List of season years to validate.

    Returns:
        A tuple of (valid_seasons, warning_message).
        warning_message is None if all seasons are valid.
    """
    if not seasons:
        return [], "No seasons provided. Please specify at least one season."

    if len(seasons) > MAX_SEASONS:
        return (
            seasons[:MAX_SEASONS],
            f"Too many seasons requested ({len(seasons)}). "
            f"Limited to {MAX_SEASONS} seasons: {seasons[:MAX_SEASONS]}",
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
            f"Play-by-play data is available from {MIN_SEASON} onwards."
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


def get_play_by_play_impl(
    seasons: list[int],
    weeks: list[int] | None = None,
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

    Returns:
        SuccessResponse with play-by-play data (up to 100 rows),
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

    # Fetch the data
    fetcher = DataFetcher()
    result = fetcher.fetch_play_by_play(valid_seasons, valid_weeks)

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
