"""Game schedules data retrieval tool.

This module provides the MCP tool for retrieving NFL game schedules
including game results, dates, and team matchups.
"""

from typing import Any

from fast_nfl_mcp.constants import MAX_SEASONS_SCHEDULES, MIN_SEASON
from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)


def validate_seasons(
    seasons: list[int], max_seasons: int
) -> tuple[list[int], str | None]:
    """Validate the seasons parameter with a specified limit.

    Args:
        seasons: List of season years to validate.
        max_seasons: Maximum number of seasons allowed.

    Returns:
        A tuple of (valid_seasons, warning_message).
        warning_message is None if all seasons are valid.
    """
    if not seasons:
        return [], "No seasons provided. Please specify at least one season."

    warnings: list[str] = []

    # First filter out invalid seasons (before applying max limit)
    valid_seasons = []
    invalid_seasons = []
    for season in seasons:
        if season >= MIN_SEASON:
            valid_seasons.append(season)
        else:
            invalid_seasons.append(season)

    if invalid_seasons:
        warnings.append(
            f"Invalid seasons removed: {invalid_seasons}. "
            f"Data is available from {MIN_SEASON} onwards."
        )

    # Then apply max_seasons limit to valid seasons only
    if len(valid_seasons) > max_seasons:
        warnings.append(
            f"Too many seasons requested ({len(valid_seasons)} valid). "
            f"Limited to {max_seasons} seasons: {valid_seasons[:max_seasons]}"
        )
        valid_seasons = valid_seasons[:max_seasons]

    warning = " ".join(warnings) if warnings else None
    return valid_seasons, warning


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


def get_schedules_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL game schedules and results.

    This function retrieves game schedule information for the specified seasons,
    including matchups, dates, scores, and game outcomes.

    Args:
        seasons: List of seasons (e.g., [2020, 2021, 2022]). Maximum 10 seasons allowed.
        columns: List of column names to include in output (optional).
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with schedules data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(seasons, MAX_SEASONS_SCHEDULES)
    if season_warning:
        warnings.append(season_warning)

    if not valid_seasons:
        return create_success_response(
            data=[],
            total_available=0,
            truncated=False,
            warning=" ".join(warnings) if warnings else "No valid seasons provided.",
        )

    # Normalize filters
    combined_filters = normalize_filters(filters)

    # Fetch the data using generic fetch with filters and pagination
    fetcher = DataFetcher()
    result = fetcher.fetch(
        "schedules",
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
        return create_success_response(
            data=result.data,
            total_available=result.metadata.total_available,
            truncated=result.metadata.truncated,
            columns=result.metadata.columns,
            warning=combined_warning if combined_warning else None,
        )

    return result
