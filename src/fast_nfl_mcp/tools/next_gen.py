"""Next Gen Stats data retrieval tools.

This module provides MCP tools for retrieving NFL Next Gen Stats (NGS)
data for passing, rushing, and receiving metrics. NGS data includes
advanced tracking metrics captured by on-field sensors.
"""

from typing import Any

from fast_nfl_mcp.core.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)
from fast_nfl_mcp.data.fetcher import NFLDataPyFetcher
from fast_nfl_mcp.utils.constants import MAX_SEASONS_NGS, MIN_SEASON_NGS
from fast_nfl_mcp.utils.helpers import add_warnings_to_response
from fast_nfl_mcp.utils.validation import normalize_filters


def validate_ngs_seasons(
    seasons: list[int], max_seasons: int
) -> tuple[list[int], str | None]:
    """Validate the seasons parameter with a specified limit for NGS data.

    NGS data is only available from 2016 onwards.

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
    # NGS data is only available from 2016 onwards
    valid_seasons = []
    invalid_seasons = []
    for season in seasons:
        if season >= MIN_SEASON_NGS:
            valid_seasons.append(season)
        else:
            invalid_seasons.append(season)

    if invalid_seasons:
        warnings.append(
            f"Invalid seasons removed: {invalid_seasons}. "
            f"Next Gen Stats data is available from {MIN_SEASON_NGS} onwards."
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


def get_ngs_passing_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get Next Gen Stats passing metrics.

    Retrieves advanced passing statistics from NFL Next Gen Stats,
    which uses player tracking data to calculate metrics like
    time to throw, air yards, and completion probability.

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
        columns: List of column names to include in output (required).
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with NGS passing data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_ngs_seasons(seasons, MAX_SEASONS_NGS)
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
    fetcher = NFLDataPyFetcher()
    result = fetcher.fetch(
        "ngs_passing",
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


def get_ngs_rushing_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get Next Gen Stats rushing metrics.

    Retrieves advanced rushing statistics from NFL Next Gen Stats,
    which uses player tracking data to calculate metrics like
    rush yards over expected, time behind line of scrimmage, and efficiency.

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
        columns: List of column names to include in output (required).
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with NGS rushing data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_ngs_seasons(seasons, MAX_SEASONS_NGS)
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
    fetcher = NFLDataPyFetcher()
    result = fetcher.fetch(
        "ngs_rushing",
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


def get_ngs_receiving_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get Next Gen Stats receiving metrics.

    Retrieves advanced receiving statistics from NFL Next Gen Stats,
    which uses player tracking data to calculate metrics like
    separation, catch probability, and cushion.

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
        columns: List of column names to include in output (required).
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with NGS receiving data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_ngs_seasons(seasons, MAX_SEASONS_NGS)
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
    fetcher = NFLDataPyFetcher()
    result = fetcher.fetch(
        "ngs_receiving",
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
