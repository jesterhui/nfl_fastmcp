"""Game schedules data retrieval tool.

This module provides the MCP tool for retrieving NFL game schedules
including game results, dates, and team matchups.
"""

from typing import Any

from fast_nfl_mcp.core.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)
from fast_nfl_mcp.data.fetcher import NFLDataPyFetcher
from fast_nfl_mcp.utils.constants import MAX_SEASONS_SCHEDULES
from fast_nfl_mcp.utils.helpers import add_warnings_to_response
from fast_nfl_mcp.utils.validation import normalize_filters, validate_seasons


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
    fetcher = NFLDataPyFetcher()
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
        return add_warnings_to_response(result, warnings)

    return result
