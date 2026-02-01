"""Miscellaneous NFL data retrieval tools.

This module provides MCP tools for retrieving miscellaneous NFL data
including snap counts, injuries, depth charts, combine data, and QBR ratings.
"""

from typing import Any

from fast_nfl_mcp.core.models import (
    ErrorResponse,
    SuccessResponse,
    create_success_response,
)
from fast_nfl_mcp.data.fetcher import NFLDataPyFetcher
from fast_nfl_mcp.utils.constants import (
    MAX_SEASONS_COMBINE,
    MAX_SEASONS_DEPTH_CHARTS,
    MAX_SEASONS_INJURIES,
    MAX_SEASONS_QBR,
    MAX_SEASONS_SNAP_COUNTS,
)
from fast_nfl_mcp.utils.helpers import add_warnings_to_response
from fast_nfl_mcp.utils.validation import normalize_filters, validate_seasons


def get_snap_counts_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL player snap count data.

    Retrieves snap participation data for players across specified seasons.
    Includes snap counts for offense, defense, and special teams.

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
        columns: List of column names to include in output.
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with snap count data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(seasons, MAX_SEASONS_SNAP_COUNTS)
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
        "snap_counts",
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


def get_injuries_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL injury report data.

    Retrieves injury reports and player status for specified seasons.
    Includes injury type, practice participation, and game status.

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
        columns: List of column names to include in output.
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with injury data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(seasons, MAX_SEASONS_INJURIES)
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
        "injuries",
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


def get_depth_charts_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL team depth chart data.

    Retrieves depth chart information showing player positions and rankings
    on their respective teams for specified seasons.

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
        columns: List of column names to include in output.
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with depth chart data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(seasons, MAX_SEASONS_DEPTH_CHARTS)
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
        "depth_charts",
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


def get_combine_data_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL combine data.

    Retrieves NFL Scouting Combine results including player measurements
    and athletic testing results (40-yard dash, bench press, etc.).

    Args:
        seasons: List of seasons (e.g., [2020, 2021, 2022]). Maximum 10 seasons allowed.
        columns: List of column names to include in output.
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with combine data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(seasons, MAX_SEASONS_COMBINE)
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
        "combine_data",
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


def get_qbr_impl(
    seasons: list[int],
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get ESPN QBR (Total Quarterback Rating) data.

    Retrieves ESPN's proprietary Total QBR metric for quarterbacks.
    QBR is a comprehensive quarterback rating that accounts for all
    aspects of quarterback play including rushing, turnovers, and
    the value of plays in different game situations.

    Note: QBR data is only available from 2006 onwards.

    Args:
        seasons: List of seasons (e.g., [2020, 2021, 2022]). Maximum 10 seasons allowed.
        columns: List of column names to include in output.
        filters: Optional dict mapping column names to filter values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        SuccessResponse with QBR data (up to 100 rows),
        or ErrorResponse on failure.
    """
    warnings: list[str] = []

    # Validate seasons
    valid_seasons, season_warning = validate_seasons(seasons, MAX_SEASONS_QBR)
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
        "qbr",
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
