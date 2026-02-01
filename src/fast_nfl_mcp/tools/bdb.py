"""Big Data Bowl tracking data retrieval tools.

This module provides MCP tools for accessing NFL Big Data Bowl (BDB) 2025
competition data via the Kaggle API. The data includes:
- Game metadata
- Play-level data
- Player information
- Per-frame tracking data (x, y coordinates, speed, acceleration, orientation)

Note: Users must configure ~/.kaggle/kaggle.json and accept the competition
rules before using these tools.
"""

from typing import Any

from fast_nfl_mcp.constants import BDB_AVAILABLE_WEEKS
from fast_nfl_mcp.kaggle_fetcher import fetch_bdb_data
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse, create_error_response
from fast_nfl_mcp.tools.validation import normalize_filters


def get_bdb_games_impl(
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get BDB game metadata.

    Retrieves game information from the NFL Big Data Bowl 2025 dataset.
    Includes game IDs, teams, scores, and other metadata.

    Args:
        columns: Optional list of column names to include.
        filters: Optional dict to filter on any column.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 50, max 100).

    Returns:
        SuccessResponse with game data, or ErrorResponse on failure.
    """
    normalized_filters = normalize_filters(filters)

    return fetch_bdb_data(
        data_type="games",
        filters=normalized_filters,
        offset=offset,
        limit=limit,
        columns=columns,
    )


def get_bdb_plays_impl(
    game_id: int | None = None,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get BDB play-level data.

    Retrieves play information from the NFL Big Data Bowl 2025 dataset.
    Includes play descriptions, formations, and outcomes.

    Args:
        game_id: Optional game ID to filter plays for a specific game.
        columns: Optional list of column names to include.
        filters: Optional dict to filter on any column.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 50, max 100).

    Returns:
        SuccessResponse with play data, or ErrorResponse on failure.
    """
    normalized_filters = normalize_filters(filters)

    # Add game_id filter if provided
    if game_id is not None:
        normalized_filters["game_id"] = [game_id]

    return fetch_bdb_data(
        data_type="plays",
        filters=normalized_filters,
        offset=offset,
        limit=limit,
        columns=columns,
    )


def get_bdb_players_impl(
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get BDB player information.

    Retrieves player data from the NFL Big Data Bowl 2025 dataset.
    Includes player names, positions, and physical attributes.

    Args:
        columns: Optional list of column names to include.
        filters: Optional dict to filter on any column.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 50, max 100).

    Returns:
        SuccessResponse with player data, or ErrorResponse on failure.
    """
    normalized_filters = normalize_filters(filters)

    return fetch_bdb_data(
        data_type="players",
        filters=normalized_filters,
        offset=offset,
        limit=limit,
        columns=columns,
    )


def get_bdb_tracking_impl(
    week: int,
    game_id: int | None = None,
    play_id: int | None = None,
    nfl_id: int | None = None,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Get BDB per-frame tracking data.

    Retrieves high-frequency tracking data from the NFL Big Data Bowl 2025
    dataset. Each row represents a single frame (10 frames per second) with
    player position (x, y), speed, acceleration, and orientation.

    IMPORTANT: Tracking data is very large. Always filter by game_id and/or
    play_id to avoid extremely slow queries.

    Args:
        week: Week number (1-9). Required.
        game_id: Game ID to filter. Strongly recommended.
        play_id: Play ID to filter. Requires game_id.
        nfl_id: NFL player ID to filter for a specific player.
        columns: Optional list of column names to include.
        filters: Optional dict to filter on any column.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 50, max 100).

    Returns:
        SuccessResponse with tracking data, or ErrorResponse on failure.
    """
    # Validate week
    if week not in BDB_AVAILABLE_WEEKS:
        return create_error_response(
            error=f"Invalid week: {week}. "
            f"Valid weeks for BDB 2025 are: {list(BDB_AVAILABLE_WEEKS)}"
        )

    normalized_filters = normalize_filters(filters)

    # Validate play_id requires game_id (either as parameter or in filters)
    has_game_id = game_id is not None or "game_id" in normalized_filters
    if play_id is not None and not has_game_id:
        return create_error_response(
            error="play_id filter requires game_id. "
            "Please specify game_id along with play_id."
        )

    # Add provided filters
    if game_id is not None:
        normalized_filters["game_id"] = [game_id]
    if play_id is not None:
        normalized_filters["play_id"] = [play_id]
    if nfl_id is not None:
        normalized_filters["nfl_id"] = [nfl_id]

    return fetch_bdb_data(
        data_type="tracking",
        week=week,
        filters=normalized_filters,
        offset=offset,
        limit=limit,
        columns=columns,
    )
