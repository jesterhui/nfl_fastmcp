"""Reference data retrieval tools.

This module provides tools for accessing NFL reference data that doesn't
require season parameters. These datasets contain metadata about players,
teams, officials, and contracts.
"""

import logging
from typing import Any

import pandas as pd

from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.schema_manager import DATASET_DEFINITIONS

logger = logging.getLogger(__name__)


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


# Default and maximum limits for lookup_player
LOOKUP_PLAYER_DEFAULT_LIMIT = 10
LOOKUP_PLAYER_MAX_LIMIT = 100

# Columns to return from lookup_player
LOOKUP_PLAYER_COLUMNS = ["gsis_id", "name", "team", "position", "merge_name"]


def lookup_player_impl(
    name: str,
    limit: int | None = None,
) -> SuccessResponse | ErrorResponse:
    """Search for players by name and return their player_id (gsis_id).

    This utility tool enables searching for players by name to find their
    gsis_id, which can then be used with datasets that only contain player_id
    (like seasonal_stats).

    Args:
        name: Player name to search for (case-insensitive, partial match supported).
        limit: Maximum number of results to return (default 10, max 100).

    Returns:
        SuccessResponse with matching players including gsis_id, name, team,
        position, and merge_name columns, or ErrorResponse on failure.

    Examples:
        >>> lookup_player_impl("Mahomes")
        # Returns players matching "Mahomes" (e.g., Patrick Mahomes)
        >>> lookup_player_impl("jameis winston")
        # Case-insensitive match for Jameis Winston
    """
    # Validate name parameter
    if not name or not name.strip():
        return create_success_response(
            data=[],
            total_available=0,
            truncated=False,
            columns=LOOKUP_PLAYER_COLUMNS,
            warning="Empty search name provided. Please provide a player name to search.",
        )

    # Normalize the search name
    search_name = name.strip().lower()

    # Determine effective limit
    if limit is None:
        effective_limit = LOOKUP_PLAYER_DEFAULT_LIMIT
    else:
        effective_limit = min(max(1, limit), LOOKUP_PLAYER_MAX_LIMIT)

    # Get the player_ids loader from DATASET_DEFINITIONS
    definition = DATASET_DEFINITIONS.get("player_ids")
    if definition is None:
        return create_error_response(
            error="player_ids dataset not found in configuration."
        )

    loader, _, _, _ = definition

    try:
        logger.info(f"Looking up player with name: {name}")

        # Load the player_ids dataset
        df = loader(None)

        if df is None or df.empty:
            return create_success_response(
                data=[],
                total_available=0,
                truncated=False,
                columns=LOOKUP_PLAYER_COLUMNS,
                warning="No player data available.",
            )

        # Use merge_name for matching if available (it's already normalized/lowercase)
        # Otherwise fall back to the name column
        if "merge_name" in df.columns:
            # merge_name is already lowercase, so just check contains
            mask = df["merge_name"].fillna("").str.contains(search_name, regex=False)
        elif "name" in df.columns:
            # Fall back to name column with case-insensitive matching
            mask = (
                df["name"].fillna("").str.lower().str.contains(search_name, regex=False)
            )
        else:
            return create_error_response(
                error="Player name column not found in player_ids dataset."
            )

        # Filter to matching rows
        matching_df = df[mask]

        # Get total matches before limiting
        total_matches = len(matching_df)

        if total_matches == 0:
            return create_success_response(
                data=[],
                total_available=0,
                truncated=False,
                columns=LOOKUP_PLAYER_COLUMNS,
                warning=f"No players found matching '{name}'.",
            )

        # Select only the required columns (if they exist)
        available_cols = [
            col for col in LOOKUP_PLAYER_COLUMNS if col in matching_df.columns
        ]
        if not available_cols:
            return create_error_response(
                error=f"Required columns not found in player_ids dataset. "
                f"Expected: {LOOKUP_PLAYER_COLUMNS}"
            )

        result_df = matching_df[available_cols]

        # Check if results will be truncated
        truncated = total_matches > effective_limit

        # Apply limit
        if truncated:
            result_df = result_df.head(effective_limit)

        # Convert to records, handling NaN values
        records: list[dict[str, Any]] = []
        for _, row in result_df.iterrows():
            record: dict[str, Any] = {}
            for col in available_cols:
                value = row[col]
                if pd.isna(value):
                    record[col] = None
                elif hasattr(value, "item"):
                    record[col] = value.item()
                else:
                    record[col] = value
            records.append(record)

        # Build warning message
        warning = None
        if truncated:
            warning = (
                f"Found {total_matches} players matching '{name}'. "
                f"Showing first {effective_limit} results. "
                f"Use a more specific name to narrow results."
            )

        return create_success_response(
            data=records,
            total_available=total_matches,
            truncated=truncated,
            columns=available_cols,
            warning=warning,
        )

    except Exception as e:
        logger.error(f"Error looking up player '{name}': {e}")
        return create_error_response(error=f"Error looking up player: {str(e)}")
