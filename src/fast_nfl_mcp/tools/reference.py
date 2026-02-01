"""Reference data retrieval tools.

This module provides tools for accessing NFL reference data that doesn't
require season parameters. These datasets contain metadata about players,
teams, officials, and contracts.
"""

import logging
from typing import Any

import pandas as pd

from fast_nfl_mcp.constants import (
    DEFAULT_PLAYER_IDS_CACHE_TTL_SECONDS,
    LOOKUP_PLAYER_COLUMNS,
    LOOKUP_PLAYER_DEFAULT_LIMIT,
    LOOKUP_PLAYER_MAX_LIMIT,
)
from fast_nfl_mcp.enums import DatasetName
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.nfl_data_py_fetcher import NFLDataPyFetcher
from fast_nfl_mcp.redis_cache import (
    PLAYER_IDS_CACHE_KEY,
    get_cached_dataframe,
    set_cached_dataframe,
)
from fast_nfl_mcp.schema_manager import DATASET_DEFINITIONS
from fast_nfl_mcp.serialization import convert_value

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
    fetcher = NFLDataPyFetcher()
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
    fetcher = NFLDataPyFetcher()
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
    fetcher = NFLDataPyFetcher()
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
    fetcher = NFLDataPyFetcher()
    return fetcher.fetch(
        "contracts",
        offset=offset,
        limit=limit,
    )


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
    definition = DATASET_DEFINITIONS.get(DatasetName.PLAYER_IDS)
    if definition is None:
        return create_error_response(
            error="player_ids dataset not found in configuration."
        )

    try:
        logger.info(f"Looking up player with name: {name}")

        # Try to get player_ids from cache first
        df = get_cached_dataframe(PLAYER_IDS_CACHE_KEY)

        if df is None:
            # Cache miss - load from source
            logger.info("Cache miss for player_ids, loading from source")
            df = definition.loader(None)

            # Cache the result if load succeeded
            if df is not None and not df.empty:
                ttl = DEFAULT_PLAYER_IDS_CACHE_TTL_SECONDS
                if set_cached_dataframe(PLAYER_IDS_CACHE_KEY, df, ttl):
                    logger.info(f"Cached player_ids DataFrame (TTL: {ttl}s)")
        else:
            logger.info("Cache hit for player_ids")

        if df is None or df.empty:
            return create_success_response(
                data=[],
                total_available=0,
                truncated=False,
                columns=LOOKUP_PLAYER_COLUMNS,
                warning="No player data available.",
            )

        # Match against both merge_name AND name columns to handle cases where
        # the user's search term might match one but not the other (e.g., "T.J."
        # vs "tj", "Jr." vs normalized without suffix)
        mask = pd.Series([False] * len(df), index=df.index)

        if "merge_name" in df.columns:
            # merge_name is already lowercase, so just check contains
            mask = mask | df["merge_name"].fillna("").str.contains(
                search_name, regex=False
            )

        if "name" in df.columns:
            # Also check against name column with case-insensitive matching
            mask = mask | df["name"].fillna("").str.lower().str.contains(
                search_name, regex=False
            )

        if "merge_name" not in df.columns and "name" not in df.columns:
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

        # Convert to records using list comprehension to ensure Python native types
        # (astype(object) leaves numpy scalars intact, breaking JSON serialization)
        for col in available_cols:
            dtype = result_df[col].dtype
            # Convert datetime columns to strings
            if pd.api.types.is_datetime64_any_dtype(dtype):
                values = [None if pd.isna(v) else str(v) for v in result_df[col]]
                result_df[col] = pd.Series(values, index=result_df.index, dtype=object)
            else:
                # Use convert_value to convert numpy scalars to native Python
                # and NaN/NA to None (must rebuild Series to preserve None values)
                values = [convert_value(v) for v in result_df[col]]
                result_df[col] = pd.Series(values, index=result_df.index, dtype=object)

        # Use to_dict for efficient conversion
        records: list[dict[str, Any]] = [
            {str(k): v for k, v in r.items()}
            for r in result_df.to_dict(orient="records")
        ]

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

    except (OSError, ConnectionError, TimeoutError) as e:
        # Network and I/O errors
        logger.error(f"Network error looking up player '{name}': {e}")
        return create_error_response(error=f"Network error looking up player: {str(e)}")

    except KeyError as e:
        # Missing expected columns in the data
        logger.error(f"Data format error looking up player '{name}': {e}")
        return create_error_response(
            error=f"Data format error: missing expected column {str(e)}"
        )

    except (ValueError, RuntimeError) as e:
        # Validation and runtime errors from the library
        logger.error(f"Error looking up player '{name}': {e}")
        return create_error_response(error=f"Error looking up player: {str(e)}")

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error looking up player '{name}': {e}")
        return create_error_response(error=f"Error looking up player: {str(e)}")
