"""Data fetcher for NFL data with error handling and row limits.

This module provides the NFLDataPyFetcher class that wraps nfl_data_py calls
with consistent error handling, row limits, and standardized response
formatting using the Pydantic models defined in models.py.
"""

import logging
from typing import Any

import pandas as pd

from fast_nfl_mcp.constants import DEFAULT_MAX_ROWS, get_current_season_year
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.schema_manager import DATASET_DEFINITIONS
from fast_nfl_mcp.serialization import convert_dataframe_to_records
from fast_nfl_mcp.types import DatasetDefinition

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""

    pass


class NFLDataPyFetcher:
    """Fetches NFL data from nfl_data_py with error handling and row limits.

    The NFLDataPyFetcher wraps nfl_data_py function calls with:
    - Consistent error handling for network failures
    - Row limits to prevent excessive data responses
    - Pagination support via offset parameter
    - Standardized response formatting using Pydantic models

    Attributes:
        MAX_ROWS: Maximum number of rows to return (default 100).
    """

    MAX_ROWS: int = DEFAULT_MAX_ROWS

    def __init__(self, max_rows: int | None = None) -> None:
        """Initialize the NFLDataPyFetcher.

        Args:
            max_rows: Optional override for the maximum row limit.
                     If None, uses the class default (100).
        """
        if max_rows is not None:
            self._max_rows = max_rows
        else:
            self._max_rows = self.MAX_ROWS

    def _get_dataset_loader(self, dataset: str) -> DatasetDefinition | None:
        """Get the loader function and metadata for a dataset.

        Args:
            dataset: The name of the dataset to fetch.

        Returns:
            A DatasetDefinition with loader, description, and supports_seasons,
            or None if the dataset is not found.
        """
        return DATASET_DEFINITIONS.get(dataset)

    def _build_params_for_loader(
        self,
        supports_seasons: bool,
        params: dict[str, Any],
    ) -> Any:
        """Build the parameter to pass to the loader function.

        Args:
            supports_seasons: Whether the dataset supports season filtering.
            params: User-provided parameters.

        Returns:
            The parameter to pass to the loader (seasons list or None).
        """
        if supports_seasons:
            # Extract seasons from params, use default if not provided
            seasons = params.get("seasons")
            if seasons is None:
                # Use the current season as the default
                return [get_current_season_year()]
            # Ensure seasons is a list
            if isinstance(seasons, int):
                return [seasons]
            return list(seasons)
        return None

    def _apply_filters(
        self, df: pd.DataFrame, filters: dict[str, list[Any]]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply filters to a DataFrame.

        Args:
            df: The DataFrame to filter.
            filters: Dict mapping column names to lists of acceptable values.

        Returns:
            A tuple of (filtered DataFrame, list of invalid filter column names).
        """
        invalid_filters: list[str] = []
        for column, values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(values)]
            else:
                invalid_filters.append(column)
        return df, invalid_filters

    def fetch(
        self,
        dataset: str,
        params: dict[str, Any] | None = None,
        filters: dict[str, list[Any]] | None = None,
        offset: int = 0,
        limit: int | None = None,
        columns: list[str] | None = None,
    ) -> SuccessResponse | ErrorResponse:
        """Fetch data from nfl_data_py with error handling and row limits.

        Args:
            dataset: The name of the dataset to fetch (e.g., "play_by_play").
            params: Optional dictionary of parameters for the fetch.
                   For seasonal datasets, can include "seasons" as a list
                   or single integer.
            filters: Optional dict mapping column names to lists of acceptable
                    values. Filters are applied BEFORE row limit truncation.
            offset: Number of rows to skip before returning results (for pagination).
                   Defaults to 0.
            limit: Maximum number of rows to return. If None, uses the instance
                  max_rows setting.
            columns: Optional list of column names to include in the output.
                    If None, all columns are included.

        Returns:
            A SuccessResponse with data and metadata, or an ErrorResponse
            if an error occurred.

        Examples:
            >>> fetcher = NFLDataPyFetcher()
            >>> response = fetcher.fetch("play_by_play", {"seasons": [2024]})
            >>> response = fetcher.fetch("play_by_play", {"seasons": [2024]}, {"week": [1, 2]})
            >>> response = fetcher.fetch("play_by_play", {"seasons": [2024]}, offset=10, limit=10)
            >>> response = fetcher.fetch("team_descriptions")
        """
        if params is None:
            params = {}
        if filters is None:
            filters = {}

        # Validate dataset name
        definition = self._get_dataset_loader(dataset)
        if definition is None:
            valid_datasets = list(DATASET_DEFINITIONS.keys())
            return create_error_response(
                error=f"Unknown dataset: '{dataset}'. "
                f"Valid datasets are: {', '.join(sorted(valid_datasets))}"
            )

        # Build parameters for the loader
        loader_param = self._build_params_for_loader(
            definition.supports_seasons, params
        )

        try:
            logger.info(f"Fetching data for {dataset} with params: {params}")

            # Call the loader function
            df = definition.loader(loader_param)

            # Handle None or empty DataFrame
            if df is None or df.empty:
                logger.warning(f"Dataset {dataset} returned no data")
                return create_success_response(
                    data=[],
                    total_available=0,
                    truncated=False,
                    columns=[],
                    warning=f"No data found for {dataset} with the given parameters.",
                )

            # Apply filters before truncation
            invalid_filter_cols: list[str] = []
            if filters:
                df, invalid_filter_cols = self._apply_filters(df, filters)
                if df.empty:
                    filter_warning = "No data matched the specified filters."
                    if invalid_filter_cols:
                        filter_warning += (
                            f" Note: The following filter columns do not exist "
                            f"in the dataset and were ignored: {invalid_filter_cols}"
                        )
                    return create_success_response(
                        data=[],
                        total_available=0,
                        truncated=False,
                        columns=[str(col) for col in df.columns],
                        warning=filter_warning,
                    )

            # Select specific columns if requested
            if columns is not None:
                if not columns:
                    return create_error_response(
                        error="Empty columns list provided. "
                        "Please specify at least one column name."
                    )
                available_cols = set(df.columns)
                valid_cols = [c for c in columns if c in available_cols]
                invalid_cols = [c for c in columns if c not in available_cols]
                if not valid_cols:
                    return create_error_response(
                        error=f"None of the requested columns exist: {invalid_cols}. "
                        f"Use describe_dataset('{dataset}') to see available columns."
                    )
                df = df[valid_cols]
                if invalid_cols:
                    logger.warning(f"Requested columns not found: {invalid_cols}")

            # Get total count before pagination
            total_available = len(df)

            # Determine effective limit
            effective_limit = limit if limit is not None else self._max_rows

            # Apply offset first
            if offset > 0:
                df = df.iloc[offset:]
                logger.info(f"Applied offset {offset} to {dataset}")

            # Check if results will be truncated
            rows_after_offset = len(df)
            truncated = rows_after_offset > effective_limit

            # Apply row limit
            if truncated:
                df = df.head(effective_limit)
                logger.info(
                    f"Truncated {dataset} from {rows_after_offset} "
                    f"to {effective_limit} rows"
                )

            # Convert to records
            records, columns = convert_dataframe_to_records(df)

            # Build warning message
            warnings: list[str] = []
            if invalid_filter_cols:
                warnings.append(
                    f"The following filter columns do not exist in the dataset "
                    f"and were ignored: {invalid_filter_cols}"
                )
            if truncated:
                next_offset = offset + effective_limit
                warnings.append(
                    f"Results truncated. Showing {effective_limit} of {total_available} total rows "
                    f"(offset: {offset}). Use offset={next_offset} to get the next page."
                )
            warning = " ".join(warnings) if warnings else None

            return create_success_response(
                data=records,
                total_available=total_available,
                truncated=truncated,
                columns=columns,
                warning=warning,
            )

        except ConnectionError as e:
            logger.error(f"Network error fetching {dataset}: {e}")
            return create_error_response(
                error=f"Network error: Unable to connect to data source. {str(e)}"
            )

        except TimeoutError as e:
            logger.error(f"Timeout fetching {dataset}: {e}")
            return create_error_response(
                error=f"Timeout error: Request timed out while fetching {dataset}. "
                f"Try again or use more specific filters."
            )

        except ValueError as e:
            # Invalid parameters typically raise ValueError
            logger.warning(f"Invalid parameters for {dataset}: {e}")
            return create_success_response(
                data=[],
                total_available=0,
                truncated=False,
                columns=[],
                warning=f"Invalid parameters for {dataset}: {str(e)}",
            )

        except OSError as e:
            # Catch OS-level errors (includes IOError, socket errors, etc.)
            logger.error(f"OS error fetching {dataset}: {e}")
            return create_error_response(
                error=f"System error fetching {dataset}: {str(e)}"
            )

        except RuntimeError as e:
            # Catch runtime errors from the nfl_data_py library
            logger.error(f"Runtime error fetching {dataset}: {e}")
            return create_error_response(error=f"Error fetching {dataset}: {str(e)}")

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error fetching {dataset}: {e}")
            return create_error_response(error=f"Error fetching {dataset}: {str(e)}")

    def get_available_datasets(self) -> list[str]:
        """Get a list of all available dataset names.

        Returns:
            A sorted list of dataset names that can be fetched.
        """
        return sorted(DATASET_DEFINITIONS.keys())

    def get_dataset_info(self, dataset: str) -> dict[str, Any] | None:
        """Get information about a specific dataset.

        Args:
            dataset: The name of the dataset.

        Returns:
            A dictionary with dataset information, or None if not found.
        """
        definition = self._get_dataset_loader(dataset)
        if definition is None:
            return None

        # For seasonal datasets, the default season is always the current season
        default_season = (
            get_current_season_year() if definition.supports_seasons else None
        )
        return {
            "name": dataset,
            "description": definition.description,
            "supports_seasons": definition.supports_seasons,
            "default_season": default_season,
        }
