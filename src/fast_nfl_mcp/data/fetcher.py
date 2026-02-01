"""Data fetcher for NFL data with error handling and row limits.

This module provides the NFLDataPyFetcher class that wraps nfl_data_py calls
with consistent error handling, row limits, and standardized response
formatting using the Pydantic models defined in models.py.
"""

import logging
from typing import Any

import pandas as pd

from fast_nfl_mcp.core.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.data.schema import DATASET_DEFINITIONS
from fast_nfl_mcp.utils.constants import DEFAULT_MAX_ROWS, get_current_season_year
from fast_nfl_mcp.utils.serialization import convert_dataframe_to_records
from fast_nfl_mcp.utils.types import DatasetDefinition

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

    def _validate_dataset(
        self, dataset: str
    ) -> tuple[DatasetDefinition | None, ErrorResponse | None]:
        """Validate the dataset name and return its definition.

        Args:
            dataset: The name of the dataset to validate.

        Returns:
            A tuple of (DatasetDefinition, None) if valid, or (None, ErrorResponse) if invalid.
        """
        definition = self._get_dataset_loader(dataset)
        if definition is None:
            valid_datasets = list(DATASET_DEFINITIONS.keys())
            error = create_error_response(
                error=f"Unknown dataset: '{dataset}'. "
                f"Valid datasets are: {', '.join(sorted(valid_datasets))}"
            )
            return None, error
        return definition, None

    def _handle_empty_dataframe(self, dataset: str) -> SuccessResponse:
        """Create a response for empty or None DataFrame results.

        Args:
            dataset: The name of the dataset that returned no data.

        Returns:
            A SuccessResponse indicating no data was found.
        """
        logger.warning(f"Dataset {dataset} returned no data")
        return create_success_response(
            data=[],
            total_available=0,
            truncated=False,
            columns=[],
            warning=f"No data found for {dataset} with the given parameters.",
        )

    def _handle_empty_filter_results(
        self, df: pd.DataFrame, invalid_filter_cols: list[str]
    ) -> SuccessResponse:
        """Create a response when filters produce no matching rows.

        Args:
            df: The empty DataFrame (with column info preserved).
            invalid_filter_cols: List of filter columns that didn't exist.

        Returns:
            A SuccessResponse indicating no data matched the filters.
        """
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

    def _validate_columns(
        self, df: pd.DataFrame, columns: list[str], dataset: str
    ) -> tuple[pd.DataFrame | None, ErrorResponse | None]:
        """Validate and select requested columns from the DataFrame.

        Args:
            df: The DataFrame to select columns from.
            columns: List of column names to select.
            dataset: The dataset name (for error messages).

        Returns:
            A tuple of (filtered DataFrame, None) if valid columns exist,
            or (None, ErrorResponse) if validation fails.
        """
        if not columns:
            return None, create_error_response(
                error="Empty columns list provided. "
                "Please specify at least one column name."
            )

        available_cols = set(df.columns)
        valid_cols = [c for c in columns if c in available_cols]
        invalid_cols = [c for c in columns if c not in available_cols]

        if not valid_cols:
            return None, create_error_response(
                error=f"None of the requested columns exist: {invalid_cols}. "
                f"Use describe_dataset('{dataset}') to see available columns."
            )

        if invalid_cols:
            logger.warning(f"Requested columns not found: {invalid_cols}")

        return df[valid_cols], None

    def _paginate_dataframe(
        self, df: pd.DataFrame, offset: int, limit: int
    ) -> tuple[pd.DataFrame, int, bool]:
        """Apply pagination to a DataFrame.

        Args:
            df: The DataFrame to paginate.
            offset: Number of rows to skip.
            limit: Maximum number of rows to return.

        Returns:
            A tuple of (paginated DataFrame, total rows before pagination, was truncated).
        """
        total_available = len(df)

        # Apply offset first
        if offset > 0:
            df = df.iloc[offset:]

        # Check if results will be truncated
        rows_after_offset = len(df)
        truncated = rows_after_offset > limit

        # Apply row limit
        if truncated:
            df = df.head(limit)

        return df, total_available, truncated

    def _build_warning_message(
        self,
        invalid_filter_cols: list[str],
        truncated: bool,
        offset: int,
        effective_limit: int,
        total_available: int,
    ) -> str | None:
        """Build a warning message from various conditions.

        Args:
            invalid_filter_cols: List of filter columns that didn't exist.
            truncated: Whether results were truncated.
            offset: The offset used for pagination.
            effective_limit: The row limit applied.
            total_available: Total rows before truncation.

        Returns:
            A warning message string, or None if no warnings.
        """
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

        return " ".join(warnings) if warnings else None

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
        # Initialize defaults
        params = params or {}
        filters = filters or {}

        # Phase 1: Validate dataset
        definition, error = self._validate_dataset(dataset)
        if error:
            return error
        assert definition is not None  # Guaranteed by _validate_dataset

        # Phase 2: Load data
        loader_param = self._build_params_for_loader(
            definition.supports_seasons, params
        )

        try:
            logger.info(f"Fetching data for {dataset} with params: {params}")
            df = definition.loader(loader_param)

            # Phase 3: Handle empty data
            if df is None or df.empty:
                return self._handle_empty_dataframe(dataset)

            # Phase 4: Apply filters
            invalid_filter_cols: list[str] = []
            if filters:
                df, invalid_filter_cols = self._apply_filters(df, filters)
                if df.empty:
                    return self._handle_empty_filter_results(df, invalid_filter_cols)

            # Phase 5: Select columns
            if columns is not None:
                selected_df, error = self._validate_columns(df, columns, dataset)
                if error:
                    return error
                assert selected_df is not None  # Guaranteed by _validate_columns
                df = selected_df

            # Phase 6: Paginate
            effective_limit = limit if limit is not None else self._max_rows
            df, total_available, truncated = self._paginate_dataframe(
                df, offset, effective_limit
            )

            if offset > 0:
                logger.info(f"Applied offset {offset} to {dataset}")
            if truncated:
                logger.info(f"Truncated {dataset} to {effective_limit} rows")

            # Phase 7: Build response
            records, result_columns = convert_dataframe_to_records(df)
            warning = self._build_warning_message(
                invalid_filter_cols, truncated, offset, effective_limit, total_available
            )

            return create_success_response(
                data=records,
                total_available=total_available,
                truncated=truncated,
                columns=result_columns,
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
            logger.warning(f"Invalid parameters for {dataset}: {e}")
            return create_success_response(
                data=[],
                total_available=0,
                truncated=False,
                columns=[],
                warning=f"Invalid parameters for {dataset}: {str(e)}",
            )

        except OSError as e:
            logger.error(f"OS error fetching {dataset}: {e}")
            return create_error_response(
                error=f"System error fetching {dataset}: {str(e)}"
            )

        except RuntimeError as e:
            logger.error(f"Runtime error fetching {dataset}: {e}")
            return create_error_response(error=f"Error fetching {dataset}: {str(e)}")

        except Exception as e:
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
