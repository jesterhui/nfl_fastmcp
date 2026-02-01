"""Data fetcher for NFL data with error handling and row limits.

This module provides the DataFetcher class that wraps nfl_data_py calls
with consistent error handling, row limits, and standardized response
formatting using the Pydantic models defined in models.py.
"""

import logging
from typing import Any

import pandas as pd

from fast_nfl_mcp.constants import DEFAULT_MAX_ROWS
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.schema_manager import DATASET_DEFINITIONS

logger = logging.getLogger(__name__)


def _convert_object_value(val: Any) -> Any:
    """Convert non-serializable values in object columns to JSON-safe types.

    Handles pd.Timestamp, numpy scalars, and NaN values that may appear
    in object-dtype columns (common after merges or with mixed-type data).
    """
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return str(val)
    if hasattr(val, "item"):
        return val.item()
    return val


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""

    pass


class DataFetcher:
    """Fetches NFL data with error handling and row limits.

    The DataFetcher wraps nfl_data_py function calls with:
    - Consistent error handling for network failures
    - Row limits to prevent excessive data responses
    - Pagination support via offset parameter
    - Standardized response formatting using Pydantic models

    Attributes:
        MAX_ROWS: Maximum number of rows to return (default 10).
    """

    MAX_ROWS: int = DEFAULT_MAX_ROWS

    def __init__(self, max_rows: int | None = None) -> None:
        """Initialize the DataFetcher.

        Args:
            max_rows: Optional override for the maximum row limit.
                     If None, uses the class default (100).
        """
        if max_rows is not None:
            self._max_rows = max_rows
        else:
            self._max_rows = self.MAX_ROWS

    def _get_dataset_loader(
        self, dataset: str
    ) -> tuple[Any, str, bool, int | None] | None:
        """Get the loader function and metadata for a dataset.

        Args:
            dataset: The name of the dataset to fetch.

        Returns:
            A tuple of (loader_function, description, supports_seasons, default_season)
            or None if the dataset is not found.
        """
        return DATASET_DEFINITIONS.get(dataset)

    def _build_params_for_loader(
        self,
        supports_seasons: bool,
        default_season: int | None,
        params: dict[str, Any],
    ) -> Any:
        """Build the parameter to pass to the loader function.

        Args:
            supports_seasons: Whether the dataset supports season filtering.
            default_season: The default season to use if none provided.
            params: User-provided parameters.

        Returns:
            The parameter to pass to the loader (seasons list or None).
        """
        if supports_seasons:
            # Extract seasons from params, use default if not provided
            seasons = params.get("seasons")
            if seasons is None:
                if default_season is not None:
                    return [default_season]
                return None
            # Ensure seasons is a list
            if isinstance(seasons, int):
                return [seasons]
            return list(seasons)
        return None

    def _convert_dataframe_to_records(
        self, df: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert a DataFrame to a list of dictionaries.

        Handles special cases like NaN values and numpy types to ensure
        JSON serialization compatibility.

        Args:
            df: The DataFrame to convert.

        Returns:
            A tuple of (list of row dictionaries, list of column names).
        """
        if df is None or df.empty:
            return [], []

        # Pre-compute column names once (avoid repeated str() calls in loops)
        columns: list[str] = [str(col) for col in df.columns]

        # Create a working copy for vectorized transformations
        result_df = df.copy()

        # Vectorized type conversions per column (much faster than per-cell checks)
        for col in result_df.columns:
            dtype = result_df[col].dtype

            # Convert Timestamp columns to strings vectorially
            if pd.api.types.is_datetime64_any_dtype(dtype):
                result_df[col] = result_df[col].astype(str)
                # Replace 'NaT' strings with None
                result_df[col] = result_df[col].replace("NaT", None)
            # Convert numpy integer types to Python native (preserving NaN as None)
            elif pd.api.types.is_integer_dtype(dtype):
                result_df[col] = result_df[col].astype(object)
                result_df.loc[result_df[col].isna(), col] = None
            # Convert numpy float types to Python native (preserving NaN as None)
            elif pd.api.types.is_float_dtype(dtype):
                mask = result_df[col].isna()
                result_df[col] = result_df[col].astype(object)
                result_df.loc[mask, col] = None
            # Handle string and object dtype columns - replace NaN with None
            # and convert any embedded Timestamps/numpy scalars
            # Must rebuild Series from list to preserve None (pandas converts None to nan)
            elif pd.api.types.is_string_dtype(dtype) or dtype == object:
                values = [_convert_object_value(v) for v in result_df[col]]
                result_df[col] = pd.Series(values, index=result_df.index, dtype=object)

        # Convert to records - now all values should be JSON-serializable
        records = result_df.to_dict(orient="records")

        # Ensure all keys are strings (handle non-string column names)
        cleaned_records: list[dict[str, Any]] = [
            {str(k): v for k, v in record.items()} for record in records
        ]

        return cleaned_records, columns

    def _apply_filters(
        self, df: pd.DataFrame, filters: dict[str, list[Any]]
    ) -> pd.DataFrame:
        """Apply filters to a DataFrame.

        Args:
            df: The DataFrame to filter.
            filters: Dict mapping column names to lists of acceptable values.

        Returns:
            Filtered DataFrame.
        """
        for column, values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(values)]
        return df

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
            >>> fetcher = DataFetcher()
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

        loader, description, supports_seasons, default_season = definition

        # Build parameters for the loader
        loader_param = self._build_params_for_loader(
            supports_seasons, default_season, params
        )

        try:
            logger.info(f"Fetching data for {dataset} with params: {params}")

            # Call the loader function
            df = loader(loader_param)

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
            if filters:
                df = self._apply_filters(df, filters)
                if df.empty:
                    return create_success_response(
                        data=[],
                        total_available=0,
                        truncated=False,
                        columns=[str(col) for col in df.columns],
                        warning="No data matched the specified filters.",
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
            records, columns = self._convert_dataframe_to_records(df)

            # Build warning message if truncated
            warning = None
            if truncated:
                next_offset = offset + effective_limit
                warning = (
                    f"Results truncated. Showing {effective_limit} of {total_available} total rows "
                    f"(offset: {offset}). Use offset={next_offset} to get the next page."
                )

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

        _, description, supports_seasons, default_season = definition
        return {
            "name": dataset,
            "description": description,
            "supports_seasons": supports_seasons,
            "default_season": default_season,
        }
