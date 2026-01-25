"""Data fetcher for NFL data with error handling and row limits.

This module provides the DataFetcher class that wraps nfl_data_py calls
with consistent error handling, row limits, and standardized response
formatting using the Pydantic models defined in models.py.
"""

import logging
from typing import Any

import pandas as pd

from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.schema_manager import DATASET_DEFINITIONS

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""

    pass


class DataFetcher:
    """Fetches NFL data with error handling and row limits.

    The DataFetcher wraps nfl_data_py function calls with:
    - Consistent error handling for network failures
    - Row limits to prevent excessive data responses
    - Standardized response formatting using Pydantic models

    Attributes:
        MAX_ROWS: Maximum number of rows to return (default 100).
    """

    MAX_ROWS: int = 100

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

        # Convert DataFrame to records, handling NaN values
        # Use 'records' orientation for list of dicts
        records = df.to_dict(orient="records")

        # Clean up NaN values to None for JSON serialization
        cleaned_records: list[dict[str, Any]] = []
        for record in records:
            cleaned: dict[str, Any] = {}
            for key, value in record.items():
                str_key = str(key)
                if pd.isna(value):
                    cleaned[str_key] = None
                elif hasattr(value, "item"):
                    # Convert numpy types to Python native types
                    cleaned[str_key] = value.item()
                elif isinstance(value, pd.Timestamp):
                    cleaned[str_key] = str(value)
                else:
                    cleaned[str_key] = value
            cleaned_records.append(cleaned)

        columns: list[str] = [str(col) for col in df.columns]
        return cleaned_records, columns

    def fetch(
        self, dataset: str, params: dict[str, Any] | None = None
    ) -> SuccessResponse | ErrorResponse:
        """Fetch data from nfl_data_py with error handling and row limits.

        Args:
            dataset: The name of the dataset to fetch (e.g., "play_by_play").
            params: Optional dictionary of parameters for the fetch.
                   For seasonal datasets, can include "seasons" as a list
                   or single integer.

        Returns:
            A SuccessResponse with data and metadata, or an ErrorResponse
            if an error occurred.

        Examples:
            >>> fetcher = DataFetcher()
            >>> response = fetcher.fetch("play_by_play", {"seasons": [2024]})
            >>> response = fetcher.fetch("team_descriptions")
        """
        if params is None:
            params = {}

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

            # Get total count before truncation
            total_available = len(df)
            truncated = total_available > self._max_rows

            # Apply row limit
            if truncated:
                df = df.head(self._max_rows)
                logger.info(
                    f"Truncated {dataset} from {total_available} "
                    f"to {self._max_rows} rows"
                )

            # Convert to records
            records, columns = self._convert_dataframe_to_records(df)

            # Build warning message if truncated
            warning = None
            if truncated:
                warning = (
                    f"Results truncated from {total_available} to {self._max_rows} rows. "
                    f"Use more specific filters to reduce the result set."
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
