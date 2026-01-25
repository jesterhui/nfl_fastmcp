"""Data fetcher for NFL data with error handling and row limits.

This module provides the DataFetcher class that wraps nfl_data_py calls
with consistent error handling, row limits, and response formatting.
All data retrieval tools should use DataFetcher to ensure consistent
behavior across the API.
"""

import logging
from typing import Any

import nfl_data_py as nfl
import pandas as pd

from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches NFL data with consistent error handling and row limits.

    The DataFetcher wraps nfl_data_py function calls to provide:
    - Consistent error handling for network failures
    - Automatic row limiting (default 100 rows)
    - Standardized response formatting
    - Parameter validation warnings

    Attributes:
        MAX_ROWS: Maximum number of rows to return (default 100).
    """

    MAX_ROWS: int = 100

    def __init__(self, max_rows: int | None = None) -> None:
        """Initialize the DataFetcher.

        Args:
            max_rows: Optional override for the maximum row limit.
        """
        self._max_rows = max_rows if max_rows is not None else self.MAX_ROWS

    def _dataframe_to_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert a DataFrame to a list of record dictionaries.

        Handles conversion of numpy/pandas types to Python native types
        for JSON serialization.

        Args:
            df: The DataFrame to convert.

        Returns:
            A list of dictionaries, one per row.
        """
        # Convert DataFrame to records, handling special types
        records = df.to_dict(orient="records")

        # Convert numpy types to native Python types
        clean_records: list[dict[str, Any]] = []
        for record in records:
            clean_record: dict[str, Any] = {}
            for key, value in record.items():
                str_key = str(key)
                if pd.isna(value):
                    clean_record[str_key] = None
                elif hasattr(value, "item"):
                    # numpy scalar types
                    clean_record[str_key] = value.item()
                elif isinstance(value, pd.Timestamp):
                    clean_record[str_key] = str(value)
                else:
                    clean_record[str_key] = value
            clean_records.append(clean_record)

        return clean_records

    def fetch_play_by_play(
        self,
        seasons: list[int],
        weeks: list[int] | None = None,
    ) -> SuccessResponse | ErrorResponse:
        """Fetch play-by-play data for specified seasons and weeks.

        Args:
            seasons: List of seasons to fetch (e.g., [2023, 2024]).
            weeks: Optional list of weeks to filter (1-18).

        Returns:
            SuccessResponse with play data, or ErrorResponse on failure.
        """
        try:
            logger.info(
                f"Fetching play-by-play data for seasons={seasons}, weeks={weeks}"
            )

            # Fetch the data from nfl_data_py
            df = nfl.import_pbp_data(seasons)

            if df is None or df.empty:
                return create_success_response(
                    data=[],
                    total_available=0,
                    truncated=False,
                    warning="No play-by-play data available for the specified seasons.",
                )

            # Filter by weeks if specified
            if weeks is not None and "week" in df.columns:
                df = df[df["week"].isin(weeks)]

            total_available = len(df)

            # Apply row limit
            truncated = total_available > self._max_rows
            if truncated:
                df = df.head(self._max_rows)

            # Convert to records
            records = self._dataframe_to_records(df)

            # Get column names (convert to strings for type safety)
            columns = [str(col) for col in df.columns] if not df.empty else []

            return create_success_response(
                data=records,
                total_available=total_available,
                truncated=truncated,
                columns=columns,
            )

        except Exception as e:
            logger.error(f"Error fetching play-by-play data: {e}")
            return create_error_response(
                error=f"Failed to fetch play-by-play data: {str(e)}",
                warning="This may be a temporary network issue. Please try again.",
            )
