"""Tests for the DataFetcher class.

This module tests the DataFetcher's error handling, row limits,
and data conversion functionality using mocked data.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse


class TestDataFetcherInit:
    """Tests for DataFetcher initialization."""

    def test_default_max_rows(self) -> None:
        """Test that default MAX_ROWS is 100."""
        fetcher = DataFetcher()
        assert fetcher._max_rows == 100

    def test_custom_max_rows(self) -> None:
        """Test that custom max_rows can be set."""
        fetcher = DataFetcher(max_rows=50)
        assert fetcher._max_rows == 50

    def test_max_rows_class_attribute(self) -> None:
        """Test MAX_ROWS class attribute."""
        assert DataFetcher.MAX_ROWS == 100


class TestDataFrameToRecords:
    """Tests for the _dataframe_to_records method."""

    @pytest.fixture
    def fetcher(self) -> DataFetcher:
        """Create a DataFetcher instance."""
        return DataFetcher()

    def test_simple_dataframe(self, fetcher: DataFetcher) -> None:
        """Test conversion of a simple DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        records = fetcher._dataframe_to_records(df)

        assert len(records) == 2
        assert records[0] == {"a": 1, "b": "x"}
        assert records[1] == {"a": 2, "b": "y"}

    def test_null_values_converted(self, fetcher: DataFetcher) -> None:
        """Test that NaN/None values are converted to None."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": ["x", None, "z"]})
        records = fetcher._dataframe_to_records(df)

        assert records[1]["a"] is None
        assert records[1]["b"] is None

    def test_numpy_types_converted(self, fetcher: DataFetcher) -> None:
        """Test that numpy types are converted to Python native types."""
        df = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype=np.int64),
                "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            }
        )
        records = fetcher._dataframe_to_records(df)

        # Check that values are Python native types
        assert isinstance(records[0]["int_col"], int)
        assert isinstance(records[0]["float_col"], float)

    def test_timestamp_converted(self, fetcher: DataFetcher) -> None:
        """Test that timestamps are converted to strings."""
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        records = fetcher._dataframe_to_records(df)

        assert isinstance(records[0]["date"], str)
        assert "2024-01-01" in records[0]["date"]

    def test_empty_dataframe(self, fetcher: DataFetcher) -> None:
        """Test conversion of empty DataFrame."""
        df = pd.DataFrame()
        records = fetcher._dataframe_to_records(df)
        assert records == []

    def test_mixed_types(self, fetcher: DataFetcher) -> None:
        """Test DataFrame with mixed types."""
        df = pd.DataFrame(
            {
                "int": [1],
                "float": [1.5],
                "str": ["text"],
                "bool": [True],
            }
        )
        records = fetcher._dataframe_to_records(df)

        assert records[0]["int"] == 1
        assert records[0]["float"] == 1.5
        assert records[0]["str"] == "text"
        assert records[0]["bool"] is True


class TestFetchPlayByPlay:
    """Tests for the fetch_play_by_play method."""

    @pytest.fixture
    def fetcher(self) -> DataFetcher:
        """Create a DataFetcher instance."""
        return DataFetcher()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample play-by-play DataFrame."""
        return pd.DataFrame(
            {
                "game_id": ["2024_01_KC_DET", "2024_02_SF_PIT"],
                "play_id": [1, 2],
                "week": [1, 2],
                "epa": [0.5, 1.2],
            }
        )

    def test_successful_fetch(
        self, fetcher: DataFetcher, sample_df: pd.DataFrame
    ) -> None:
        """Test successful data fetch."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_df

            result = fetcher.fetch_play_by_play([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 2
            mock_import.assert_called_once_with([2024])

    def test_week_filtering(
        self, fetcher: DataFetcher, sample_df: pd.DataFrame
    ) -> None:
        """Test filtering by weeks."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_df

            result = fetcher.fetch_play_by_play([2024], [1])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 1
            assert result.data[0]["week"] == 1

    def test_row_limit_enforced(self, fetcher: DataFetcher) -> None:
        """Test that row limit is enforced."""
        large_df = pd.DataFrame({"col": list(range(150))})

        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = large_df

            result = fetcher.fetch_play_by_play([2024])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 100
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_custom_row_limit(self) -> None:
        """Test custom row limit."""
        fetcher = DataFetcher(max_rows=10)
        large_df = pd.DataFrame({"col": list(range(50))})

        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = large_df

            result = fetcher.fetch_play_by_play([2024])

            assert len(result.data) == 10
            assert result.metadata.truncated is True

    def test_no_truncation_when_under_limit(
        self, fetcher: DataFetcher, sample_df: pd.DataFrame
    ) -> None:
        """Test no truncation when data is under limit."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_df

            result = fetcher.fetch_play_by_play([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.truncated is False

    def test_network_error_returns_error_response(self, fetcher: DataFetcher) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.side_effect = Exception("Connection timeout")

            result = fetcher.fetch_play_by_play([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Failed to fetch" in result.error
            assert "Connection timeout" in result.error

    def test_empty_dataframe_returns_success_with_warning(
        self, fetcher: DataFetcher
    ) -> None:
        """Test that empty DataFrame returns success with warning."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = pd.DataFrame()

            result = fetcher.fetch_play_by_play([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None
            assert "No play-by-play data" in result.warning

    def test_none_return_from_import(self, fetcher: DataFetcher) -> None:
        """Test handling of None return from import function."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = None

            result = fetcher.fetch_play_by_play([2024])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 0
            assert result.warning is not None

    def test_columns_in_metadata(
        self, fetcher: DataFetcher, sample_df: pd.DataFrame
    ) -> None:
        """Test that columns are included in metadata."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_df

            result = fetcher.fetch_play_by_play([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "game_id" in result.metadata.columns
            assert "epa" in result.metadata.columns

    def test_multiple_seasons(
        self, fetcher: DataFetcher, sample_df: pd.DataFrame
    ) -> None:
        """Test fetching multiple seasons."""
        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_df

            result = fetcher.fetch_play_by_play([2023, 2024])

            mock_import.assert_called_once_with([2023, 2024])
            assert isinstance(result, SuccessResponse)

    def test_weeks_without_week_column(self, fetcher: DataFetcher) -> None:
        """Test that week filtering is skipped if no week column."""
        df_no_week = pd.DataFrame({"game_id": ["a", "b"], "epa": [0.5, 1.2]})

        with patch("fast_nfl_mcp.data_fetcher.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = df_no_week

            # Should not error even though weeks are specified
            result = fetcher.fetch_play_by_play([2024], [1, 2])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 2  # All rows returned
