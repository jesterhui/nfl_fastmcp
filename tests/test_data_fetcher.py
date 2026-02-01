"""Tests for the DataFetcher.

This module tests the data fetching functionality including error handling,
row limits, and response formatting using mocked nfl_data_py calls.
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fast_nfl_mcp.constants import DEFAULT_MAX_ROWS
from fast_nfl_mcp.data_fetcher import DataFetcher
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse


class TestDataFetcherInit:
    """Tests for DataFetcher initialization."""

    def test_default_max_rows(self) -> None:
        """Test that default MAX_ROWS matches the constant."""
        fetcher = DataFetcher()
        assert fetcher._max_rows == DEFAULT_MAX_ROWS

    def test_custom_max_rows(self) -> None:
        """Test that custom max_rows can be set."""
        fetcher = DataFetcher(max_rows=50)
        assert fetcher._max_rows == 50

    def test_max_rows_class_constant(self) -> None:
        """Test that MAX_ROWS class constant matches the constant."""
        assert DataFetcher.MAX_ROWS == DEFAULT_MAX_ROWS


class TestDataFetcherGetAvailableDatasets:
    """Tests for get_available_datasets method."""

    def test_returns_sorted_list(self) -> None:
        """Test that available datasets are returned sorted."""
        fetcher = DataFetcher()
        datasets = fetcher.get_available_datasets()

        assert isinstance(datasets, list)
        assert datasets == sorted(datasets)

    def test_contains_expected_datasets(self) -> None:
        """Test that expected datasets are in the list."""
        fetcher = DataFetcher()
        datasets = fetcher.get_available_datasets()

        expected = [
            "play_by_play",
            "weekly_stats",
            "rosters",
            "player_ids",
            "team_descriptions",
        ]
        for ds in expected:
            assert ds in datasets


class TestDataFetcherGetDatasetInfo:
    """Tests for get_dataset_info method."""

    def test_returns_info_for_valid_dataset(self) -> None:
        """Test that info is returned for a valid dataset."""
        fetcher = DataFetcher()
        info = fetcher.get_dataset_info("play_by_play")

        assert info is not None
        assert info["name"] == "play_by_play"
        assert "description" in info
        assert info["supports_seasons"] is True
        assert info["default_season"] is not None

    def test_returns_none_for_invalid_dataset(self) -> None:
        """Test that None is returned for invalid dataset."""
        fetcher = DataFetcher()
        info = fetcher.get_dataset_info("nonexistent_dataset")

        assert info is None

    def test_non_seasonal_dataset_info(self) -> None:
        """Test info for a non-seasonal dataset."""
        fetcher = DataFetcher()
        info = fetcher.get_dataset_info("team_descriptions")

        assert info is not None
        assert info["supports_seasons"] is False
        assert info["default_season"] is None


class TestDataFetcherFetchUnknownDataset:
    """Tests for fetch with unknown dataset names."""

    def test_unknown_dataset_returns_error(self) -> None:
        """Test that unknown dataset returns an ErrorResponse."""
        fetcher = DataFetcher()
        response = fetcher.fetch("nonexistent_dataset")

        assert isinstance(response, ErrorResponse)
        assert response.success is False
        assert "Unknown dataset" in response.error
        assert "nonexistent_dataset" in response.error

    def test_unknown_dataset_lists_valid_datasets(self) -> None:
        """Test that error message includes valid dataset names."""
        fetcher = DataFetcher()
        response = fetcher.fetch("bad_dataset")

        assert isinstance(response, ErrorResponse)
        assert "play_by_play" in response.error


class TestDataFetcherFetchWithMocks:
    """Tests for fetch method with mocked nfl_data_py calls."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "game_id": ["2024_01_KC_DET", "2024_01_SF_PIT", "2024_02_KC_BUF"],
                "play_id": [1, 2, 3],
                "epa": [0.5, -0.3, 1.2],
                "player_name": ["Patrick Mahomes", "Brock Purdy", "Patrick Mahomes"],
            }
        )

    @pytest.fixture
    def large_dataframe(self) -> pd.DataFrame:
        """Create a large DataFrame for testing truncation."""
        return pd.DataFrame(
            {
                "id": list(range(150)),
                "value": [f"value_{i}" for i in range(150)],
            }
        )

    def test_successful_fetch(self, sample_dataframe: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: sample_dataframe,
                    "Test description",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset")

            assert isinstance(response, SuccessResponse)
            assert response.success is True
            assert len(response.data) == 3
            assert response.metadata.row_count == 3
            assert response.metadata.truncated is False
            assert response.warning is None

    def test_fetch_with_seasons_param(self, sample_dataframe: pd.DataFrame) -> None:
        """Test fetch with custom seasons parameter."""
        loader_called_with = []

        def mock_loader(seasons: list[int] | None) -> pd.DataFrame:
            loader_called_with.append(seasons)
            return sample_dataframe

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    mock_loader,
                    "Test description",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            fetcher.fetch("test_dataset", {"seasons": [2023, 2024]})

            assert loader_called_with == [[2023, 2024]]

    def test_fetch_with_single_season_int(self, sample_dataframe: pd.DataFrame) -> None:
        """Test fetch with a single season as int."""
        loader_called_with = []

        def mock_loader(seasons: list[int] | None) -> pd.DataFrame:
            loader_called_with.append(seasons)
            return sample_dataframe

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    mock_loader,
                    "Test description",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            fetcher.fetch("test_dataset", {"seasons": 2023})

            assert loader_called_with == [[2023]]

    def test_fetch_enforces_row_limit(self, large_dataframe: pd.DataFrame) -> None:
        """Test that fetch enforces the DEFAULT_MAX_ROWS limit."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: large_dataframe,
                    "Large dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset")

            assert isinstance(response, SuccessResponse)
            assert len(response.data) == DEFAULT_MAX_ROWS
            assert response.metadata.row_count == DEFAULT_MAX_ROWS
            assert response.metadata.total_available == 150
            assert response.metadata.truncated is True
            assert response.warning is not None
            assert "truncated" in response.warning.lower()

    def test_fetch_with_custom_max_rows(self, large_dataframe: pd.DataFrame) -> None:
        """Test that custom max_rows is respected."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: large_dataframe,
                    "Large dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher(max_rows=50)
            response = fetcher.fetch("test_dataset")

            assert isinstance(response, SuccessResponse)
            assert len(response.data) == 50
            assert response.metadata.row_count == 50
            assert response.metadata.total_available == 150
            assert response.metadata.truncated is True

    def test_fetch_returns_columns_in_metadata(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test that column names are included in metadata."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: sample_dataframe,
                    "Test description",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset")

            assert isinstance(response, SuccessResponse)
            assert response.metadata.columns is not None
            assert "game_id" in response.metadata.columns
            assert "play_id" in response.metadata.columns
            assert "epa" in response.metadata.columns

    def test_fetch_non_seasonal_dataset(self, sample_dataframe: pd.DataFrame) -> None:
        """Test fetching a non-seasonal dataset."""
        loader_called_with = []

        def mock_loader(seasons: list[int] | None) -> pd.DataFrame:
            loader_called_with.append(seasons)
            return sample_dataframe

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "non_seasonal": (
                    mock_loader,
                    "Non-seasonal data",
                    False,
                    None,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("non_seasonal")

            assert isinstance(response, SuccessResponse)
            assert loader_called_with == [None]

    def test_fetch_with_offset(self, large_dataframe: pd.DataFrame) -> None:
        """Test that offset skips the specified number of rows."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: large_dataframe,
                    "Large dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset", offset=10)

            assert isinstance(response, SuccessResponse)
            assert len(response.data) == DEFAULT_MAX_ROWS
            # First row should be id=10 (skipped first 10)
            assert response.data[0]["id"] == 10
            assert response.metadata.total_available == 150

    def test_fetch_with_limit(self, large_dataframe: pd.DataFrame) -> None:
        """Test that limit overrides the default max_rows."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: large_dataframe,
                    "Large dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset", limit=5)

            assert isinstance(response, SuccessResponse)
            assert len(response.data) == 5
            assert response.metadata.row_count == 5
            assert response.metadata.total_available == 150
            assert response.metadata.truncated is True

    def test_fetch_with_offset_and_limit(self, large_dataframe: pd.DataFrame) -> None:
        """Test pagination with both offset and limit."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: large_dataframe,
                    "Large dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset", offset=20, limit=15)

            assert isinstance(response, SuccessResponse)
            assert len(response.data) == 15
            # First row should be id=20 (skipped first 20)
            assert response.data[0]["id"] == 20
            assert response.metadata.total_available == 150

    def test_fetch_offset_beyond_data(self, large_dataframe: pd.DataFrame) -> None:
        """Test that offset beyond data returns empty results."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: large_dataframe,
                    "Large dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset", offset=200)

            assert isinstance(response, SuccessResponse)
            assert len(response.data) == 0
            assert response.metadata.total_available == 150
            assert response.metadata.truncated is False

    def test_fetch_warning_includes_next_offset(
        self, large_dataframe: pd.DataFrame
    ) -> None:
        """Test that truncation warning includes next offset for pagination."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: large_dataframe,
                    "Large dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test_dataset", offset=10, limit=10)

            assert isinstance(response, SuccessResponse)
            assert response.warning is not None
            assert "offset=20" in response.warning


class TestDataFetcherErrorHandling:
    """Tests for error handling in fetch method."""

    def test_network_error_returns_error_response(self) -> None:
        """Test that network errors return ErrorResponse."""

        def raise_connection_error(_: object) -> pd.DataFrame:
            raise ConnectionError("Unable to connect to server")

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "failing_dataset": (
                    raise_connection_error,
                    "Will fail",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("failing_dataset")

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "Network error" in response.error
            assert response.data == []
            assert response.metadata.row_count == 0

    def test_timeout_error_returns_error_response(self) -> None:
        """Test that timeout errors return ErrorResponse."""

        def raise_timeout_error(_: object) -> pd.DataFrame:
            raise TimeoutError("Request timed out")

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "slow_dataset": (
                    raise_timeout_error,
                    "Will timeout",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("slow_dataset")

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "Timeout" in response.error

    def test_value_error_returns_warning_response(self) -> None:
        """Test that ValueError returns SuccessResponse with warning."""

        def raise_value_error(_: object) -> pd.DataFrame:
            raise ValueError("Invalid season: 1950")

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "dataset": (
                    raise_value_error,
                    "Invalid params",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("dataset", {"seasons": [1950]})

            assert isinstance(response, SuccessResponse)
            assert response.success is True
            assert response.data == []
            assert response.warning is not None
            assert "Invalid parameters" in response.warning

    def test_runtime_error_returns_error_response(self) -> None:
        """Test that RuntimeError returns ErrorResponse."""

        def raise_runtime_error(_: object) -> pd.DataFrame:
            raise RuntimeError("Something went wrong")

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "broken_dataset": (
                    raise_runtime_error,
                    "Will break",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("broken_dataset")

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "Error fetching" in response.error

    def test_os_error_returns_error_response(self) -> None:
        """Test that OSError returns ErrorResponse with system error message."""

        def raise_os_error(_: object) -> pd.DataFrame:
            raise OSError("Disk I/O error")

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "broken_dataset": (
                    raise_os_error,
                    "Will break",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("broken_dataset")

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "System error" in response.error

    def test_keyboard_interrupt_propagates(self) -> None:
        """Test that KeyboardInterrupt is not caught and propagates correctly."""

        def raise_keyboard_interrupt(_: object) -> pd.DataFrame:
            raise KeyboardInterrupt()

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "interrupt_dataset": (
                    raise_keyboard_interrupt,
                    "Will interrupt",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            with pytest.raises(KeyboardInterrupt):
                fetcher.fetch("interrupt_dataset")

    def test_system_exit_propagates(self) -> None:
        """Test that SystemExit is not caught and propagates correctly."""

        def raise_system_exit(_: object) -> pd.DataFrame:
            raise SystemExit(1)

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "exit_dataset": (
                    raise_system_exit,
                    "Will exit",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            with pytest.raises(SystemExit):
                fetcher.fetch("exit_dataset")

    def test_attribute_error_returns_error_response(self) -> None:
        """Test that AttributeError returns ErrorResponse (not crash)."""

        def raise_attribute_error(_: object) -> pd.DataFrame:
            raise AttributeError("object has no attribute 'foo'")

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "buggy_dataset": (
                    raise_attribute_error,
                    "Will fail",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("buggy_dataset")

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "Error fetching" in response.error

    def test_type_error_returns_error_response(self) -> None:
        """Test that TypeError returns ErrorResponse (not crash)."""

        def raise_type_error(_: object) -> pd.DataFrame:
            raise TypeError("unsupported operand type")

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "buggy_dataset": (
                    raise_type_error,
                    "Will fail",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("buggy_dataset")

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "Error fetching" in response.error

    def test_empty_dataframe_returns_warning(self) -> None:
        """Test that empty DataFrame returns SuccessResponse with warning."""
        empty_df = pd.DataFrame()

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "empty_dataset": (
                    lambda _: empty_df,
                    "Empty data",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("empty_dataset")

            assert isinstance(response, SuccessResponse)
            assert response.success is True
            assert response.data == []
            assert response.warning is not None
            assert "No data found" in response.warning

    def test_none_dataframe_returns_warning(self) -> None:
        """Test that None DataFrame returns SuccessResponse with warning."""
        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "null_dataset": (
                    lambda _: None,
                    "Returns None",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("null_dataset")

            assert isinstance(response, SuccessResponse)
            assert response.success is True
            assert response.data == []
            assert response.warning is not None


class TestDataFetcherDataConversion:
    """Tests for DataFrame to dict conversion."""

    def test_nan_values_converted_to_none(self) -> None:
        """Test that NaN values are converted to None."""
        df = pd.DataFrame(
            {
                "col1": [1, np.nan, 3],
                "col2": ["a", "b", np.nan],
            }
        )

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test")

            assert isinstance(response, SuccessResponse)
            # Check that NaN was converted to None
            assert response.data[1]["col1"] is None
            assert response.data[2]["col2"] is None

    def test_numpy_types_converted_to_native(self) -> None:
        """Test that numpy types are converted to Python native types."""
        df = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype=np.int64),
                "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64),
            }
        )

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test")

            assert isinstance(response, SuccessResponse)
            # Values should be native Python types
            assert isinstance(response.data[0]["int_col"], int)
            assert isinstance(response.data[0]["float_col"], float)

    def test_timestamps_converted_to_strings(self) -> None:
        """Test that timestamps are converted to strings."""
        df = pd.DataFrame(
            {
                "date_col": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test")

            assert isinstance(response, SuccessResponse)
            assert isinstance(response.data[0]["date_col"], str)

    def test_response_is_json_serializable(self) -> None:
        """Test that the response can be JSON serialized."""
        df = pd.DataFrame(
            {
                "int_col": np.array([1, 2], dtype=np.int64),
                "float_col": np.array([1.1, np.nan], dtype=np.float64),
                "str_col": ["a", "b"],
                "date_col": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test")

            # Should not raise
            json_str = json.dumps(response.model_dump())
            assert isinstance(json_str, str)


class TestDataFetcherParamsHandling:
    """Tests for parameter handling in fetch."""

    def test_fetch_with_none_params(self) -> None:
        """Test fetch with None params uses defaults."""
        loader_called_with = []

        def mock_loader(seasons: list[int] | None) -> pd.DataFrame:
            loader_called_with.append(seasons)
            return pd.DataFrame({"col": [1]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (mock_loader, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            fetcher.fetch("test", None)

            # Should use default season
            assert loader_called_with == [[2024]]

    def test_fetch_with_empty_params(self) -> None:
        """Test fetch with empty params dict uses defaults."""
        loader_called_with = []

        def mock_loader(seasons: list[int] | None) -> pd.DataFrame:
            loader_called_with.append(seasons)
            return pd.DataFrame({"col": [1]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (mock_loader, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            fetcher.fetch("test", {})

            # Should use default season
            assert loader_called_with == [[2024]]

    def test_fetch_ignores_unknown_params(self) -> None:
        """Test that unknown params are ignored."""
        df = pd.DataFrame({"col": [1]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            # Should not raise
            response = fetcher.fetch("test", {"unknown_param": "value"})

            assert isinstance(response, SuccessResponse)


class TestDataFetcherColumnSelection:
    """Tests for column selection in fetch method."""

    def test_empty_columns_list_returns_error(self) -> None:
        """Test that an empty columns list returns an error."""
        df = pd.DataFrame({"col1": [1], "col2": [2]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test", columns=[])

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "Empty columns list" in response.error

    def test_all_invalid_columns_returns_error(self) -> None:
        """Test that all invalid column names returns an error."""
        df = pd.DataFrame({"col1": [1], "col2": [2]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test", columns=["invalid1", "invalid2"])

            assert isinstance(response, ErrorResponse)
            assert response.success is False
            assert "None of the requested columns exist" in response.error
            assert "invalid1" in response.error
            assert "invalid2" in response.error

    def test_valid_columns_selected(self) -> None:
        """Test that valid columns are properly selected."""
        df = pd.DataFrame({"col1": [1], "col2": [2], "col3": [3]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test", columns=["col1", "col3"])

            assert isinstance(response, SuccessResponse)
            assert response.metadata.columns == ["col1", "col3"]
            assert "col2" not in response.data[0]

    def test_mixed_valid_invalid_columns_returns_valid_only(self) -> None:
        """Test that mixed valid/invalid columns returns only valid ones."""
        df = pd.DataFrame({"col1": [1], "col2": [2]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test", columns=["col1", "invalid"])

            assert isinstance(response, SuccessResponse)
            assert response.metadata.columns == ["col1"]
            assert "col2" not in response.data[0]

    def test_columns_none_returns_all_columns(self) -> None:
        """Test that columns=None returns all columns."""
        df = pd.DataFrame({"col1": [1], "col2": [2]})

        with patch.dict(
            "fast_nfl_mcp.data_fetcher.DATASET_DEFINITIONS",
            {
                "test": (lambda _: df, "Test", True, 2024),
            },
            clear=True,
        ):
            fetcher = DataFetcher()
            response = fetcher.fetch("test", columns=None)

            assert isinstance(response, SuccessResponse)
            assert set(response.metadata.columns) == {"col1", "col2"}
