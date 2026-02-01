"""Tests for the SchemaManager.

This module tests the schema preloading and caching functionality
using mocked nfl_data_py calls to avoid network dependencies.
"""

import json
import logging
import time
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import fast_nfl_mcp.constants as constants_module
from fast_nfl_mcp.constants import MIN_SEASON, get_current_season_year
from fast_nfl_mcp.models import ColumnSchema, DatasetSchema
from fast_nfl_mcp.schema_manager import (
    DATASET_DEFINITIONS,
    SchemaManager,
    _extract_column_schema,
    _extract_sample_values,
)


class TestExtractSampleValues:
    """Tests for the _extract_sample_values helper function."""

    def test_extract_from_string_series(self) -> None:
        """Test extracting samples from a string Series."""
        series = pd.Series(["a", "b", "c", "d", "e", "f"])
        samples = _extract_sample_values(series, max_samples=3)

        assert len(samples) == 3
        assert all(isinstance(s, str) for s in samples)

    def test_extract_from_int_series(self) -> None:
        """Test extracting samples from an integer Series."""
        series = pd.Series([1, 2, 3, 4, 5])
        samples = _extract_sample_values(series, max_samples=3)

        assert len(samples) == 3
        assert all(isinstance(s, int) for s in samples)

    def test_extract_from_float_series(self) -> None:
        """Test extracting samples from a float Series."""
        series = pd.Series([1.1, 2.2, 3.3, 4.4])
        samples = _extract_sample_values(series, max_samples=2)

        assert len(samples) == 2
        assert all(isinstance(s, float) for s in samples)

    def test_extract_handles_nan_values(self) -> None:
        """Test that NaN values are excluded from samples."""
        series = pd.Series([np.nan, 1, np.nan, 2, 3])
        samples = _extract_sample_values(series)

        assert np.nan not in samples
        assert None not in samples
        assert len(samples) <= 3  # Only non-null unique values

    def test_extract_from_empty_series(self) -> None:
        """Test extracting from an empty Series."""
        series = pd.Series([], dtype=object)
        samples = _extract_sample_values(series)

        assert samples == []

    def test_extract_from_all_nan_series(self) -> None:
        """Test extracting from a Series with only NaN values."""
        series = pd.Series([np.nan, np.nan, np.nan])
        samples = _extract_sample_values(series)

        assert samples == []

    def test_extract_respects_max_samples(self) -> None:
        """Test that max_samples limit is respected."""
        series = pd.Series(list(range(100)))
        samples = _extract_sample_values(series, max_samples=5)

        assert len(samples) == 5

    def test_extract_converts_numpy_types(self) -> None:
        """Test that numpy types are converted to Python natives."""
        series = pd.Series(np.array([1, 2, 3], dtype=np.int64))
        samples = _extract_sample_values(series)

        # Should be Python int, not numpy.int64
        assert all(isinstance(s, int) for s in samples)

    def test_extract_handles_timestamps(self) -> None:
        """Test that timestamps are converted to strings."""
        series = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        samples = _extract_sample_values(series)

        assert all(isinstance(s, str) for s in samples)


class TestExtractColumnSchema:
    """Tests for the _extract_column_schema helper function."""

    def test_extract_basic_column(self) -> None:
        """Test extracting schema from a basic column."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        schema = _extract_column_schema(df, "col1")

        assert schema.name == "col1"
        assert schema.dtype == "int64"
        assert len(schema.sample_values) <= 2
        assert schema.null_count == 0
        assert schema.unique_count == 5

    def test_extract_column_with_nulls(self) -> None:
        """Test extracting schema from a column with null values."""
        df = pd.DataFrame({"col1": [1.0, np.nan, 3.0, np.nan, 5.0]})
        schema = _extract_column_schema(df, "col1")

        assert schema.null_count == 2
        assert schema.unique_count == 3  # 1.0, 3.0, 5.0

    def test_extract_string_column(self) -> None:
        """Test extracting schema from a string column."""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Alice", "Charlie"]})
        schema = _extract_column_schema(df, "name")

        assert schema.name == "name"
        # pandas 2.x may use 'str' or 'object' for string columns
        assert schema.dtype in ("object", "str", "string")
        assert schema.unique_count == 3  # Alice, Bob, Charlie
        assert schema.null_count == 0

    def test_returns_column_schema_instance(self) -> None:
        """Test that the function returns a ColumnSchema instance."""
        df = pd.DataFrame({"test": [1, 2, 3]})
        schema = _extract_column_schema(df, "test")

        assert isinstance(schema, ColumnSchema)


class TestSchemaManager:
    """Tests for the SchemaManager class."""

    def test_init_creates_empty_caches(self) -> None:
        """Test that initialization creates empty caches."""
        manager = SchemaManager()

        assert manager._schemas == {}
        assert manager._failed_datasets == set()

    def test_get_dataset_names_returns_all_datasets(self) -> None:
        """Test that get_dataset_names returns all known datasets."""
        manager = SchemaManager()
        names = manager.get_dataset_names()

        assert len(names) == 20
        assert "play_by_play" in names
        assert "weekly_stats" in names
        assert "rosters" in names

    def test_list_datasets_returns_all_with_not_loaded_status(self) -> None:
        """Test list_datasets before preloading."""
        manager = SchemaManager()
        datasets = manager.list_datasets()

        assert len(datasets) == 20
        assert all(d["status"] == "not_loaded" for d in datasets)
        assert all("name" in d for d in datasets)
        assert all("description" in d for d in datasets)
        assert all("supports_seasons" in d for d in datasets)

    def test_get_schema_returns_none_when_not_loaded(self) -> None:
        """Test get_schema returns None for unloaded datasets."""
        manager = SchemaManager()

        assert manager.get_schema("play_by_play") is None
        assert manager.get_schema("nonexistent") is None

    def test_is_loaded_returns_false_before_preload(self) -> None:
        """Test is_loaded returns False before preloading."""
        manager = SchemaManager()

        assert manager.is_loaded("play_by_play") is False

    def test_get_loaded_count_is_zero_before_preload(self) -> None:
        """Test get_loaded_count returns 0 before preloading."""
        manager = SchemaManager()

        assert manager.get_loaded_count() == 0


class TestSchemaManagerWithMocks:
    """Tests for SchemaManager with mocked nfl_data_py calls."""

    @pytest.fixture
    def mock_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "game_id": ["2024_01_KC_DET", "2024_01_SF_PIT"],
                "play_id": [1, 2],
                "epa": [0.5, -0.3],
                "player_name": ["Patrick Mahomes", "Brock Purdy"],
            }
        )

    @pytest.fixture
    def mock_nfl_module(self, mock_dataframe: pd.DataFrame) -> MagicMock:
        """Create a mock nfl_data_py module."""
        mock = MagicMock()
        mock.import_pbp_data.return_value = mock_dataframe
        mock.import_weekly_data.return_value = mock_dataframe
        mock.import_seasonal_data.return_value = mock_dataframe
        mock.import_rosters.return_value = mock_dataframe
        mock.import_ids.return_value = mock_dataframe
        mock.import_draft_picks.return_value = mock_dataframe
        mock.import_schedules.return_value = mock_dataframe
        mock.import_team_desc.return_value = mock_dataframe
        mock.import_combine_data.return_value = mock_dataframe
        mock.import_sc_lines.return_value = mock_dataframe
        mock.import_win_totals.return_value = mock_dataframe
        mock.import_ngs_data.return_value = mock_dataframe
        mock.import_snap_counts.return_value = mock_dataframe
        mock.import_injuries.return_value = mock_dataframe
        mock.import_depth_charts.return_value = mock_dataframe
        mock.import_contracts.return_value = mock_dataframe
        mock.import_officials.return_value = mock_dataframe
        mock.import_qbr.return_value = mock_dataframe
        return mock

    def test_preload_all_loads_all_datasets(
        self, mock_nfl_module: MagicMock, mock_dataframe: pd.DataFrame
    ) -> None:
        """Test that preload_all loads all datasets successfully."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: mock_dataframe,
                    "Test description",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            assert manager.get_loaded_count() == 1
            assert manager.is_loaded("test_dataset")

    def test_preload_caches_schema(
        self, mock_nfl_module: MagicMock, mock_dataframe: pd.DataFrame
    ) -> None:
        """Test that loaded schemas are cached."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: mock_dataframe,
                    "Test description",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("test_dataset")
            assert schema is not None
            assert isinstance(schema, DatasetSchema)
            assert schema.name == "test_dataset"
            assert schema.description == "Test description"

    def test_preload_extracts_columns(
        self, mock_nfl_module: MagicMock, mock_dataframe: pd.DataFrame
    ) -> None:
        """Test that column schemas are correctly extracted."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: mock_dataframe,
                    "Test description",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("test_dataset")
            assert schema is not None
            assert len(schema.columns) == 4

            column_names = [c.name for c in schema.columns]
            assert "game_id" in column_names
            assert "play_id" in column_names
            assert "epa" in column_names
            assert "player_name" in column_names

    def test_preload_handles_failed_datasets(self) -> None:
        """Test that failed datasets are tracked when OSError occurs."""

        def raise_error(_: object) -> pd.DataFrame:
            raise OSError("Network error")

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "failing_dataset": (
                    raise_error,
                    "Will fail",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            assert "failing_dataset" in manager.get_failed_datasets()
            assert manager.get_schema("failing_dataset") is None
            assert not manager.is_loaded("failing_dataset")

    def test_preload_handles_connection_error(self) -> None:
        """Test that ConnectionError is handled during preload."""

        def raise_connection_error(_: object) -> pd.DataFrame:
            raise ConnectionError("Unable to connect")

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "failing_dataset": (
                    raise_connection_error,
                    "Will fail",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            assert "failing_dataset" in manager.get_failed_datasets()

    def test_preload_handles_runtime_error(self) -> None:
        """Test that RuntimeError is handled during preload."""

        def raise_runtime_error(_: object) -> pd.DataFrame:
            raise RuntimeError("Something went wrong")

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "failing_dataset": (
                    raise_runtime_error,
                    "Will fail",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            assert "failing_dataset" in manager.get_failed_datasets()

    def test_preload_keyboard_interrupt_propagates(self) -> None:
        """Test that KeyboardInterrupt is not caught and propagates correctly."""

        def raise_keyboard_interrupt(_: object) -> pd.DataFrame:
            raise KeyboardInterrupt()

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "interrupt_dataset": (
                    raise_keyboard_interrupt,
                    "Will interrupt",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            with pytest.raises(KeyboardInterrupt):
                manager.preload_all()

    def test_preload_handles_empty_dataframe(self) -> None:
        """Test that empty DataFrames are handled gracefully."""
        empty_df = pd.DataFrame()

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "empty_dataset": (
                    lambda _: empty_df,
                    "Empty dataset",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("empty_dataset")
            assert schema is not None
            assert schema.columns == []
            assert schema.row_count == 0

    def test_list_datasets_shows_loaded_status(
        self, mock_dataframe: pd.DataFrame
    ) -> None:
        """Test that list_datasets shows correct status after loading."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "loaded_dataset": (
                    lambda _: mock_dataframe,
                    "Will load",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            datasets = manager.list_datasets()
            assert len(datasets) == 1
            assert datasets[0]["status"] == "loaded"

    def test_list_datasets_shows_failed_status(self) -> None:
        """Test that list_datasets shows failed status for failed datasets."""

        def raise_error(_: object) -> pd.DataFrame:
            raise Exception("Network error")

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "failing_dataset": (
                    raise_error,
                    "Will fail",
                    False,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            datasets = manager.list_datasets()
            assert len(datasets) == 1
            assert datasets[0]["status"] == "failed"

    def test_schema_includes_available_seasons(
        self, mock_dataframe: pd.DataFrame
    ) -> None:
        """Test that seasonal datasets include available_seasons."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "seasonal_dataset": (
                    lambda _: mock_dataframe,
                    "Seasonal data",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("seasonal_dataset")
            assert schema is not None
            assert schema.available_seasons is not None
            # The current season should always be included
            current_season = get_current_season_year()
            assert current_season in schema.available_seasons
            # MIN_SEASON should always be the start
            assert MIN_SEASON in schema.available_seasons

    def test_non_seasonal_dataset_has_none_seasons(
        self, mock_dataframe: pd.DataFrame
    ) -> None:
        """Test that non-seasonal datasets have None for available_seasons."""
        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "non_seasonal": (
                    lambda _: mock_dataframe,
                    "Non-seasonal data",
                    False,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("non_seasonal")
            assert schema is not None
            assert schema.available_seasons is None


class TestDatasetDefinitions:
    """Tests for the DATASET_DEFINITIONS constant."""

    def test_has_20_datasets(self) -> None:
        """Test that all 20 expected datasets are defined."""
        assert len(DATASET_DEFINITIONS) == 20

    def test_all_required_datasets_present(self) -> None:
        """Test that all datasets from the design doc are present."""
        required_datasets = [
            "play_by_play",
            "weekly_stats",
            "seasonal_stats",
            "rosters",
            "player_ids",
            "draft_picks",
            "schedules",
            "team_descriptions",
            "combine_data",
            "scoring_lines",
            "win_totals",
            "ngs_passing",
            "ngs_rushing",
            "ngs_receiving",
            "snap_counts",
            "injuries",
            "depth_charts",
            "contracts",
            "officials",
            "qbr",
        ]

        for dataset in required_datasets:
            assert dataset in DATASET_DEFINITIONS, f"Missing dataset: {dataset}"

    def test_definitions_have_correct_structure(self) -> None:
        """Test that all definitions have the correct tuple structure."""
        for name, definition in DATASET_DEFINITIONS.items():
            assert len(definition) == 3, f"Invalid definition for {name}"
            loader, description, supports_seasons = definition

            assert callable(loader), f"Loader for {name} is not callable"
            assert isinstance(description, str), f"Description for {name} is not str"
            assert isinstance(
                supports_seasons, bool
            ), f"supports_seasons for {name} is not bool"

    def test_non_seasonal_datasets(self) -> None:
        """Test that non-seasonal datasets are correctly marked."""
        non_seasonal = ["player_ids", "team_descriptions", "contracts", "officials"]

        for dataset in non_seasonal:
            _, _, supports_seasons = DATASET_DEFINITIONS[dataset]
            assert not supports_seasons, f"{dataset} should be non-seasonal"


class TestSchemaManagerSerialization:
    """Tests for schema serialization capabilities."""

    def test_schema_to_dict_works(self) -> None:
        """Test that DatasetSchema.to_dict works for cached schemas."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test": (
                    lambda _: df,
                    "Test",
                    False,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("test")
            assert schema is not None

            schema_dict = schema.to_dict()
            assert schema_dict["name"] == "test"
            assert len(schema_dict["columns"]) == 2

    def test_column_sample_values_are_json_serializable(self) -> None:
        """Test that sample values can be JSON serialized."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
            }
        )

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test": (
                    lambda _: df,
                    "Test",
                    False,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("test")
            assert schema is not None

            # Should not raise
            json_str = json.dumps(schema.to_dict())
            assert isinstance(json_str, str)


class TestGetCurrentSeasonYear:
    """Tests for the get_current_season_year() helper function."""

    def test_january_returns_previous_year(self) -> None:
        """Test that January returns the previous year's season."""
        with patch("fast_nfl_mcp.constants.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 15)
            original_date = constants_module.date
            try:
                constants_module.date = mock_date
                result = get_current_season_year()
            finally:
                constants_module.date = original_date

            assert result == 2025

    def test_february_returns_previous_year(self) -> None:
        """Test that February (Super Bowl time) returns previous year's season."""
        with patch("fast_nfl_mcp.constants.date") as mock_date:
            mock_date.today.return_value = date(2026, 2, 12)
            original_date = constants_module.date
            try:
                constants_module.date = mock_date
                result = get_current_season_year()
            finally:
                constants_module.date = original_date

            assert result == 2025

    def test_august_returns_previous_year(self) -> None:
        """Test that August (pre-season) returns previous year's season."""
        with patch("fast_nfl_mcp.constants.date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 31)
            original_date = constants_module.date
            try:
                constants_module.date = mock_date
                result = get_current_season_year()
            finally:
                constants_module.date = original_date

            assert result == 2025

    def test_september_returns_current_year(self) -> None:
        """Test that September (season start) returns current year's season."""
        with patch("fast_nfl_mcp.constants.date") as mock_date:
            mock_date.today.return_value = date(2026, 9, 1)
            original_date = constants_module.date
            try:
                constants_module.date = mock_date
                result = get_current_season_year()
            finally:
                constants_module.date = original_date

            assert result == 2026

    def test_december_returns_current_year(self) -> None:
        """Test that December (mid-season) returns current year's season."""
        with patch("fast_nfl_mcp.constants.date") as mock_date:
            mock_date.today.return_value = date(2026, 12, 15)
            original_date = constants_module.date
            try:
                constants_module.date = mock_date
                result = get_current_season_year()
            finally:
                constants_module.date = original_date

            assert result == 2026

    def test_november_returns_current_year(self) -> None:
        """Test that November returns current year's season."""
        with patch("fast_nfl_mcp.constants.date") as mock_date:
            mock_date.today.return_value = date(2026, 11, 20)
            original_date = constants_module.date
            try:
                constants_module.date = mock_date
                result = get_current_season_year()
            finally:
                constants_module.date = original_date

            assert result == 2026

    def test_function_returns_integer(self) -> None:
        """Test that the function returns an integer."""
        result = get_current_season_year()
        assert isinstance(result, int)
        # Should be a reasonable year
        assert 2020 <= result <= 2100


class TestDynamicSeasonInSchemaManager:
    """Tests to verify SchemaManager uses dynamic seasons correctly."""

    def test_available_seasons_starts_at_min_season(self) -> None:
        """Test that available_seasons starts at MIN_SEASON (1999)."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_seasonal": (
                    lambda _: mock_df,
                    "Test seasonal data",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("test_seasonal")
            assert schema is not None
            assert schema.available_seasons is not None
            assert schema.available_seasons[0] == MIN_SEASON

    def test_available_seasons_ends_at_current_season(self) -> None:
        """Test that available_seasons ends at the current season."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_seasonal": (
                    lambda _: mock_df,
                    "Test seasonal data",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("test_seasonal")
            assert schema is not None
            assert schema.available_seasons is not None

            current_season = get_current_season_year()
            assert schema.available_seasons[-1] == current_season

    def test_available_seasons_is_continuous_range(self) -> None:
        """Test that available_seasons is a continuous range from MIN_SEASON to current."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_seasonal": (
                    lambda _: mock_df,
                    "Test seasonal data",
                    True,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all()

            schema = manager.get_schema("test_seasonal")
            assert schema is not None
            assert schema.available_seasons is not None

            current_season = get_current_season_year()
            expected = list(range(MIN_SEASON, current_season + 1))
            assert schema.available_seasons == expected


class TestSchemaManagerTimeoutHandling:
    """Tests for preload timeout handling behavior."""

    def test_preload_completes_without_raising_on_global_timeout(self) -> None:
        """Test that preload_all completes without raising when global timeout is reached."""

        def slow_loader(_: object) -> pd.DataFrame:
            """A loader that takes too long."""
            time.sleep(2.0)  # Takes 2 seconds
            return pd.DataFrame({"col": [1, 2, 3]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "slow_dataset": (
                    slow_loader,
                    "Takes too long",
                    False,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            # Use a very short timeout to trigger global timeout
            # This should NOT raise - just mark the dataset as failed
            manager.preload_all(max_workers=1, timeout=0.1)

            # Should have completed without raising
            assert "slow_dataset" in manager.get_failed_datasets()
            assert manager.get_schema("slow_dataset") is None
            assert not manager.is_loaded("slow_dataset")

    def test_preload_marks_pending_datasets_as_failed_on_timeout(self) -> None:
        """Test that pending datasets are added to _failed_datasets on timeout."""
        call_count = {"fast": 0, "slow": 0}

        def fast_loader(_: object) -> pd.DataFrame:
            """A loader that completes quickly."""
            call_count["fast"] += 1
            return pd.DataFrame({"col": [1, 2, 3]})

        def slow_loader(_: object) -> pd.DataFrame:
            """A loader that takes too long."""
            call_count["slow"] += 1
            time.sleep(5.0)  # Takes 5 seconds
            return pd.DataFrame({"col": [1, 2, 3]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "fast_dataset": (
                    fast_loader,
                    "Fast one",
                    False,
                ),
                "slow_dataset": (
                    slow_loader,
                    "Slow one",
                    False,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            # Use timeout that allows fast dataset but not slow one
            manager.preload_all(max_workers=2, timeout=0.5)

            # Fast dataset should have loaded
            assert manager.is_loaded("fast_dataset")

            # Slow dataset should be marked as failed
            assert "slow_dataset" in manager.get_failed_datasets()

    def test_preload_logs_timeout_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that timeout is logged appropriately."""

        def slow_loader(_: object) -> pd.DataFrame:
            """A loader that takes too long."""
            time.sleep(2.0)
            return pd.DataFrame({"col": [1, 2, 3]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "slow_dataset": (
                    slow_loader,
                    "Slow dataset",
                    False,
                ),
            },
            clear=True,
        ):
            with caplog.at_level(logging.WARNING):
                manager = SchemaManager()
                manager.preload_all(max_workers=1, timeout=0.1)

            # Should have logged warning about global timeout
            assert any(
                "Global preload timeout" in record.message
                or "Timeout loading schema" in record.message
                for record in caplog.records
            )

    def test_preload_handles_mixed_success_and_timeout(self) -> None:
        """Test preload handles a mix of successful and timed-out datasets."""

        def fast_loader(_: object) -> pd.DataFrame:
            return pd.DataFrame({"a": [1, 2]})

        def medium_loader(_: object) -> pd.DataFrame:
            time.sleep(0.05)
            return pd.DataFrame({"b": [3, 4]})

        def slow_loader(_: object) -> pd.DataFrame:
            time.sleep(5.0)
            return pd.DataFrame({"c": [5, 6]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "fast": (fast_loader, "Fast", False),
                "medium": (medium_loader, "Medium", False),
                "slow": (slow_loader, "Slow", False),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all(max_workers=3, timeout=0.3)

            # Fast and medium should succeed
            assert manager.is_loaded("fast")
            assert manager.is_loaded("medium")

            # Slow should fail
            assert "slow" in manager.get_failed_datasets()
            assert not manager.is_loaded("slow")

    def test_preload_reports_correct_counts_after_timeout(self) -> None:
        """Test that loaded and failed counts are correct after timeout."""

        def fast_loader(_: object) -> pd.DataFrame:
            return pd.DataFrame({"col": [1]})

        def slow_loader(_: object) -> pd.DataFrame:
            time.sleep(3.0)
            return pd.DataFrame({"col": [2]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "fast1": (fast_loader, "Fast 1", False),
                "fast2": (fast_loader, "Fast 2", False),
                "slow1": (slow_loader, "Slow 1", False),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all(max_workers=3, timeout=0.3)

            # Check counts
            assert manager.get_loaded_count() == 2
            assert len(manager.get_failed_datasets()) == 1

    def test_list_datasets_shows_timeout_as_failed(self) -> None:
        """Test that list_datasets shows timed-out datasets as failed."""

        def slow_loader(_: object) -> pd.DataFrame:
            time.sleep(2.0)
            return pd.DataFrame({"col": [1]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "timeout_dataset": (
                    slow_loader,
                    "Will timeout",
                    False,
                ),
            },
            clear=True,
        ):
            manager = SchemaManager()
            manager.preload_all(max_workers=1, timeout=0.1)

            datasets = manager.list_datasets()
            assert len(datasets) == 1
            assert datasets[0]["name"] == "timeout_dataset"
            assert datasets[0]["status"] == "failed"
