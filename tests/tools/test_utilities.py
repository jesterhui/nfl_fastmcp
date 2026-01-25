"""Tests for the utility MCP tools.

This module tests the list_datasets and describe_dataset tools
using mocked SchemaManager instances.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.models import ColumnSchema, DatasetSchema, ErrorResponse
from fast_nfl_mcp.schema_manager import DATASET_DEFINITIONS, SchemaManager
from fast_nfl_mcp.tools.utilities import describe_dataset_impl, list_datasets_impl


class TestListDatasets:
    """Tests for the list_datasets tool."""

    def test_returns_success_response(self) -> None:
        """Test that list_datasets returns a SuccessResponse."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        assert response.success is True
        assert response.data is not None
        assert isinstance(response.data, list)

    def test_returns_all_20_datasets(self) -> None:
        """Test that list_datasets returns all 20 defined datasets."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        assert len(response.data) == 20

    def test_datasets_sorted_alphabetically(self) -> None:
        """Test that datasets are returned in alphabetical order."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        names = [d["name"] for d in response.data]
        assert names == sorted(names)

    def test_each_dataset_has_required_fields(self) -> None:
        """Test that each dataset has name, description, supports_seasons, status."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        for dataset in response.data:
            assert "name" in dataset
            assert "description" in dataset
            assert "supports_seasons" in dataset
            assert "status" in dataset

    def test_status_not_loaded_before_preload(self) -> None:
        """Test that status is 'not_loaded' before preloading."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        for dataset in response.data:
            assert dataset["status"] == "not_loaded"

    def test_status_available_after_preload(self) -> None:
        """Test that status is 'available' after successful preload."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: mock_df,
                    "Test description",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            # Also patch in utilities module
            with patch.dict(
                "fast_nfl_mcp.tools.utilities.DATASET_DEFINITIONS",
                {
                    "test_dataset": (
                        lambda _: mock_df,
                        "Test description",
                        True,
                        2024,
                    ),
                },
                clear=True,
            ):
                manager = SchemaManager()
                manager.preload_all()
                response = list_datasets_impl(manager)

                assert len(response.data) == 1
                assert response.data[0]["status"] == "available"

    def test_status_unavailable_for_failed_datasets(self) -> None:
        """Test that status is 'unavailable' for failed datasets."""

        def raise_error(_: object) -> pd.DataFrame:
            raise Exception("Network error")

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "failing_dataset": (
                    raise_error,
                    "Will fail",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            with patch.dict(
                "fast_nfl_mcp.tools.utilities.DATASET_DEFINITIONS",
                {
                    "failing_dataset": (
                        raise_error,
                        "Will fail",
                        True,
                        2024,
                    ),
                },
                clear=True,
            ):
                manager = SchemaManager()
                manager.preload_all()
                response = list_datasets_impl(manager)

                assert response.data[0]["status"] == "unavailable"

    def test_includes_known_datasets(self) -> None:
        """Test that known datasets are included in the response."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        names = [d["name"] for d in response.data]
        assert "play_by_play" in names
        assert "weekly_stats" in names
        assert "rosters" in names
        assert "player_ids" in names

    def test_supports_seasons_matches_definition(self) -> None:
        """Test that supports_seasons matches DATASET_DEFINITIONS."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        for dataset in response.data:
            name = dataset["name"]
            _, _, expected_supports_seasons, _ = DATASET_DEFINITIONS[name]
            assert dataset["supports_seasons"] == expected_supports_seasons

    def test_metadata_includes_columns(self) -> None:
        """Test that response metadata includes column names."""
        manager = SchemaManager()
        response = list_datasets_impl(manager)

        assert response.metadata.columns is not None
        assert "name" in response.metadata.columns
        assert "description" in response.metadata.columns


class TestDescribeDataset:
    """Tests for the describe_dataset tool."""

    @pytest.fixture
    def preloaded_manager(self) -> SchemaManager:
        """Create a SchemaManager with a test dataset preloaded."""
        manager = SchemaManager()
        # Manually set up the schema
        columns = [
            ColumnSchema(
                name="game_id",
                dtype="object",
                sample_values=["2024_01_KC_DET", "2024_01_SF_PIT"],
                null_count=0,
                unique_count=2,
            ),
            ColumnSchema(
                name="play_id",
                dtype="int64",
                sample_values=[1, 2],
                null_count=0,
                unique_count=2,
            ),
            ColumnSchema(
                name="epa",
                dtype="float64",
                sample_values=[0.5, -0.3],
                null_count=0,
                unique_count=2,
            ),
            ColumnSchema(
                name="player_name",
                dtype="object",
                sample_values=["Patrick Mahomes", "Brock Purdy"],
                null_count=0,
                unique_count=2,
            ),
        ]
        schema = DatasetSchema(
            name="play_by_play",
            description="Play-by-play data with EPA, WPA, and detailed play outcomes",
            columns=columns,
            row_count=2,
            available_seasons=list(range(1999, 2025)),
        )
        manager._schemas["play_by_play"] = schema
        return manager

    def test_returns_success_for_valid_dataset(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that describe_dataset returns success for a valid dataset."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        assert response.success is True
        assert len(response.data) == 1

    def test_returns_error_for_unknown_dataset(self) -> None:
        """Test that describe_dataset returns error for unknown dataset."""
        manager = SchemaManager()
        response = describe_dataset_impl(manager, "nonexistent_dataset")

        assert response.success is False
        assert isinstance(response, ErrorResponse)
        assert "Unknown dataset" in response.error
        assert "nonexistent_dataset" in response.error

    def test_error_includes_available_datasets(self) -> None:
        """Test that error for unknown dataset lists available ones."""
        manager = SchemaManager()
        response = describe_dataset_impl(manager, "nonexistent")

        assert response.warning is not None
        assert "Available datasets" in response.warning

    def test_returns_error_for_failed_dataset(self) -> None:
        """Test that describe_dataset returns error for a failed dataset."""
        manager = SchemaManager()
        manager._failed_datasets.add("play_by_play")

        response = describe_dataset_impl(manager, "play_by_play")

        assert response.success is False
        assert isinstance(response, ErrorResponse)
        assert "failed to load" in response.error

    def test_returns_error_for_not_loaded_dataset(self) -> None:
        """Test that describe_dataset returns error if schema not loaded."""
        manager = SchemaManager()
        # Don't preload - schema will be None

        response = describe_dataset_impl(manager, "play_by_play")

        assert response.success is False
        assert isinstance(response, ErrorResponse)
        assert "not loaded" in response.error

    def test_response_includes_dataset_name(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that response includes the dataset name."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        assert response.data[0]["dataset"] == "play_by_play"

    def test_response_includes_description(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that response includes the dataset description."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        assert "description" in response.data[0]
        assert "EPA" in response.data[0]["description"]

    def test_response_includes_column_count(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that response includes column count."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        assert response.data[0]["column_count"] == 4

    def test_response_includes_available_seasons(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that response includes available seasons."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        seasons = response.data[0]["available_seasons"]
        assert seasons is not None
        assert 2024 in seasons

    def test_response_includes_columns_list(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that response includes list of columns."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        columns = response.data[0]["columns"]
        assert len(columns) == 4

        column_names = [c["name"] for c in columns]
        assert "game_id" in column_names
        assert "play_id" in column_names
        assert "epa" in column_names
        assert "player_name" in column_names

    def test_columns_include_dtype(self, preloaded_manager: SchemaManager) -> None:
        """Test that column info includes dtype."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        columns = response.data[0]["columns"]
        epa_col = next(c for c in columns if c["name"] == "epa")
        assert epa_col["dtype"] == "float64"

    def test_columns_include_sample_values(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that column info includes sample values."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        columns = response.data[0]["columns"]
        player_col = next(c for c in columns if c["name"] == "player_name")
        assert "Patrick Mahomes" in player_col["sample_values"]

    def test_columns_include_null_count(self, preloaded_manager: SchemaManager) -> None:
        """Test that column info includes null count."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        columns = response.data[0]["columns"]
        for col in columns:
            assert "null_count" in col

    def test_columns_include_unique_count(
        self, preloaded_manager: SchemaManager
    ) -> None:
        """Test that column info includes unique count."""
        response = describe_dataset_impl(preloaded_manager, "play_by_play")

        columns = response.data[0]["columns"]
        for col in columns:
            assert "unique_count" in col


class TestDescribeDatasetWithAllDatasets:
    """Test describe_dataset with each known dataset name."""

    def test_recognizes_all_defined_datasets(self) -> None:
        """Test that all defined datasets are recognized (not 'unknown')."""
        manager = SchemaManager()

        for dataset_name in DATASET_DEFINITIONS.keys():
            response = describe_dataset_impl(manager, dataset_name)
            # Should not return "Unknown dataset" error
            if not response.success:
                assert isinstance(response, ErrorResponse)
                assert "Unknown dataset" not in response.error


class TestUtilitiesIntegration:
    """Integration tests for utility tools with full preload."""

    def test_list_then_describe_workflow(self) -> None:
        """Test the typical workflow: list datasets, then describe one."""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        with patch.dict(
            "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
            {
                "test_dataset": (
                    lambda _: mock_df,
                    "A test dataset",
                    True,
                    2024,
                ),
            },
            clear=True,
        ):
            with patch.dict(
                "fast_nfl_mcp.tools.utilities.DATASET_DEFINITIONS",
                {
                    "test_dataset": (
                        lambda _: mock_df,
                        "A test dataset",
                        True,
                        2024,
                    ),
                },
                clear=True,
            ):
                manager = SchemaManager()
                manager.preload_all()

                # Step 1: List datasets
                list_response = list_datasets_impl(manager)
                assert list_response.success
                assert len(list_response.data) == 1
                dataset_name = list_response.data[0]["name"]

                # Step 2: Describe the dataset
                describe_response = describe_dataset_impl(manager, dataset_name)
                assert describe_response.success
                assert describe_response.data[0]["dataset"] == dataset_name
