"""Tests for the play-by-play data retrieval tool.

This module tests the get_play_by_play tool including parameter validation,
data fetching, and error handling using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.constants import (
    DEFAULT_MAX_ROWS,
    MAX_SEASONS,
    MAX_WEEK,
    MIN_SEASON,
    MIN_WEEK,
)
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.play_by_play import (
    get_play_by_play_impl,
    normalize_filters,
    validate_seasons,
    validate_weeks,
)


class TestValidateSeasons:
    """Tests for the validate_seasons function."""

    def test_valid_seasons(self) -> None:
        """Test that valid seasons pass without warning."""
        valid, warning = validate_seasons([2023, 2024])
        assert valid == [2023, 2024]
        assert warning is None

    def test_empty_seasons(self) -> None:
        """Test that empty seasons list returns warning."""
        valid, warning = validate_seasons([])
        assert valid == []
        assert warning is not None
        assert "No seasons provided" in warning

    def test_too_many_seasons(self) -> None:
        """Test that exceeding MAX_SEASONS triggers truncation."""
        seasons = [2020, 2021, 2022, 2023, 2024]
        valid, warning = validate_seasons(seasons)
        assert len(valid) == MAX_SEASONS
        assert valid == seasons[:MAX_SEASONS]
        assert warning is not None
        assert "Too many seasons" in warning

    def test_exactly_max_seasons(self) -> None:
        """Test that exactly MAX_SEASONS passes without warning."""
        seasons = [2022, 2023, 2024]
        valid, warning = validate_seasons(seasons)
        assert valid == seasons
        assert warning is None

    def test_invalid_old_season(self) -> None:
        """Test that seasons before MIN_SEASON are filtered out."""
        valid, warning = validate_seasons([1990, 2023])
        assert valid == [2023]
        assert warning is not None
        assert "Invalid seasons removed" in warning
        assert "1990" in warning

    def test_all_invalid_seasons(self) -> None:
        """Test that all invalid seasons returns empty list."""
        valid, warning = validate_seasons([1990, 1995])
        assert valid == []
        assert warning is not None

    def test_single_valid_season(self) -> None:
        """Test single valid season."""
        valid, warning = validate_seasons([2024])
        assert valid == [2024]
        assert warning is None

    def test_min_valid_season(self) -> None:
        """Test minimum valid season."""
        valid, warning = validate_seasons([MIN_SEASON])
        assert valid == [MIN_SEASON]
        assert warning is None


class TestNormalizeFilters:
    """Tests for the normalize_filters function."""

    def test_none_filters(self) -> None:
        """Test that None filters returns empty dict."""
        result = normalize_filters(None)
        assert result == {}

    def test_empty_filters(self) -> None:
        """Test that empty filters returns empty dict."""
        result = normalize_filters({})
        assert result == {}

    def test_single_value_normalized_to_list(self) -> None:
        """Test that single values are normalized to lists."""
        result = normalize_filters({"team": "TB"})
        assert result == {"team": ["TB"]}

    def test_list_value_preserved(self) -> None:
        """Test that list values are preserved."""
        result = normalize_filters({"team": ["TB", "KC"]})
        assert result == {"team": ["TB", "KC"]}

    def test_mixed_values(self) -> None:
        """Test mixed single and list values."""
        result = normalize_filters({"team": "TB", "play_type": ["pass", "run"]})
        assert result == {"team": ["TB"], "play_type": ["pass", "run"]}

    def test_integer_values(self) -> None:
        """Test integer values are normalized."""
        result = normalize_filters({"down": 3})
        assert result == {"down": [3]}


class TestValidateWeeks:
    """Tests for the validate_weeks function."""

    def test_none_weeks(self) -> None:
        """Test that None weeks returns None without warning."""
        valid, warning = validate_weeks(None)
        assert valid is None
        assert warning is None

    def test_empty_weeks(self) -> None:
        """Test that empty weeks list returns None without warning."""
        valid, warning = validate_weeks([])
        assert valid is None
        assert warning is None

    def test_valid_weeks(self) -> None:
        """Test valid weeks pass without warning."""
        valid, warning = validate_weeks([1, 2, 3])
        assert valid == [1, 2, 3]
        assert warning is None

    def test_all_weeks(self) -> None:
        """Test all regular season weeks."""
        weeks = list(range(MIN_WEEK, MAX_WEEK + 1))
        valid, warning = validate_weeks(weeks)
        assert valid == weeks
        assert warning is None

    def test_invalid_week_zero(self) -> None:
        """Test that week 0 is filtered out."""
        valid, warning = validate_weeks([0, 1, 2])
        assert valid == [1, 2]
        assert warning is not None
        assert "Invalid weeks removed" in warning

    def test_invalid_week_too_high(self) -> None:
        """Test that weeks > MAX_WEEK are filtered out."""
        valid, warning = validate_weeks([17, 18, 19, 20])
        assert valid == [17, 18]
        assert warning is not None
        assert "19" in warning
        assert "20" in warning

    def test_all_invalid_weeks(self) -> None:
        """Test that all invalid weeks returns None."""
        valid, warning = validate_weeks([0, 19, 20])
        assert valid is None
        assert warning is not None

    def test_single_valid_week(self) -> None:
        """Test single valid week."""
        valid, warning = validate_weeks([10])
        assert valid == [10]
        assert warning is None

    def test_boundary_weeks(self) -> None:
        """Test boundary week values."""
        valid, warning = validate_weeks([MIN_WEEK, MAX_WEEK])
        assert valid == [MIN_WEEK, MAX_WEEK]
        assert warning is None


class TestGetPlayByPlayImpl:
    """Tests for the get_play_by_play_impl function."""

    @pytest.fixture
    def sample_pbp_df(self) -> pd.DataFrame:
        """Create a sample play-by-play DataFrame."""
        return pd.DataFrame(
            {
                "game_id": ["2024_01_KC_DET", "2024_01_KC_DET", "2024_02_SF_PIT"],
                "play_id": [1, 2, 3],
                "week": [1, 1, 2],
                "posteam": ["KC", "KC", "SF"],
                "defteam": ["DET", "DET", "PIT"],
                "yards_gained": [5, -2, 15],
                "epa": [0.5, -0.8, 1.2],
                "wpa": [0.02, -0.03, 0.05],
                "play_type": ["pass", "run", "pass"],
            }
        )

    @pytest.fixture
    def large_pbp_df(self) -> pd.DataFrame:
        """Create a large play-by-play DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "game_id": [f"2024_{i:02d}_KC_DET" for i in range(150)],
                "play_id": list(range(150)),
                "week": [1] * 150,
                "yards_gained": [5] * 150,
                "epa": [0.5] * 150,
            }
        )

    def test_successful_fetch(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_week_filtering(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test that week filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024], [1])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only week 1 plays should be returned
            assert len(result.data) == 2
            for row in result.data:
                assert row["week"] == 1

    def test_truncation_at_max_rows(self, large_pbp_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = large_pbp_df

            result = get_play_by_play_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_play_by_play_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_invalid_seasons_warning(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test that invalid seasons produce warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([1990, 2024])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid seasons removed" in result.warning

    def test_too_many_seasons_warning(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test that exceeding MAX_SEASONS produces warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2020, 2021, 2022, 2023, 2024])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Too many seasons" in result.warning

    def test_invalid_weeks_warning(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test that invalid weeks produce warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024], [0, 1, 20])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid weeks removed" in result.warning

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_play_by_play_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_empty_dataframe_returns_success(self) -> None:
        """Test that empty DataFrame returns success with warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = pd.DataFrame()

            result = get_play_by_play_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None

    def test_columns_in_metadata(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "game_id" in result.metadata.columns
            assert "epa" in result.metadata.columns

    def test_data_types_converted(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test that numpy types are converted to Python native types."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024])

            assert isinstance(result, SuccessResponse)
            # Check that values are Python native types, not numpy
            for row in result.data:
                assert isinstance(row["play_id"], int)
                assert isinstance(row["yards_gained"], int)
                assert isinstance(row["epa"], float)

    def test_combined_warnings(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test that multiple warnings are combined."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            # Both invalid season and invalid week
            result = get_play_by_play_impl([1990, 2024], [0, 1])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid seasons removed" in result.warning
            assert "Invalid weeks removed" in result.warning

    def test_filter_by_single_team(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test filtering by a single team value."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024], filters={"posteam": "KC"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only KC plays should be returned
            assert len(result.data) == 2
            for row in result.data:
                assert row["posteam"] == "KC"

    def test_filter_by_team_list(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test filtering by a list of teams."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024], filters={"posteam": ["KC", "SF"]})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            for row in result.data:
                assert row["posteam"] in ["KC", "SF"]

    def test_filter_by_play_type(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test filtering by play type."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024], filters={"play_type": "pass"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 2
            for row in result.data:
                assert row["play_type"] == "pass"

    def test_combined_filters(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test multiple filters applied together."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl(
                [2024], filters={"posteam": "KC", "play_type": "pass"}
            )

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only KC pass plays
            assert len(result.data) == 1
            assert result.data[0]["posteam"] == "KC"
            assert result.data[0]["play_type"] == "pass"

    def test_filter_with_weeks(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test filters combined with week filtering."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl(
                [2024], weeks=[1], filters={"play_type": "pass"}
            )

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Week 1 pass plays only
            assert len(result.data) == 1
            assert result.data[0]["week"] == 1
            assert result.data[0]["play_type"] == "pass"

    def test_filter_no_matches(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test filters that match no rows."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            result = get_play_by_play_impl([2024], filters={"posteam": "NYG"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None
            assert "No data matched" in result.warning

    def test_filter_nonexistent_column(self, sample_pbp_df: pd.DataFrame) -> None:
        """Test filtering on a column that doesn't exist (ignored)."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = sample_pbp_df

            # Nonexistent column should be ignored
            result = get_play_by_play_impl(
                [2024], filters={"nonexistent_column": "value"}
            )

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # All rows returned since filter was ignored
            assert len(result.data) == 3

    def test_pagination_with_offset(self, large_pbp_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = large_pbp_df

            result = get_play_by_play_impl([2024], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            # First row should be play_id=10 (skipped first 10)
            assert result.data[0]["play_id"] == 10

    def test_pagination_with_limit(self, large_pbp_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = large_pbp_df

            result = get_play_by_play_impl([2024], limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_pagination_with_offset_and_limit(self, large_pbp_df: pd.DataFrame) -> None:
        """Test pagination with both offset and limit."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = large_pbp_df

            result = get_play_by_play_impl([2024], offset=20, limit=15)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 15
            # First row should be play_id=20 (skipped first 20)
            assert result.data[0]["play_id"] == 20

    def test_pagination_warning_shows_next_offset(
        self, large_pbp_df: pd.DataFrame
    ) -> None:
        """Test that truncation warning includes next offset."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_pbp_data") as mock_import:
            mock_import.return_value = large_pbp_df

            result = get_play_by_play_impl([2024], offset=10, limit=10)

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "offset=20" in result.warning


class TestGetPlayByPlayIntegration:
    """Integration tests for get_play_by_play (without mocks where possible)."""

    def test_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_play_by_play_impl([1990, 1991], [0, 25])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
