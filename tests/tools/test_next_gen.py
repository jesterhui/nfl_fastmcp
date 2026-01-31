"""Tests for the Next Gen Stats data retrieval tools.

This module tests the get_ngs_passing, get_ngs_rushing, and get_ngs_receiving
tools including parameter validation, data fetching, and error handling
using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.constants import DEFAULT_MAX_ROWS, MAX_SEASONS_NGS, MIN_SEASON_NGS
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.next_gen import (
    get_ngs_passing_impl,
    get_ngs_receiving_impl,
    get_ngs_rushing_impl,
    normalize_filters,
    validate_seasons,
)


class TestValidateSeasons:
    """Tests for the validate_seasons function."""

    def test_valid_seasons(self) -> None:
        """Test that valid seasons pass without warning."""
        valid, warning = validate_seasons([2023, 2024], MAX_SEASONS_NGS)
        assert valid == [2023, 2024]
        assert warning is None

    def test_max_seasons_allowed(self) -> None:
        """Test that exactly MAX_SEASONS_NGS passes without warning."""
        seasons = [2020, 2021, 2022, 2023, 2024]
        valid, warning = validate_seasons(seasons, MAX_SEASONS_NGS)
        assert valid == seasons
        assert warning is None

    def test_empty_seasons(self) -> None:
        """Test that empty seasons list returns warning."""
        valid, warning = validate_seasons([], MAX_SEASONS_NGS)
        assert valid == []
        assert warning is not None
        assert "No seasons provided" in warning

    def test_too_many_seasons(self) -> None:
        """Test that exceeding MAX_SEASONS_NGS triggers truncation."""
        seasons = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        valid, warning = validate_seasons(seasons, MAX_SEASONS_NGS)
        assert len(valid) == MAX_SEASONS_NGS
        assert valid == seasons[:MAX_SEASONS_NGS]
        assert warning is not None
        assert "Too many seasons" in warning

    def test_invalid_old_season(self) -> None:
        """Test that seasons before MIN_SEASON are filtered out."""
        valid, warning = validate_seasons([1990, 2023], MAX_SEASONS_NGS)
        assert valid == [2023]
        assert warning is not None
        assert "Invalid seasons removed" in warning
        assert "1990" in warning

    def test_all_invalid_seasons(self) -> None:
        """Test that all invalid seasons returns empty list."""
        valid, warning = validate_seasons([1990, 1995], MAX_SEASONS_NGS)
        assert valid == []
        assert warning is not None

    def test_invalid_seasons_filtered_before_max_limit(self) -> None:
        """Test that invalid seasons are filtered before applying max limit."""
        # 10 invalid seasons followed by 3 valid seasons
        seasons = list(range(1989, 1999)) + [2022, 2023, 2024]
        valid, warning = validate_seasons(seasons, MAX_SEASONS_NGS)

        # Should get the 3 valid seasons
        assert valid == [2022, 2023, 2024]
        assert warning is not None
        assert "Invalid seasons removed" in warning

    def test_mixed_invalid_valid_with_truncation(self) -> None:
        """Test filtering invalid seasons then truncating to max limit."""
        # Mix of invalid and valid seasons, more valid than max allows
        seasons = [1990, 1991, 1992, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        valid, warning = validate_seasons(seasons, MAX_SEASONS_NGS)

        # Should get first 5 valid seasons after filtering
        assert valid == [2018, 2019, 2020, 2021, 2022]
        assert len(valid) == MAX_SEASONS_NGS
        assert warning is not None
        assert "Invalid seasons removed" in warning
        assert "Too many seasons" in warning

    def test_single_valid_season(self) -> None:
        """Test single valid season."""
        valid, warning = validate_seasons([2024], MAX_SEASONS_NGS)
        assert valid == [2024]
        assert warning is None

    def test_min_valid_season(self) -> None:
        """Test minimum valid season for NGS data (2016)."""
        valid, warning = validate_seasons([MIN_SEASON_NGS], MAX_SEASONS_NGS)
        assert valid == [MIN_SEASON_NGS]
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
        result = normalize_filters({"team_abbr": "KC"})
        assert result == {"team_abbr": ["KC"]}

    def test_list_value_preserved(self) -> None:
        """Test that list values are preserved."""
        result = normalize_filters({"team_abbr": ["KC", "SF"]})
        assert result == {"team_abbr": ["KC", "SF"]}

    def test_mixed_values(self) -> None:
        """Test mixed single and list values."""
        result = normalize_filters({"team_abbr": "KC", "week": [1, 2, 3]})
        assert result == {"team_abbr": ["KC"], "week": [1, 2, 3]}

    def test_integer_values(self) -> None:
        """Test integer values are normalized."""
        result = normalize_filters({"season": 2024})
        assert result == {"season": [2024]}


class TestGetNgsPassingImpl:
    """Tests for the get_ngs_passing_impl function."""

    @pytest.fixture
    def sample_ngs_passing_df(self) -> pd.DataFrame:
        """Create a sample NGS passing DataFrame."""
        return pd.DataFrame(
            {
                "player_display_name": ["Patrick Mahomes", "Josh Allen", "Joe Burrow"],
                "player_gsis_id": ["P001", "P002", "P003"],
                "team_abbr": ["KC", "BUF", "CIN"],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 1],
                "avg_time_to_throw": [2.5, 2.8, 2.6],
                "avg_completed_air_yards": [8.2, 7.5, 8.0],
                "aggressiveness": [18.5, 15.2, 16.8],
                "passer_rating": [110.5, 105.2, 98.7],
                "completion_percentage": [68.5, 65.2, 70.1],
                "passing_yards": [320, 280, 250],
                "passing_tds": [3, 2, 2],
            }
        )

    @pytest.fixture
    def large_ngs_df(self) -> pd.DataFrame:
        """Create a large NGS DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player_display_name": [f"Player {i}" for i in range(150)],
                "player_gsis_id": [f"P{i:03d}" for i in range(150)],
                "team_abbr": ["KC"] * 150,
                "season": [2024] * 150,
                "week": [1] * 150,
                "avg_time_to_throw": [2.5] * 150,
                "passing_yards": [300] * 150,
            }
        )

    def test_successful_fetch(self, sample_ngs_passing_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_passing_df

            result = get_ngs_passing_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_team_filtering(self, sample_ngs_passing_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_passing_df

            result = get_ngs_passing_impl([2024], filters={"team_abbr": "KC"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 1
            assert result.data[0]["team_abbr"] == "KC"

    def test_truncation_at_max_rows(self, large_ngs_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = large_ngs_df

            result = get_ngs_passing_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_ngs_passing_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_too_many_seasons_warning(
        self, sample_ngs_passing_df: pd.DataFrame
    ) -> None:
        """Test that exceeding MAX_SEASONS_NGS produces warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_passing_df

            result = get_ngs_passing_impl([2018, 2019, 2020, 2021, 2022, 2023, 2024])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Too many seasons" in result.warning

    def test_pagination_with_offset(self, large_ngs_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = large_ngs_df

            result = get_ngs_passing_impl([2024], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            # First row should have player_gsis_id "P010" (skipped first 10)
            assert result.data[0]["player_gsis_id"] == "P010"

    def test_pagination_with_limit(self, large_ngs_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = large_ngs_df

            result = get_ngs_passing_impl([2024], limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_ngs_passing_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_ngs_passing_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_passing_df

            result = get_ngs_passing_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player_display_name" in result.metadata.columns
            assert "avg_time_to_throw" in result.metadata.columns


class TestGetNgsRushingImpl:
    """Tests for the get_ngs_rushing_impl function."""

    @pytest.fixture
    def sample_ngs_rushing_df(self) -> pd.DataFrame:
        """Create a sample NGS rushing DataFrame."""
        return pd.DataFrame(
            {
                "player_display_name": [
                    "Derrick Henry",
                    "Christian McCaffrey",
                    "Saquon Barkley",
                ],
                "player_gsis_id": ["R001", "R002", "R003"],
                "team_abbr": ["TEN", "SF", "NYG"],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 1],
                "efficiency": [105.2, 110.5, 98.7],
                "avg_time_to_los": [2.8, 2.5, 2.9],
                "rush_yards": [125, 95, 80],
                "rush_touchdowns": [2, 1, 0],
                "rush_yards_over_expected": [15.5, 22.3, -5.2],
            }
        )

    def test_successful_fetch(self, sample_ngs_rushing_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_rushing_df

            result = get_ngs_rushing_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_team_filtering(self, sample_ngs_rushing_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_rushing_df

            result = get_ngs_rushing_impl([2024], filters={"team_abbr": "SF"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 1
            assert result.data[0]["team_abbr"] == "SF"

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_ngs_rushing_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_ngs_rushing_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_ngs_rushing_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_rushing_df

            result = get_ngs_rushing_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player_display_name" in result.metadata.columns
            assert "efficiency" in result.metadata.columns


class TestGetNgsReceivingImpl:
    """Tests for the get_ngs_receiving_impl function."""

    @pytest.fixture
    def sample_ngs_receiving_df(self) -> pd.DataFrame:
        """Create a sample NGS receiving DataFrame."""
        return pd.DataFrame(
            {
                "player_display_name": [
                    "Tyreek Hill",
                    "Davante Adams",
                    "CeeDee Lamb",
                ],
                "player_gsis_id": ["W001", "W002", "W003"],
                "team_abbr": ["MIA", "LV", "DAL"],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 1],
                "avg_cushion": [5.2, 4.8, 5.5],
                "avg_separation": [3.2, 2.8, 2.5],
                "avg_intended_air_yards": [12.5, 10.2, 8.5],
                "catch_percentage": [72.5, 68.2, 75.0],
                "avg_yac": [6.8, 4.5, 5.2],
                "receiving_yards": [145, 110, 95],
                "receiving_touchdowns": [2, 1, 1],
            }
        )

    def test_successful_fetch(self, sample_ngs_receiving_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_receiving_df

            result = get_ngs_receiving_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_team_filtering(self, sample_ngs_receiving_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_receiving_df

            result = get_ngs_receiving_impl([2024], filters={"team_abbr": "MIA"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 1
            assert result.data[0]["team_abbr"] == "MIA"

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_ngs_receiving_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_ngs_receiving_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_ngs_receiving_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ngs_data") as mock_import:
            mock_import.return_value = sample_ngs_receiving_df

            result = get_ngs_receiving_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player_display_name" in result.metadata.columns
            assert "avg_separation" in result.metadata.columns


class TestNgsIntegration:
    """Integration tests for NGS tools (without mocks where possible)."""

    def test_passing_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_ngs_passing_impl([1990, 1991])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None

    def test_rushing_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_ngs_rushing_impl([1990, 1991])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None

    def test_receiving_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_ngs_receiving_impl([1990, 1991])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
