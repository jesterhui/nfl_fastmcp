"""Tests for the schedules data retrieval tool.

This module tests the get_schedules tool including parameter validation,
data fetching, and error handling using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.core.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.schedules import (
    get_schedules_impl,
    normalize_filters,
    validate_seasons,
)
from fast_nfl_mcp.utils.constants import (
    DEFAULT_MAX_ROWS,
    MAX_SEASONS_SCHEDULES,
    MIN_SEASON,
)


class TestValidateSeasons:
    """Tests for the validate_seasons function."""

    def test_valid_seasons(self) -> None:
        """Test that valid seasons pass without warning."""
        valid, warning = validate_seasons([2023, 2024], MAX_SEASONS_SCHEDULES)
        assert valid == [2023, 2024]
        assert warning is None

    def test_empty_seasons(self) -> None:
        """Test that empty seasons list returns warning."""
        valid, warning = validate_seasons([], MAX_SEASONS_SCHEDULES)
        assert valid == []
        assert warning is not None
        assert "No seasons provided" in warning

    def test_too_many_seasons(self) -> None:
        """Test that exceeding MAX_SEASONS_SCHEDULES triggers truncation."""
        # 15 seasons, more than max 10
        seasons = list(range(2010, 2025))
        valid, warning = validate_seasons(seasons, MAX_SEASONS_SCHEDULES)
        assert len(valid) == MAX_SEASONS_SCHEDULES
        assert valid == seasons[:MAX_SEASONS_SCHEDULES]
        assert warning is not None
        assert "Too many seasons" in warning

    def test_exactly_max_seasons(self) -> None:
        """Test that exactly MAX_SEASONS_SCHEDULES passes without warning."""
        seasons = list(range(2015, 2025))  # 10 seasons
        valid, warning = validate_seasons(seasons, MAX_SEASONS_SCHEDULES)
        assert valid == seasons
        assert warning is None

    def test_invalid_old_season(self) -> None:
        """Test that seasons before MIN_SEASON are filtered out."""
        valid, warning = validate_seasons([1990, 2023], MAX_SEASONS_SCHEDULES)
        assert valid == [2023]
        assert warning is not None
        assert "Invalid seasons removed" in warning
        assert "1990" in warning

    def test_all_invalid_seasons(self) -> None:
        """Test that all invalid seasons returns empty list."""
        valid, warning = validate_seasons([1990, 1995], MAX_SEASONS_SCHEDULES)
        assert valid == []
        assert warning is not None

    def test_invalid_seasons_filtered_before_max_limit(self) -> None:
        """Test that invalid seasons are filtered before applying max limit."""
        # 10 invalid seasons followed by 8 valid seasons
        seasons = list(range(1989, 1999)) + list(range(2017, 2025))
        valid, warning = validate_seasons(seasons, MAX_SEASONS_SCHEDULES)

        # Should get all 8 valid seasons (since max is 10)
        assert valid == list(range(2017, 2025))
        assert warning is not None
        assert "Invalid seasons removed" in warning

    def test_single_valid_season(self) -> None:
        """Test single valid season."""
        valid, warning = validate_seasons([2024], MAX_SEASONS_SCHEDULES)
        assert valid == [2024]
        assert warning is None

    def test_min_valid_season(self) -> None:
        """Test minimum valid season."""
        valid, warning = validate_seasons([MIN_SEASON], MAX_SEASONS_SCHEDULES)
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
        result = normalize_filters({"home_team": "KC"})
        assert result == {"home_team": ["KC"]}

    def test_list_value_preserved(self) -> None:
        """Test that list values are preserved."""
        result = normalize_filters({"home_team": ["KC", "SF"]})
        assert result == {"home_team": ["KC", "SF"]}

    def test_mixed_values(self) -> None:
        """Test mixed single and list values."""
        result = normalize_filters({"home_team": "KC", "week": [1, 2, 3]})
        assert result == {"home_team": ["KC"], "week": [1, 2, 3]}

    def test_integer_values(self) -> None:
        """Test integer values are normalized."""
        result = normalize_filters({"week": 1})
        assert result == {"week": [1]}


class TestGetSchedulesImpl:
    """Tests for the get_schedules_impl function."""

    @pytest.fixture
    def sample_schedules_df(self) -> pd.DataFrame:
        """Create a sample schedules DataFrame."""
        return pd.DataFrame(
            {
                "game_id": [
                    "2024_01_KC_BAL",
                    "2024_01_SF_PIT",
                    "2024_02_KC_CIN",
                    "2024_02_SF_LA",
                    "2024_03_KC_ATL",
                ],
                "season": [2024, 2024, 2024, 2024, 2024],
                "week": [1, 1, 2, 2, 3],
                "gameday": [
                    "2024-09-05",
                    "2024-09-08",
                    "2024-09-15",
                    "2024-09-15",
                    "2024-09-22",
                ],
                "home_team": ["KC", "PIT", "CIN", "SF", "ATL"],
                "away_team": ["BAL", "SF", "KC", "LA", "KC"],
                "home_score": [27, 21, 28, 24, 17],
                "away_score": [20, 30, 31, 27, 22],
            }
        )

    @pytest.fixture
    def large_schedules_df(self) -> pd.DataFrame:
        """Create a large schedules DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "game_id": [f"2024_{i:02d}_KC_OPP" for i in range(1, 151)],
                "season": [2024] * 150,
                "week": list(range(1, 151)),
                "home_team": ["KC"] * 150,
                "away_team": ["OPP"] * 150,
                "home_score": [27] * 150,
                "away_score": [20] * 150,
            }
        )

    def test_successful_fetch(self, sample_schedules_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = sample_schedules_df

            result = get_schedules_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_week_filtering(self, sample_schedules_df: pd.DataFrame) -> None:
        """Test that week filtering works correctly."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = sample_schedules_df

            result = get_schedules_impl([2024], filters={"week": 1})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only week 1 games should be returned
            assert len(result.data) == 2
            for row in result.data:
                assert row["week"] == 1

    def test_team_filtering(self, sample_schedules_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = sample_schedules_df

            result = get_schedules_impl([2024], filters={"home_team": "KC"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 1
            for row in result.data:
                assert row["home_team"] == "KC"

    def test_truncation_at_max_rows(self, large_schedules_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = large_schedules_df

            result = get_schedules_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_schedules_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_too_many_seasons_warning(self, sample_schedules_df: pd.DataFrame) -> None:
        """Test that exceeding MAX_SEASONS_SCHEDULES produces warning."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = sample_schedules_df

            # 15 seasons, more than max 10
            seasons = list(range(2010, 2025))
            result = get_schedules_impl(seasons)

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Too many seasons" in result.warning

    def test_pagination_with_offset(self, large_schedules_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = large_schedules_df

            result = get_schedules_impl([2024], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            # First row should have week 11 (skipped first 10)
            assert result.data[0]["week"] == 11

    def test_pagination_with_limit(self, large_schedules_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = large_schedules_df

            result = get_schedules_impl([2024], limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_schedules_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_schedules_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = sample_schedules_df

            result = get_schedules_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "home_team" in result.metadata.columns
            assert "away_team" in result.metadata.columns

    def test_column_selection(self, sample_schedules_df: pd.DataFrame) -> None:
        """Test that specific columns can be selected."""
        with patch("fast_nfl_mcp.data.schema.nfl.import_schedules") as mock_import:
            mock_import.return_value = sample_schedules_df

            result = get_schedules_impl(
                [2024], columns=["game_id", "home_team", "away_team"]
            )

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert result.metadata.columns == ["game_id", "home_team", "away_team"]
            # Check that only requested columns are in data
            for row in result.data:
                assert set(row.keys()) == {"game_id", "home_team", "away_team"}


class TestSchedulesIntegration:
    """Integration tests for schedules tools (without mocks where possible)."""

    def test_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_schedules_impl([1990, 1991])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
