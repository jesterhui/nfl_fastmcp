"""Tests for the player stats data retrieval tools.

This module tests the get_weekly_stats and get_seasonal_stats tools
including parameter validation, data fetching, and error handling
using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.constants import (
    DEFAULT_MAX_ROWS,
    MAX_SEASONS_SEASONAL,
    MAX_SEASONS_WEEKLY,
    MIN_SEASON,
)
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.player_stats import (
    get_seasonal_stats_impl,
    get_weekly_stats_impl,
    normalize_filters,
    validate_seasons,
)


class TestValidateSeasons:
    """Tests for the validate_seasons function."""

    def test_valid_seasons_weekly(self) -> None:
        """Test that valid seasons pass without warning for weekly stats."""
        valid, warning = validate_seasons([2023, 2024], MAX_SEASONS_WEEKLY)
        assert valid == [2023, 2024]
        assert warning is None

    def test_valid_seasons_seasonal(self) -> None:
        """Test that valid seasons pass without warning for seasonal stats."""
        valid, warning = validate_seasons(
            [2020, 2021, 2022, 2023, 2024], MAX_SEASONS_SEASONAL
        )
        assert valid == [2020, 2021, 2022, 2023, 2024]
        assert warning is None

    def test_empty_seasons(self) -> None:
        """Test that empty seasons list returns warning."""
        valid, warning = validate_seasons([], MAX_SEASONS_WEEKLY)
        assert valid == []
        assert warning is not None
        assert "No seasons provided" in warning

    def test_too_many_seasons_weekly(self) -> None:
        """Test that exceeding MAX_SEASONS_WEEKLY triggers truncation."""
        seasons = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        valid, warning = validate_seasons(seasons, MAX_SEASONS_WEEKLY)
        assert len(valid) == MAX_SEASONS_WEEKLY
        assert valid == seasons[:MAX_SEASONS_WEEKLY]
        assert warning is not None
        assert "Too many seasons" in warning

    def test_too_many_seasons_seasonal(self) -> None:
        """Test that exceeding MAX_SEASONS_SEASONAL triggers truncation."""
        seasons = list(range(2010, 2025))  # 15 seasons
        valid, warning = validate_seasons(seasons, MAX_SEASONS_SEASONAL)
        assert len(valid) == MAX_SEASONS_SEASONAL
        assert valid == seasons[:MAX_SEASONS_SEASONAL]
        assert warning is not None
        assert "Too many seasons" in warning

    def test_exactly_max_seasons_weekly(self) -> None:
        """Test that exactly MAX_SEASONS_WEEKLY passes without warning."""
        seasons = [2020, 2021, 2022, 2023, 2024]
        valid, warning = validate_seasons(seasons, MAX_SEASONS_WEEKLY)
        assert valid == seasons
        assert warning is None

    def test_exactly_max_seasons_seasonal(self) -> None:
        """Test that exactly MAX_SEASONS_SEASONAL passes without warning."""
        seasons = list(range(2015, 2025))  # 10 seasons
        valid, warning = validate_seasons(seasons, MAX_SEASONS_SEASONAL)
        assert valid == seasons
        assert warning is None

    def test_invalid_old_season(self) -> None:
        """Test that seasons before MIN_SEASON are filtered out."""
        valid, warning = validate_seasons([1990, 2023], MAX_SEASONS_WEEKLY)
        assert valid == [2023]
        assert warning is not None
        assert "Invalid seasons removed" in warning
        assert "1990" in warning

    def test_all_invalid_seasons(self) -> None:
        """Test that all invalid seasons returns empty list."""
        valid, warning = validate_seasons([1990, 1995], MAX_SEASONS_WEEKLY)
        assert valid == []
        assert warning is not None

    def test_single_valid_season(self) -> None:
        """Test single valid season."""
        valid, warning = validate_seasons([2024], MAX_SEASONS_WEEKLY)
        assert valid == [2024]
        assert warning is None

    def test_min_valid_season(self) -> None:
        """Test minimum valid season."""
        valid, warning = validate_seasons([MIN_SEASON], MAX_SEASONS_WEEKLY)
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
        result = normalize_filters({"team": "KC"})
        assert result == {"team": ["KC"]}

    def test_list_value_preserved(self) -> None:
        """Test that list values are preserved."""
        result = normalize_filters({"team": ["KC", "SF"]})
        assert result == {"team": ["KC", "SF"]}

    def test_mixed_values(self) -> None:
        """Test mixed single and list values."""
        result = normalize_filters({"team": "KC", "week": [1, 2, 3]})
        assert result == {"team": ["KC"], "week": [1, 2, 3]}

    def test_integer_values(self) -> None:
        """Test integer values are normalized."""
        result = normalize_filters({"season": 2024})
        assert result == {"season": [2024]}


class TestGetWeeklyStatsImpl:
    """Tests for the get_weekly_stats_impl function."""

    @pytest.fixture
    def sample_weekly_df(self) -> pd.DataFrame:
        """Create a sample weekly stats DataFrame."""
        return pd.DataFrame(
            {
                "player_id": ["P001", "P002", "P003", "P001", "P002"],
                "player_name": [
                    "Player A",
                    "Player B",
                    "Player C",
                    "Player A",
                    "Player B",
                ],
                "season": [2024, 2024, 2024, 2024, 2024],
                "week": [1, 1, 1, 2, 2],
                "team": ["KC", "SF", "BUF", "KC", "SF"],
                "passing_yards": [320, 280, 250, 350, 300],
                "passing_tds": [3, 2, 1, 4, 3],
                "rushing_yards": [25, 15, 45, 30, 20],
                "fantasy_points": [25.5, 20.0, 18.5, 30.0, 24.5],
            }
        )

    @pytest.fixture
    def large_weekly_df(self) -> pd.DataFrame:
        """Create a large weekly stats DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player_id": [f"P{i:03d}" for i in range(150)],
                "player_name": [f"Player {i}" for i in range(150)],
                "season": [2024] * 150,
                "week": [1] * 150,
                "team": ["KC"] * 150,
                "passing_yards": [300] * 150,
            }
        )

    def test_successful_fetch(self, sample_weekly_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = sample_weekly_df

            result = get_weekly_stats_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_week_filtering(self, sample_weekly_df: pd.DataFrame) -> None:
        """Test that week filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = sample_weekly_df

            result = get_weekly_stats_impl([2024], filters={"week": 1})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only week 1 stats should be returned
            assert len(result.data) == 3
            for row in result.data:
                assert row["week"] == 1

    def test_team_filtering(self, sample_weekly_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = sample_weekly_df

            result = get_weekly_stats_impl([2024], filters={"team": "KC"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 2
            for row in result.data:
                assert row["team"] == "KC"

    def test_truncation_at_max_rows(self, large_weekly_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = large_weekly_df

            result = get_weekly_stats_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_weekly_stats_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_too_many_seasons_warning(self, sample_weekly_df: pd.DataFrame) -> None:
        """Test that exceeding MAX_SEASONS_WEEKLY produces warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = sample_weekly_df

            result = get_weekly_stats_impl([2018, 2019, 2020, 2021, 2022, 2023, 2024])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Too many seasons" in result.warning

    def test_pagination_with_offset(self, large_weekly_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = large_weekly_df

            result = get_weekly_stats_impl([2024], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            # First row should have player_id "P010" (skipped first 10)
            assert result.data[0]["player_id"] == "P010"

    def test_pagination_with_limit(self, large_weekly_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = large_weekly_df

            result = get_weekly_stats_impl([2024], limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_weekly_stats_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_weekly_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_weekly_data") as mock_import:
            mock_import.return_value = sample_weekly_df

            result = get_weekly_stats_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player_name" in result.metadata.columns
            assert "passing_yards" in result.metadata.columns


class TestGetSeasonalStatsImpl:
    """Tests for the get_seasonal_stats_impl function."""

    @pytest.fixture
    def sample_seasonal_df(self) -> pd.DataFrame:
        """Create a sample seasonal stats DataFrame."""
        return pd.DataFrame(
            {
                "player_id": ["P001", "P002", "P003", "P001", "P002"],
                "player_name": [
                    "Player A",
                    "Player B",
                    "Player C",
                    "Player A",
                    "Player B",
                ],
                "season": [2023, 2023, 2023, 2024, 2024],
                "team": ["KC", "SF", "BUF", "KC", "SF"],
                "passing_yards": [4500, 4200, 3800, 4800, 4400],
                "passing_tds": [35, 28, 22, 40, 32],
                "interceptions": [10, 12, 8, 8, 10],
                "games": [17, 16, 17, 17, 17],
            }
        )

    @pytest.fixture
    def large_seasonal_df(self) -> pd.DataFrame:
        """Create a large seasonal stats DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player_id": [f"P{i:03d}" for i in range(150)],
                "player_name": [f"Player {i}" for i in range(150)],
                "season": [2024] * 150,
                "team": ["KC"] * 150,
                "passing_yards": [4000] * 150,
            }
        )

    def test_successful_fetch(self, sample_seasonal_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = sample_seasonal_df

            result = get_seasonal_stats_impl([2023, 2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_season_filtering(self, sample_seasonal_df: pd.DataFrame) -> None:
        """Test that season filtering works correctly."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = sample_seasonal_df

            result = get_seasonal_stats_impl([2023, 2024], filters={"season": 2024})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only 2024 season stats should be returned
            assert len(result.data) == 2
            for row in result.data:
                assert row["season"] == 2024

    def test_team_filtering(self, sample_seasonal_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = sample_seasonal_df

            result = get_seasonal_stats_impl([2023, 2024], filters={"team": "KC"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 2
            for row in result.data:
                assert row["team"] == "KC"

    def test_truncation_at_max_rows(self, large_seasonal_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = large_seasonal_df

            result = get_seasonal_stats_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_seasonal_stats_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_too_many_seasons_warning(self, sample_seasonal_df: pd.DataFrame) -> None:
        """Test that exceeding MAX_SEASONS_SEASONAL produces warning."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = sample_seasonal_df

            # More than 10 seasons
            seasons = list(range(2010, 2025))  # 15 seasons
            result = get_seasonal_stats_impl(seasons)

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Too many seasons" in result.warning

    def test_max_seasons_allowed(self, sample_seasonal_df: pd.DataFrame) -> None:
        """Test that MAX_SEASONS_SEASONAL (10) is allowed without warning."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = sample_seasonal_df

            # Exactly 10 seasons
            seasons = list(range(2015, 2025))
            result = get_seasonal_stats_impl(seasons)

            assert isinstance(result, SuccessResponse)
            # No warning about too many seasons
            if result.warning:
                assert "Too many seasons" not in result.warning

    def test_pagination_with_offset(self, large_seasonal_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = large_seasonal_df

            result = get_seasonal_stats_impl([2024], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            # First row should have player_id "P010" (skipped first 10)
            assert result.data[0]["player_id"] == "P010"

    def test_pagination_with_limit(self, large_seasonal_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = large_seasonal_df

            result = get_seasonal_stats_impl([2024], limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_seasonal_stats_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_seasonal_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_seasonal_data"
        ) as mock_import:
            mock_import.return_value = sample_seasonal_df

            result = get_seasonal_stats_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player_name" in result.metadata.columns
            assert "passing_yards" in result.metadata.columns


class TestPlayerStatsIntegration:
    """Integration tests for player stats tools (without mocks where possible)."""

    def test_weekly_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_weekly_stats_impl([1990, 1991])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None

    def test_seasonal_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_seasonal_stats_impl([1990, 1991])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
