"""Tests for the rosters data retrieval tool.

This module tests the get_rosters tool including parameter validation,
data fetching, and error handling using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.constants import (
    DEFAULT_MAX_ROWS,
    MAX_ROSTERS_SEASONS,
    MAX_WEEK,
    MIN_SEASON,
    MIN_WEEK,
)
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.rosters import (
    get_rosters_impl,
    normalize_filters,
    validate_seasons,
    validate_teams,
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
        """Test that exceeding MAX_ROSTERS_SEASONS triggers truncation."""
        seasons = [2019, 2020, 2021, 2022, 2023, 2024]
        valid, warning = validate_seasons(seasons)
        assert len(valid) == MAX_ROSTERS_SEASONS
        assert valid == seasons[:MAX_ROSTERS_SEASONS]
        assert warning is not None
        assert "Too many seasons" in warning

    def test_exactly_max_seasons(self) -> None:
        """Test that exactly MAX_ROSTERS_SEASONS passes without warning."""
        seasons = [2020, 2021, 2022, 2023, 2024]
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


class TestValidateTeams:
    """Tests for the validate_teams function."""

    def test_none_teams(self) -> None:
        """Test that None teams returns None without warning."""
        valid, warning = validate_teams(None)
        assert valid is None
        assert warning is None

    def test_empty_teams(self) -> None:
        """Test that empty teams list returns None without warning."""
        valid, warning = validate_teams([])
        assert valid is None
        assert warning is None

    def test_valid_teams(self) -> None:
        """Test valid team abbreviations pass without warning."""
        valid, warning = validate_teams(["KC", "SF", "TB"])
        assert valid == ["KC", "SF", "TB"]
        assert warning is None

    def test_lowercase_teams_normalized(self) -> None:
        """Test that lowercase team abbreviations are normalized to uppercase."""
        valid, warning = validate_teams(["kc", "sf"])
        assert valid == ["KC", "SF"]
        assert warning is None

    def test_mixed_case_teams_normalized(self) -> None:
        """Test that mixed case team abbreviations are normalized."""
        valid, warning = validate_teams(["Kc", "sF", "TB"])
        assert valid == ["KC", "SF", "TB"]
        assert warning is None

    def test_invalid_team_removed(self) -> None:
        """Test that invalid team abbreviations are removed with warning."""
        valid, warning = validate_teams(["KC", "XXX", "SF"])
        assert valid == ["KC", "SF"]
        assert warning is not None
        assert "Invalid team abbreviations removed" in warning
        assert "XXX" in warning

    def test_all_invalid_teams(self) -> None:
        """Test that all invalid teams returns None."""
        valid, warning = validate_teams(["XXX", "YYY", "ZZZ"])
        assert valid is None
        assert warning is not None

    def test_historical_teams(self) -> None:
        """Test that historical team abbreviations are valid."""
        valid, warning = validate_teams(["OAK", "SD", "STL"])
        assert valid == ["OAK", "SD", "STL"]
        assert warning is None

    def test_single_valid_team(self) -> None:
        """Test single valid team."""
        valid, warning = validate_teams(["KC"])
        assert valid == ["KC"]
        assert warning is None

    def test_all_32_current_teams(self) -> None:
        """Test that all 32 current NFL teams are valid."""
        teams = [
            "ARI",
            "ATL",
            "BAL",
            "BUF",
            "CAR",
            "CHI",
            "CIN",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GB",
            "HOU",
            "IND",
            "JAX",
            "KC",
            "LA",
            "LAC",
            "LV",
            "MIA",
            "MIN",
            "NE",
            "NO",
            "NYG",
            "NYJ",
            "PHI",
            "PIT",
            "SEA",
            "SF",
            "TB",
            "TEN",
            "WAS",
        ]
        valid, warning = validate_teams(teams)
        assert valid == teams
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
        result = normalize_filters({"team": "KC", "position": ["QB", "WR"]})
        assert result == {"team": ["KC"], "position": ["QB", "WR"]}

    def test_integer_values(self) -> None:
        """Test integer values are normalized."""
        result = normalize_filters({"week": 3})
        assert result == {"week": [3]}


class TestGetRostersImpl:
    """Tests for the get_rosters_impl function."""

    @pytest.fixture
    def sample_rosters_df(self) -> pd.DataFrame:
        """Create a sample rosters DataFrame."""
        return pd.DataFrame(
            {
                "player_id": ["00-0033873", "00-0036945", "00-0036389", "00-0037543"],
                "player_name": [
                    "Patrick Mahomes",
                    "Josh Allen",
                    "Brock Purdy",
                    "Travis Kelce",
                ],
                "team": ["KC", "BUF", "SF", "KC"],
                "position": ["QB", "QB", "QB", "TE"],
                "jersey_number": [15, 17, 13, 87],
                "status": ["ACT", "ACT", "ACT", "ACT"],
                "height": ["6-2", "6-5", "6-1", "6-5"],
                "weight": [225, 237, 212, 250],
                "college": ["Texas Tech", "Wyoming", "Iowa State", "Cincinnati"],
                "week": [1, 1, 1, 1],
            }
        )

    @pytest.fixture
    def large_rosters_df(self) -> pd.DataFrame:
        """Create a large rosters DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player_id": [f"00-00{i:05d}" for i in range(150)],
                "player_name": [f"Player {i}" for i in range(150)],
                "team": ["KC"] * 150,
                "position": ["WR"] * 150,
                "week": [1] * 150,
            }
        )

    def test_successful_fetch(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 4
            assert result.metadata.row_count == 4

    def test_team_filtering(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], teams=["KC"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only KC players should be returned
            assert len(result.data) == 2
            for row in result.data:
                assert row["team"] == "KC"

    def test_week_filtering(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that week filtering works correctly."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], weeks=[1])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 4
            for row in result.data:
                assert row["week"] == 1

    def test_truncation_at_max_rows(self, large_rosters_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = large_rosters_df

            result = get_rosters_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_rosters_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_invalid_seasons_warning(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that invalid seasons produce warning."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([1990, 2024])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid seasons removed" in result.warning

    def test_too_many_seasons_warning(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that exceeding MAX_ROSTERS_SEASONS produces warning."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2019, 2020, 2021, 2022, 2023, 2024])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Too many seasons" in result.warning

    def test_invalid_teams_warning(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that invalid teams produce warning."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], teams=["KC", "XXX"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid team abbreviations removed" in result.warning

    def test_invalid_weeks_warning(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that invalid weeks produce warning."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], weeks=[0, 1, 20])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid weeks removed" in result.warning

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_rosters_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_empty_dataframe_returns_success(self) -> None:
        """Test that empty DataFrame returns success with warning."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = pd.DataFrame()

            result = get_rosters_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None

    def test_columns_in_metadata(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player_id" in result.metadata.columns
            assert "team" in result.metadata.columns

    def test_data_types_converted(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that numpy types are converted to Python native types."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024])

            assert isinstance(result, SuccessResponse)
            # Check that values are Python native types, not numpy
            for row in result.data:
                assert isinstance(row["jersey_number"], int)
                assert isinstance(row["weight"], int)

    def test_combined_warnings(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that multiple warnings are combined."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            # Invalid season, invalid week, and invalid team
            result = get_rosters_impl([1990, 2024], weeks=[0, 1], teams=["KC", "XXX"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid seasons removed" in result.warning
            assert "Invalid weeks removed" in result.warning
            assert "Invalid team abbreviations removed" in result.warning

    def test_filter_by_position(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test filtering by position using filters parameter."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], filters={"position": "QB"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only QBs should be returned
            assert len(result.data) == 3
            for row in result.data:
                assert row["position"] == "QB"

    def test_combined_team_and_position_filter(
        self, sample_rosters_df: pd.DataFrame
    ) -> None:
        """Test combining team parameter with position filter."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], teams=["KC"], filters={"position": "QB"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only KC QBs should be returned
            assert len(result.data) == 1
            assert result.data[0]["team"] == "KC"
            assert result.data[0]["position"] == "QB"

    def test_filter_no_matches(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test filters that match no rows."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], teams=["NYG"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None
            assert "No data matched" in result.warning

    def test_all_invalid_teams_returns_empty(self) -> None:
        """Test that all invalid teams returns empty result without fetching data."""
        # This test verifies the fix: when teams are provided but all are invalid,
        # we should return empty data, NOT fetch all teams
        result = get_rosters_impl([2024], teams=["XXX", "YYY", "ZZZ"])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.metadata.total_available == 0
        assert result.warning is not None
        assert "Invalid team abbreviations removed" in result.warning

    def test_all_invalid_teams_does_not_fetch_data(
        self, sample_rosters_df: pd.DataFrame
    ) -> None:
        """Test that all invalid teams does not call the data fetcher."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], teams=["XXX"])

            # The mock should NOT have been called because we short-circuit
            mock_import.assert_not_called()
            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 0

    def test_some_valid_some_invalid_teams_fetches_valid_only(
        self, sample_rosters_df: pd.DataFrame
    ) -> None:
        """Test that mixed valid/invalid teams only fetches valid teams."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl([2024], teams=["KC", "XXX"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Should only return KC players (2 in sample data)
            assert len(result.data) == 2
            for row in result.data:
                assert row["team"] == "KC"
            assert result.warning is not None
            assert "Invalid team abbreviations removed" in result.warning

    def test_pagination_with_offset(self, large_rosters_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = large_rosters_df

            result = get_rosters_impl([2024], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            # First row should be Player 10 (skipped first 10)
            assert result.data[0]["player_name"] == "Player 10"

    def test_pagination_with_limit(self, large_rosters_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = large_rosters_df

            result = get_rosters_impl([2024], limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_pagination_with_offset_and_limit(
        self, large_rosters_df: pd.DataFrame
    ) -> None:
        """Test pagination with both offset and limit."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = large_rosters_df

            result = get_rosters_impl([2024], offset=20, limit=15)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 15
            # First row should be Player 20 (skipped first 20)
            assert result.data[0]["player_name"] == "Player 20"

    def test_column_selection(self, sample_rosters_df: pd.DataFrame) -> None:
        """Test that column selection works correctly."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_weekly_rosters"
        ) as mock_import:
            mock_import.return_value = sample_rosters_df

            result = get_rosters_impl(
                [2024], columns=["player_name", "team", "position"]
            )

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert result.metadata.columns == ["player_name", "team", "position"]
            # Verify only selected columns are in data
            for row in result.data:
                assert set(row.keys()) == {"player_name", "team", "position"}


class TestGetRostersIntegration:
    """Integration tests for get_rosters (without mocks where possible)."""

    def test_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_rosters_impl([1990, 1991], weeks=[0, 25], teams=["XXX"])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
