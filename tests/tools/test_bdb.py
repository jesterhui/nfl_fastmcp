"""Tests for the Big Data Bowl tools.

This module tests the BDB tool implementations including parameter validation,
data fetching, and error handling using mocked data.
"""

from pathlib import Path

import pandas as pd
import pytest

from fast_nfl_mcp.constants import BDB_AVAILABLE_WEEKS
from fast_nfl_mcp.kaggle_fetcher import KaggleFetcher
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.bdb import (
    get_bdb_games_impl,
    get_bdb_players_impl,
    get_bdb_plays_impl,
    get_bdb_tracking_impl,
)


@pytest.fixture
def mock_bdb_data(tmp_path: Path) -> None:
    """Set up mocked KaggleFetcher with test data for BDB 2026 structure."""
    import fast_nfl_mcp.kaggle_fetcher as module

    fetcher = KaggleFetcher()
    # Create a subdirectory like the real competition data
    data_subdir = tmp_path / "competition_data"
    data_subdir.mkdir()
    fetcher._data_path = tmp_path

    # Create supplementary_data.csv (contains both games and plays info)
    supplementary_df = pd.DataFrame(
        {
            "game_id": [2023090700, 2023090700, 2023091000, 2023091001],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 1, 2],
            "game_date": ["09/07/2023", "09/07/2023", "09/10/2023", "09/17/2023"],
            "game_time_eastern": ["20:20:00", "20:20:00", "13:00:00", "13:00:00"],
            "home_team_abbr": ["KC", "KC", "BUF", "SF"],
            "visitor_team_abbr": ["DET", "DET", "LA", "SEA"],
            "play_id": [1, 2, 1, 1],
            "play_description": [
                "Pass complete",
                "Run up middle",
                "Incomplete",
                "Sack",
            ],
            "quarter": [1, 1, 2, 3],
            "down": [1, 2, 3, 4],
            "yards_to_go": [10, 8, 5, 10],
            "possession_team": ["KC", "KC", "BUF", "SF"],
            "defensive_team": ["DET", "DET", "LA", "SEA"],
            "yards_gained": [15, 4, 0, -8],
            "expected_points_added": [1.5, 0.3, -0.5, -2.0],
        }
    )
    supplementary_df.to_csv(data_subdir / "supplementary_data.csv", index=False)

    # Create train directory for tracking data
    train_dir = data_subdir / "train"
    train_dir.mkdir()

    # Tracking data with multiple games/plays (input format for BDB 2026)
    tracking_rows = []
    for i in range(100):
        tracking_rows.append(
            {
                "game_id": 2023090700,
                "play_id": 1,
                "nfl_id": 43290,
                "player_name": "Patrick Mahomes",
                "frame_id": i + 1,
                "x": 10.0 + i * 0.5,
                "y": 26.65,
                "s": 5.0,
                "a": 1.2,
                "o": 90.0,
                "dir": 85.0,
                "player_position": "QB",
                "player_height": "6-2",
                "player_weight": 225,
                "player_birth_date": "1995-09-17",
            }
        )
    for i in range(50):
        tracking_rows.append(
            {
                "game_id": 2023090700,
                "play_id": 2,
                "nfl_id": 43290,
                "player_name": "Patrick Mahomes",
                "frame_id": i + 1,
                "x": 15.0 + i * 0.3,
                "y": 26.65,
                "s": 4.0,
                "a": 0.8,
                "o": 180.0,
                "dir": 175.0,
                "player_position": "QB",
                "player_height": "6-2",
                "player_weight": 225,
                "player_birth_date": "1995-09-17",
            }
        )
    # Add another player
    for i in range(20):
        tracking_rows.append(
            {
                "game_id": 2023090700,
                "play_id": 1,
                "nfl_id": 47956,
                "player_name": "Travis Kelce",
                "frame_id": i + 1,
                "x": 20.0 + i * 0.4,
                "y": 30.0,
                "s": 6.0,
                "a": 1.5,
                "o": 85.0,
                "dir": 80.0,
                "player_position": "TE",
                "player_height": "6-5",
                "player_weight": 250,
                "player_birth_date": "1989-10-05",
            }
        )

    tracking_df = pd.DataFrame(tracking_rows)
    tracking_df.to_csv(train_dir / "input_2023_w01.csv", index=False)

    module._fetcher = fetcher


class TestGetBdbGamesImpl:
    """Tests for the get_bdb_games_impl function."""

    def test_successful_fetch(self, mock_bdb_data: None) -> None:
        """Test successful games fetch."""
        result = get_bdb_games_impl()

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 3  # 3 unique games

    def test_with_filters(self, mock_bdb_data: None) -> None:
        """Test games fetch with filters."""
        result = get_bdb_games_impl(filters={"week": 1})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 2
        for row in result.data:
            assert row["week"] == 1

    def test_with_column_selection(self, mock_bdb_data: None) -> None:
        """Test games fetch with column selection."""
        result = get_bdb_games_impl(columns=["game_id", "home_team_abbr"])

        assert isinstance(result, SuccessResponse)
        assert result.metadata.columns == ["game_id", "home_team_abbr"]
        for row in result.data:
            assert set(row.keys()) == {"game_id", "home_team_abbr"}

    def test_with_pagination(self, mock_bdb_data: None) -> None:
        """Test games fetch with pagination."""
        result = get_bdb_games_impl(offset=1, limit=1)

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 1
        assert result.metadata.total_available == 3

    def test_filter_by_team(self, mock_bdb_data: None) -> None:
        """Test games fetch filtered by team."""
        result = get_bdb_games_impl(filters={"home_team_abbr": "KC"})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 1
        assert result.data[0]["home_team_abbr"] == "KC"


class TestGetBdbPlaysImpl:
    """Tests for the get_bdb_plays_impl function."""

    def test_successful_fetch(self, mock_bdb_data: None) -> None:
        """Test successful plays fetch."""
        result = get_bdb_plays_impl()

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 4

    def test_with_game_id_filter(self, mock_bdb_data: None) -> None:
        """Test plays fetch with game_id filter."""
        result = get_bdb_plays_impl(game_id=2023090700)

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 2
        for row in result.data:
            assert row["game_id"] == 2023090700

    def test_with_additional_filters(self, mock_bdb_data: None) -> None:
        """Test plays fetch with additional filters."""
        result = get_bdb_plays_impl(filters={"down": [3, 4]})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 2
        for row in result.data:
            assert row["down"] in [3, 4]

    def test_combined_game_id_and_filters(self, mock_bdb_data: None) -> None:
        """Test plays fetch with game_id and additional filters."""
        result = get_bdb_plays_impl(game_id=2023090700, filters={"down": 1})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 1
        assert result.data[0]["game_id"] == 2023090700
        assert result.data[0]["down"] == 1

    def test_with_column_selection(self, mock_bdb_data: None) -> None:
        """Test plays fetch with column selection."""
        result = get_bdb_plays_impl(columns=["game_id", "play_id", "play_description"])

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert set(row.keys()) == {"game_id", "play_id", "play_description"}


class TestGetBdbPlayersImpl:
    """Tests for the get_bdb_players_impl function."""

    def test_successful_fetch(self, mock_bdb_data: None) -> None:
        """Test successful players fetch."""
        result = get_bdb_players_impl()

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 2  # 2 unique players in tracking data

    def test_with_position_filter(self, mock_bdb_data: None) -> None:
        """Test players fetch filtered by position."""
        result = get_bdb_players_impl(filters={"player_position": "QB"})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 1
        for row in result.data:
            assert row["player_position"] == "QB"

    def test_with_column_selection(self, mock_bdb_data: None) -> None:
        """Test players fetch with column selection."""
        result = get_bdb_players_impl(columns=["nfl_id", "player_name"])

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert set(row.keys()) == {"nfl_id", "player_name"}

    def test_filter_by_nfl_id(self, mock_bdb_data: None) -> None:
        """Test players fetch filtered by nfl_id."""
        result = get_bdb_players_impl(filters={"nfl_id": 43290})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 1
        assert result.data[0]["nfl_id"] == 43290


class TestGetBdbTrackingImpl:
    """Tests for the get_bdb_tracking_impl function."""

    def test_successful_fetch(self, mock_bdb_data: None) -> None:
        """Test successful tracking fetch with week."""
        result = get_bdb_tracking_impl(week=1, limit=10)

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 10

    def test_requires_valid_week(self, mock_bdb_data: None) -> None:
        """Test that invalid week returns error."""
        result = get_bdb_tracking_impl(week=19)  # Week 19 is invalid (max is 18)

        assert isinstance(result, ErrorResponse)
        assert "Invalid week" in result.error

    def test_play_id_requires_game_id(self, mock_bdb_data: None) -> None:
        """Test that play_id without game_id returns error."""
        result = get_bdb_tracking_impl(week=1, play_id=1)

        assert isinstance(result, ErrorResponse)
        assert "play_id filter requires game_id" in result.error

    def test_play_id_accepts_game_id_in_filters(self, mock_bdb_data: None) -> None:
        """Test that play_id accepts game_id from filters dict."""
        result = get_bdb_tracking_impl(
            week=1, play_id=1, filters={"game_id": 2023090700}
        )

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert row["game_id"] == 2023090700
            assert row["play_id"] == 1

    def test_with_game_id_filter(self, mock_bdb_data: None) -> None:
        """Test tracking fetch filtered by game_id."""
        result = get_bdb_tracking_impl(week=1, game_id=2023090700, limit=100)

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert row["game_id"] == 2023090700

    def test_with_play_id_filter(self, mock_bdb_data: None) -> None:
        """Test tracking fetch filtered by game_id and play_id."""
        result = get_bdb_tracking_impl(week=1, game_id=2023090700, play_id=1)

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert row["game_id"] == 2023090700
            assert row["play_id"] == 1

    def test_with_nfl_id_filter(self, mock_bdb_data: None) -> None:
        """Test tracking fetch filtered by nfl_id."""
        result = get_bdb_tracking_impl(week=1, nfl_id=43290, limit=100)

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert row["nfl_id"] == 43290

    def test_default_limit_is_50(self, mock_bdb_data: None) -> None:
        """Test that default limit for tracking is 50."""
        result = get_bdb_tracking_impl(week=1)

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 50
        assert result.metadata.truncated is True

    def test_max_limit_is_100(self, mock_bdb_data: None) -> None:
        """Test that limit is capped at 100."""
        # Request more than max
        result = get_bdb_tracking_impl(week=1, limit=200)

        assert isinstance(result, SuccessResponse)
        # Should be capped at 100
        assert len(result.data) == 100

    def test_with_column_selection(self, mock_bdb_data: None) -> None:
        """Test tracking fetch with column selection."""
        result = get_bdb_tracking_impl(
            week=1, columns=["game_id", "play_id", "x", "y"], limit=10
        )

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert set(row.keys()) == {"game_id", "play_id", "x", "y"}

    def test_with_position_filter(self, mock_bdb_data: None) -> None:
        """Test tracking fetch filtered by player_position."""
        result = get_bdb_tracking_impl(
            week=1, game_id=2023090700, filters={"player_position": "TE"}
        )

        assert isinstance(result, SuccessResponse)
        for row in result.data:
            assert row["player_position"] == "TE"

    def test_pagination(self, mock_bdb_data: None) -> None:
        """Test tracking fetch with pagination."""
        result = get_bdb_tracking_impl(week=1, offset=10, limit=5)

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 5
        # First row should be frame 11 (skipped first 10)
        assert result.data[0]["frame_id"] == 11

    def test_combined_filters(self, mock_bdb_data: None) -> None:
        """Test tracking with multiple filters."""
        result = get_bdb_tracking_impl(
            week=1,
            game_id=2023090700,
            play_id=1,
            nfl_id=43290,
            limit=100,
        )

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 100
        for row in result.data:
            assert row["game_id"] == 2023090700
            assert row["play_id"] == 1
            assert row["nfl_id"] == 43290


class TestBdbToolsIntegration:
    """Integration tests for BDB tools without mocks where possible."""

    def test_week_boundary_values(self, mock_bdb_data: None) -> None:
        """Test tracking with boundary week values."""
        # Min valid week (week 1 has data in mock)
        result_min = get_bdb_tracking_impl(week=min(BDB_AVAILABLE_WEEKS), limit=1)
        assert isinstance(result_min, SuccessResponse)

        # Max valid week (week 18) - no data in mock, returns error about missing file
        result_max = get_bdb_tracking_impl(week=max(BDB_AVAILABLE_WEEKS), limit=1)
        # This will return an error because we only have week 1 in the mock
        assert isinstance(result_max, ErrorResponse)

    def test_invalid_week_below_range(self) -> None:
        """Test that week below valid range returns error."""
        result = get_bdb_tracking_impl(week=0)

        assert isinstance(result, ErrorResponse)
        assert "Invalid week" in result.error

    def test_invalid_week_above_range(self) -> None:
        """Test that week above valid range returns error."""
        result = get_bdb_tracking_impl(week=19)

        assert isinstance(result, ErrorResponse)
        assert "Invalid week" in result.error

    def test_empty_result_with_no_matches(self, mock_bdb_data: None) -> None:
        """Test that filters with no matches return empty data."""
        result = get_bdb_games_impl(filters={"home_team_abbr": "INVALID"})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No data matched" in result.warning
