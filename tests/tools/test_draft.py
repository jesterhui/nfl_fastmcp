"""Tests for the draft picks data retrieval tool.

This module tests the get_draft_picks tool including parameter validation,
data fetching, and error handling using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.constants import (
    DEFAULT_MAX_ROWS,
    MAX_SEASONS_DRAFT,
    MIN_SEASON,
)
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.draft import (
    get_draft_picks_impl,
    normalize_filters,
    validate_seasons,
)


class TestValidateSeasons:
    """Tests for the validate_seasons function."""

    def test_valid_seasons(self) -> None:
        """Test that valid seasons pass without warning."""
        valid, warning = validate_seasons([2023, 2024], MAX_SEASONS_DRAFT)
        assert valid == [2023, 2024]
        assert warning is None

    def test_empty_seasons(self) -> None:
        """Test that empty seasons list returns warning."""
        valid, warning = validate_seasons([], MAX_SEASONS_DRAFT)
        assert valid == []
        assert warning is not None
        assert "No seasons provided" in warning

    def test_too_many_seasons(self) -> None:
        """Test that exceeding MAX_SEASONS_DRAFT triggers truncation."""
        # 25 seasons, more than max 20
        seasons = list(range(2000, 2025))
        valid, warning = validate_seasons(seasons, MAX_SEASONS_DRAFT)
        assert len(valid) == MAX_SEASONS_DRAFT
        assert valid == seasons[:MAX_SEASONS_DRAFT]
        assert warning is not None
        assert "Too many seasons" in warning

    def test_exactly_max_seasons(self) -> None:
        """Test that exactly MAX_SEASONS_DRAFT passes without warning."""
        seasons = list(range(2005, 2025))  # 20 seasons
        valid, warning = validate_seasons(seasons, MAX_SEASONS_DRAFT)
        assert valid == seasons
        assert warning is None

    def test_invalid_old_season(self) -> None:
        """Test that seasons before MIN_SEASON are filtered out."""
        valid, warning = validate_seasons([1990, 2023], MAX_SEASONS_DRAFT)
        assert valid == [2023]
        assert warning is not None
        assert "Invalid seasons removed" in warning
        assert "1990" in warning

    def test_all_invalid_seasons(self) -> None:
        """Test that all invalid seasons returns empty list."""
        valid, warning = validate_seasons([1990, 1995], MAX_SEASONS_DRAFT)
        assert valid == []
        assert warning is not None

    def test_invalid_seasons_filtered_before_max_limit(self) -> None:
        """Test that invalid seasons are filtered before applying max limit."""
        # 10 invalid seasons followed by 15 valid seasons
        seasons = list(range(1989, 1999)) + list(range(2010, 2025))
        valid, warning = validate_seasons(seasons, MAX_SEASONS_DRAFT)

        # Should get all 15 valid seasons (since max is 20)
        assert valid == list(range(2010, 2025))
        assert warning is not None
        assert "Invalid seasons removed" in warning

    def test_single_valid_season(self) -> None:
        """Test single valid season."""
        valid, warning = validate_seasons([2024], MAX_SEASONS_DRAFT)
        assert valid == [2024]
        assert warning is None

    def test_min_valid_season(self) -> None:
        """Test minimum valid season."""
        valid, warning = validate_seasons([MIN_SEASON], MAX_SEASONS_DRAFT)
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
        result = normalize_filters({"team": "KC", "round": [1, 2, 3]})
        assert result == {"team": ["KC"], "round": [1, 2, 3]}

    def test_integer_values(self) -> None:
        """Test integer values are normalized."""
        result = normalize_filters({"round": 1})
        assert result == {"round": [1]}


class TestGetDraftPicksImpl:
    """Tests for the get_draft_picks_impl function."""

    @pytest.fixture
    def sample_draft_df(self) -> pd.DataFrame:
        """Create a sample draft picks DataFrame."""
        return pd.DataFrame(
            {
                "season": [2024, 2024, 2024, 2024, 2024],
                "round": [1, 1, 1, 2, 2],
                "pick": [1, 2, 3, 33, 34],
                "team": ["CHI", "WAS", "NE", "KC", "SF"],
                "pfr_player_name": [
                    "Player A",
                    "Player B",
                    "Player C",
                    "Player D",
                    "Player E",
                ],
                "position": ["QB", "QB", "WR", "OT", "CB"],
                "age": [22, 21, 23, 22, 24],
                "college": [
                    "USC",
                    "North Carolina",
                    "Ohio State",
                    "Texas",
                    "Michigan",
                ],
            }
        )

    @pytest.fixture
    def large_draft_df(self) -> pd.DataFrame:
        """Create a large draft picks DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "season": [2024] * 150,
                "round": [1] * 150,
                "pick": list(range(1, 151)),
                "team": ["KC"] * 150,
                "pfr_player_name": [f"Player {i}" for i in range(150)],
                "position": ["QB"] * 150,
            }
        )

    def test_successful_fetch(self, sample_draft_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = sample_draft_df

            result = get_draft_picks_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_round_filtering(self, sample_draft_df: pd.DataFrame) -> None:
        """Test that round filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = sample_draft_df

            result = get_draft_picks_impl([2024], filters={"round": 1})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # Only round 1 picks should be returned
            assert len(result.data) == 3
            for row in result.data:
                assert row["round"] == 1

    def test_team_filtering(self, sample_draft_df: pd.DataFrame) -> None:
        """Test that team filtering works correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = sample_draft_df

            result = get_draft_picks_impl([2024], filters={"team": "KC"})

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 1
            for row in result.data:
                assert row["team"] == "KC"

    def test_truncation_at_max_rows(self, large_draft_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = large_draft_df

            result = get_draft_picks_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_empty_seasons_warning(self) -> None:
        """Test that empty seasons returns warning."""
        result = get_draft_picks_impl([])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
        assert "No seasons provided" in result.warning

    def test_too_many_seasons_warning(self, sample_draft_df: pd.DataFrame) -> None:
        """Test that exceeding MAX_SEASONS_DRAFT produces warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = sample_draft_df

            # 25 seasons, more than max 20
            seasons = list(range(2000, 2025))
            result = get_draft_picks_impl(seasons)

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Too many seasons" in result.warning

    def test_pagination_with_offset(self, large_draft_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = large_draft_df

            result = get_draft_picks_impl([2024], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            # First row should have pick 11 (skipped first 10)
            assert result.data[0]["pick"] == 11

    def test_pagination_with_limit(self, large_draft_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = large_draft_df

            result = get_draft_picks_impl([2024], limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_draft_picks_impl([2024])

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_draft_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = sample_draft_df

            result = get_draft_picks_impl([2024])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "pfr_player_name" in result.metadata.columns
            assert "position" in result.metadata.columns

    def test_column_selection(self, sample_draft_df: pd.DataFrame) -> None:
        """Test that specific columns can be selected."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_draft_picks") as mock_import:
            mock_import.return_value = sample_draft_df

            result = get_draft_picks_impl([2024], columns=["team", "pick", "position"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert result.metadata.columns == ["team", "pick", "position"]
            # Check that only requested columns are in data
            for row in result.data:
                assert set(row.keys()) == {"team", "pick", "position"}


class TestDraftIntegration:
    """Integration tests for draft tools (without mocks where possible)."""

    def test_all_invalid_input_returns_empty(self) -> None:
        """Test that completely invalid input returns empty data."""
        result = get_draft_picks_impl([1990, 1991])

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 0
        assert result.warning is not None
