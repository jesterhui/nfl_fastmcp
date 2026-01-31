"""Tests for the miscellaneous NFL data retrieval tools.

This module tests the get_snap_counts, get_injuries, get_depth_charts,
get_combine_data, and get_qbr tools using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.constants import (
    DEFAULT_MAX_ROWS,
    MAX_SEASONS_COMBINE,
    MAX_SEASONS_DEPTH_CHARTS,
    MAX_SEASONS_INJURIES,
    MAX_SEASONS_QBR,
    MAX_SEASONS_SNAP_COUNTS,
)
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.misc import (
    get_combine_data_impl,
    get_depth_charts_impl,
    get_injuries_impl,
    get_qbr_impl,
    get_snap_counts_impl,
)


class TestGetSnapCounts:
    """Tests for the get_snap_counts tool."""

    @pytest.fixture
    def sample_snap_counts_df(self) -> pd.DataFrame:
        """Create a sample snap counts DataFrame."""
        return pd.DataFrame(
            {
                "player": ["Patrick Mahomes", "Travis Kelce", "Chris Jones"],
                "team": ["KC", "KC", "KC"],
                "position": ["QB", "TE", "DT"],
                "week": [1, 1, 1],
                "offense_snaps": [65, 58, 0],
                "offense_pct": [100.0, 89.2, 0.0],
                "defense_snaps": [0, 0, 45],
                "defense_pct": [0.0, 0.0, 68.2],
                "st_snaps": [0, 2, 5],
                "st_pct": [0.0, 6.9, 17.2],
            }
        )

    @pytest.fixture
    def large_snap_counts_df(self) -> pd.DataFrame:
        """Create a large snap counts DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player": [f"Player {i}" for i in range(150)],
                "team": ["KC"] * 150,
                "position": ["WR"] * 150,
                "offense_snaps": [50] * 150,
                "offense_pct": [75.0] * 150,
            }
        )

    def test_successful_fetch(self, sample_snap_counts_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.return_value = sample_snap_counts_df

            result = get_snap_counts_impl([2024], ["player", "team", "offense_pct"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3

    def test_season_limit_enforced(self, sample_snap_counts_df: pd.DataFrame) -> None:
        """Test that season limit of 5 is enforced."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.return_value = sample_snap_counts_df

            # Request more seasons than allowed
            seasons = [2020, 2021, 2022, 2023, 2024, 2025]
            result = get_snap_counts_impl(seasons, ["player", "team"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Limited to" in result.warning
            # Verify the mock was called with truncated seasons
            call_args = mock_import.call_args[0][0]
            assert len(call_args) <= MAX_SEASONS_SNAP_COUNTS

    def test_truncation_at_max_rows(self, large_snap_counts_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.return_value = large_snap_counts_df

            result = get_snap_counts_impl([2024], ["player", "team"])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_pagination_with_offset(self, large_snap_counts_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.return_value = large_snap_counts_df

            result = get_snap_counts_impl([2024], ["player", "team"], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.data[0]["player"] == "Player 10"

    def test_pagination_with_limit(self, large_snap_counts_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.return_value = large_snap_counts_df

            result = get_snap_counts_impl([2024], ["player", "team"], limit=5)

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 5

    def test_filters_applied(self, sample_snap_counts_df: pd.DataFrame) -> None:
        """Test that filters are applied correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.return_value = sample_snap_counts_df

            result = get_snap_counts_impl(
                [2024], ["player", "position"], filters={"position": "QB"}
            )

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 1
            assert result.data[0]["player"] == "Patrick Mahomes"

    def test_empty_seasons_returns_warning(self) -> None:
        """Test that empty seasons returns success with warning."""
        result = get_snap_counts_impl([], ["player", "team"])

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 0
        assert result.warning is not None

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_snap_counts_impl([2024], ["player", "team"])

            assert isinstance(result, ErrorResponse)
            assert result.success is False


class TestGetInjuries:
    """Tests for the get_injuries tool."""

    @pytest.fixture
    def sample_injuries_df(self) -> pd.DataFrame:
        """Create a sample injuries DataFrame."""
        return pd.DataFrame(
            {
                "player": ["Player A", "Player B", "Player C"],
                "team": ["KC", "KC", "SF"],
                "position": ["WR", "RB", "QB"],
                "week": [1, 1, 1],
                "report_primary_injury": ["Knee", "Ankle", "Shoulder"],
                "report_status": ["Questionable", "Out", "Probable"],
                "practice_status": ["Limited", "DNP", "Full"],
            }
        )

    @pytest.fixture
    def large_injuries_df(self) -> pd.DataFrame:
        """Create a large injuries DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player": [f"Player {i}" for i in range(150)],
                "team": ["KC"] * 150,
                "report_status": ["Questionable"] * 150,
            }
        )

    def test_successful_fetch(self, sample_injuries_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_injuries") as mock_import:
            mock_import.return_value = sample_injuries_df

            result = get_injuries_impl([2024], ["player", "report_status"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3

    def test_season_limit_enforced(self, sample_injuries_df: pd.DataFrame) -> None:
        """Test that season limit of 5 is enforced."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_injuries") as mock_import:
            mock_import.return_value = sample_injuries_df

            seasons = [2020, 2021, 2022, 2023, 2024, 2025]
            result = get_injuries_impl(seasons, ["player", "team"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            call_args = mock_import.call_args[0][0]
            assert len(call_args) <= MAX_SEASONS_INJURIES

    def test_truncation_at_max_rows(self, large_injuries_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_injuries") as mock_import:
            mock_import.return_value = large_injuries_df

            result = get_injuries_impl([2024], ["player", "team"])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True

    def test_filters_applied(self, sample_injuries_df: pd.DataFrame) -> None:
        """Test that filters are applied correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_injuries") as mock_import:
            mock_import.return_value = sample_injuries_df

            result = get_injuries_impl(
                [2024], ["player", "report_status"], filters={"report_status": "Out"}
            )

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 1
            assert result.data[0]["player"] == "Player B"

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_injuries") as mock_import:
            mock_import.side_effect = Exception("Connection refused")

            result = get_injuries_impl([2024], ["player", "team"])

            assert isinstance(result, ErrorResponse)
            assert result.success is False


class TestGetDepthCharts:
    """Tests for the get_depth_charts tool."""

    @pytest.fixture
    def sample_depth_charts_df(self) -> pd.DataFrame:
        """Create a sample depth charts DataFrame."""
        return pd.DataFrame(
            {
                "club_code": ["KC", "KC", "KC"],
                "full_name": ["Patrick Mahomes", "Carson Wentz", "Chris Oladokun"],
                "position": ["QB", "QB", "QB"],
                "depth_team": ["Offense", "Offense", "Offense"],
                "depth_position": [1, 2, 3],
            }
        )

    @pytest.fixture
    def large_depth_charts_df(self) -> pd.DataFrame:
        """Create a large depth charts DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "club_code": ["KC"] * 150,
                "full_name": [f"Player {i}" for i in range(150)],
                "position": ["WR"] * 150,
            }
        )

    def test_successful_fetch(self, sample_depth_charts_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_depth_charts"
        ) as mock_import:
            mock_import.return_value = sample_depth_charts_df

            result = get_depth_charts_impl([2024], ["full_name", "position"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3

    def test_season_limit_enforced(self, sample_depth_charts_df: pd.DataFrame) -> None:
        """Test that season limit of 5 is enforced."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_depth_charts"
        ) as mock_import:
            mock_import.return_value = sample_depth_charts_df

            seasons = [2020, 2021, 2022, 2023, 2024, 2025]
            result = get_depth_charts_impl(seasons, ["full_name", "position"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            call_args = mock_import.call_args[0][0]
            assert len(call_args) <= MAX_SEASONS_DEPTH_CHARTS

    def test_truncation_at_max_rows(self, large_depth_charts_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_depth_charts"
        ) as mock_import:
            mock_import.return_value = large_depth_charts_df

            result = get_depth_charts_impl([2024], ["full_name", "position"])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True

    def test_filters_applied(self, sample_depth_charts_df: pd.DataFrame) -> None:
        """Test that filters are applied correctly."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_depth_charts"
        ) as mock_import:
            mock_import.return_value = sample_depth_charts_df

            result = get_depth_charts_impl(
                [2024], ["full_name", "depth_position"], filters={"depth_position": [1]}
            )

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 1
            assert result.data[0]["full_name"] == "Patrick Mahomes"

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_depth_charts"
        ) as mock_import:
            mock_import.side_effect = Exception("Network error")

            result = get_depth_charts_impl([2024], ["full_name", "position"])

            assert isinstance(result, ErrorResponse)
            assert result.success is False


class TestGetCombineData:
    """Tests for the get_combine_data tool."""

    @pytest.fixture
    def sample_combine_df(self) -> pd.DataFrame:
        """Create a sample combine data DataFrame."""
        return pd.DataFrame(
            {
                "player_name": [
                    "Caleb Williams",
                    "Marvin Harrison Jr.",
                    "Malik Nabers",
                ],
                "pos": ["QB", "WR", "WR"],
                "school": ["USC", "Ohio State", "LSU"],
                "ht": ["6-1", "6-4", "6-0"],
                "wt": [214, 209, 200],
                "forty": [4.62, 4.38, 4.35],
                "vertical": [32.5, 36.0, 39.0],
                "bench": [None, 15, 12],
            }
        )

    @pytest.fixture
    def large_combine_df(self) -> pd.DataFrame:
        """Create a large combine data DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player_name": [f"Player {i}" for i in range(150)],
                "pos": ["WR"] * 150,
                "forty": [4.50] * 150,
            }
        )

    def test_successful_fetch(self, sample_combine_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_combine_data"
        ) as mock_import:
            mock_import.return_value = sample_combine_df

            result = get_combine_data_impl([2024], ["player_name", "pos", "forty"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3

    def test_season_limit_enforced(self, sample_combine_df: pd.DataFrame) -> None:
        """Test that season limit of 10 is enforced."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_combine_data"
        ) as mock_import:
            mock_import.return_value = sample_combine_df

            seasons = list(range(2015, 2027))  # 12 seasons
            result = get_combine_data_impl(seasons, ["player_name", "pos"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            call_args = mock_import.call_args[0][0]
            assert len(call_args) <= MAX_SEASONS_COMBINE

    def test_truncation_at_max_rows(self, large_combine_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_combine_data"
        ) as mock_import:
            mock_import.return_value = large_combine_df

            result = get_combine_data_impl([2024], ["player_name", "pos"])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True

    def test_filters_applied(self, sample_combine_df: pd.DataFrame) -> None:
        """Test that filters are applied correctly."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_combine_data"
        ) as mock_import:
            mock_import.return_value = sample_combine_df

            result = get_combine_data_impl(
                [2024], ["player_name", "pos"], filters={"pos": "QB"}
            )

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 1
            assert result.data[0]["player_name"] == "Caleb Williams"

    def test_pagination_with_offset_and_limit(
        self, large_combine_df: pd.DataFrame
    ) -> None:
        """Test pagination with both offset and limit."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_combine_data"
        ) as mock_import:
            mock_import.return_value = large_combine_df

            result = get_combine_data_impl(
                [2024], ["player_name", "pos"], offset=20, limit=15
            )

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 15
            assert result.data[0]["player_name"] == "Player 20"

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch(
            "fast_nfl_mcp.schema_manager.nfl.import_combine_data"
        ) as mock_import:
            mock_import.side_effect = Exception("Timeout error")

            result = get_combine_data_impl([2024], ["player_name", "pos"])

            assert isinstance(result, ErrorResponse)
            assert result.success is False


class TestGetQbr:
    """Tests for the get_qbr tool."""

    @pytest.fixture
    def sample_qbr_df(self) -> pd.DataFrame:
        """Create a sample QBR DataFrame."""
        return pd.DataFrame(
            {
                "player_name": ["Patrick Mahomes", "Josh Allen", "Brock Purdy"],
                "team": ["KC", "BUF", "SF"],
                "qbr_total": [75.2, 68.5, 72.1],
                "pts_added": [45.3, 38.2, 42.1],
                "pass_epa": [0.25, 0.18, 0.22],
                "total_epa": [52.1, 45.3, 48.7],
            }
        )

    @pytest.fixture
    def large_qbr_df(self) -> pd.DataFrame:
        """Create a large QBR DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player_name": [f"QB {i}" for i in range(150)],
                "team": ["KC"] * 150,
                "qbr_total": [65.0] * 150,
            }
        )

    def test_successful_fetch(self, sample_qbr_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.return_value = sample_qbr_df

            result = get_qbr_impl([2024], ["player_name", "team", "qbr_total"])

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3

    def test_season_limit_enforced(self, sample_qbr_df: pd.DataFrame) -> None:
        """Test that season limit of 10 is enforced."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.return_value = sample_qbr_df

            seasons = list(range(2015, 2027))  # 12 seasons
            result = get_qbr_impl(seasons, ["player_name", "qbr_total"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            call_args = mock_import.call_args[0][0]
            assert len(call_args) <= MAX_SEASONS_QBR

    def test_truncation_at_max_rows(self, large_qbr_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.return_value = large_qbr_df

            result = get_qbr_impl([2024], ["player_name", "team"])

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True

    def test_filters_applied(self, sample_qbr_df: pd.DataFrame) -> None:
        """Test that filters are applied correctly."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.return_value = sample_qbr_df

            result = get_qbr_impl(
                [2024], ["player_name", "qbr_total"], filters={"team": "KC"}
            )

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 1
            assert result.data[0]["player_name"] == "Patrick Mahomes"

    def test_pagination_with_offset(self, large_qbr_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.return_value = large_qbr_df

            result = get_qbr_impl([2024], ["player_name", "team"], offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.data[0]["player_name"] == "QB 10"

    def test_pagination_with_limit(self, large_qbr_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.return_value = large_qbr_df

            result = get_qbr_impl([2024], ["player_name", "team"], limit=5)

            assert isinstance(result, SuccessResponse)
            assert len(result.data) == 5

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_qbr_impl([2024], ["player_name", "team"])

            assert isinstance(result, ErrorResponse)
            assert result.success is False

    def test_columns_in_metadata(self, sample_qbr_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_import:
            mock_import.return_value = sample_qbr_df

            result = get_qbr_impl([2024], ["player_name", "qbr_total"])

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player_name" in result.metadata.columns
            assert "qbr_total" in result.metadata.columns


class TestMiscToolsIntegration:
    """Integration tests for miscellaneous data tools."""

    def test_all_tools_accept_same_parameters(self) -> None:
        """Test that all tools accept the same parameter patterns."""
        # Create mock DataFrames
        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with (
            patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_snap,
            patch("fast_nfl_mcp.schema_manager.nfl.import_injuries") as mock_inj,
            patch("fast_nfl_mcp.schema_manager.nfl.import_depth_charts") as mock_depth,
            patch("fast_nfl_mcp.schema_manager.nfl.import_combine_data") as mock_comb,
            patch("fast_nfl_mcp.schema_manager.nfl.import_qbr") as mock_qbr,
        ):
            mock_snap.return_value = mock_df
            mock_inj.return_value = mock_df
            mock_depth.return_value = mock_df
            mock_comb.return_value = mock_df
            mock_qbr.return_value = mock_df

            # All should work with seasons, columns, filters, offset, and limit
            results = [
                get_snap_counts_impl(
                    [2024], ["col1"], filters={"col2": "a"}, offset=0, limit=10
                ),
                get_injuries_impl(
                    [2024], ["col1"], filters={"col2": "a"}, offset=0, limit=10
                ),
                get_depth_charts_impl(
                    [2024], ["col1"], filters={"col2": "a"}, offset=0, limit=10
                ),
                get_combine_data_impl(
                    [2024], ["col1"], filters={"col2": "a"}, offset=0, limit=10
                ),
                get_qbr_impl(
                    [2024], ["col1"], filters={"col2": "a"}, offset=0, limit=10
                ),
            ]

            for result in results:
                assert isinstance(result, SuccessResponse)
                assert result.success is True

    def test_invalid_seasons_handled_gracefully(self) -> None:
        """Test that invalid seasons are handled with warnings."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        with patch("fast_nfl_mcp.schema_manager.nfl.import_snap_counts") as mock_import:
            mock_import.return_value = mock_df

            # Seasons before MIN_SEASON should be filtered out
            result = get_snap_counts_impl([1990, 2024], ["col"])

            assert isinstance(result, SuccessResponse)
            assert result.warning is not None
            assert "Invalid seasons" in result.warning or "1990" in (
                result.warning or ""
            )
