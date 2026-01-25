"""Tests for the reference data retrieval tools.

This module tests the get_player_ids, get_team_descriptions, get_officials,
and get_contracts tools using mocked data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from fast_nfl_mcp.constants import DEFAULT_MAX_ROWS
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.tools.reference import (
    get_contracts_impl,
    get_officials_impl,
    get_player_ids_impl,
    get_team_descriptions_impl,
)


class TestGetPlayerIds:
    """Tests for the get_player_ids tool."""

    @pytest.fixture
    def sample_player_ids_df(self) -> pd.DataFrame:
        """Create a sample player IDs DataFrame."""
        return pd.DataFrame(
            {
                "gsis_id": ["00-0033873", "00-0036945", "00-0036389"],
                "name": ["Patrick Mahomes", "Josh Allen", "Brock Purdy"],
                "espn_id": [3139477, 3918298, 4361741],
                "yahoo_id": [30123, 31845, 33000],
                "sleeper_id": [4046, 6744, 8155],
            }
        )

    @pytest.fixture
    def large_player_ids_df(self) -> pd.DataFrame:
        """Create a large player IDs DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "gsis_id": [f"00-00{i:05d}" for i in range(150)],
                "name": [f"Player {i}" for i in range(150)],
                "espn_id": list(range(150)),
            }
        )

    def test_successful_fetch(self, sample_player_ids_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.return_value = sample_player_ids_df

            result = get_player_ids_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_truncation_at_max_rows(self, large_player_ids_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.return_value = large_player_ids_df

            result = get_player_ids_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_pagination_with_offset(self, large_player_ids_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.return_value = large_player_ids_df

            result = get_player_ids_impl(offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            # First row should have index 10 (skipped first 10)
            assert result.data[0]["gsis_id"] == "00-0000010"

    def test_pagination_with_limit(self, large_player_ids_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.return_value = large_player_ids_df

            result = get_player_ids_impl(limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5
            assert result.metadata.row_count == 5

    def test_pagination_with_offset_and_limit(
        self, large_player_ids_df: pd.DataFrame
    ) -> None:
        """Test pagination with both offset and limit."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.return_value = large_player_ids_df

            result = get_player_ids_impl(offset=20, limit=15)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 15
            assert result.data[0]["gsis_id"] == "00-0000020"

    def test_empty_dataframe_returns_success(self) -> None:
        """Test that empty DataFrame returns success with warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.return_value = pd.DataFrame()

            result = get_player_ids_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.side_effect = Exception("Network timeout")

            result = get_player_ids_impl()

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_player_ids_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_import:
            mock_import.return_value = sample_player_ids_df

            result = get_player_ids_impl()

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "gsis_id" in result.metadata.columns
            assert "name" in result.metadata.columns


class TestGetTeamDescriptions:
    """Tests for the get_team_descriptions tool."""

    @pytest.fixture
    def sample_team_desc_df(self) -> pd.DataFrame:
        """Create a sample team descriptions DataFrame."""
        return pd.DataFrame(
            {
                "team_abbr": ["KC", "SF", "DET"],
                "team_name": [
                    "Kansas City Chiefs",
                    "San Francisco 49ers",
                    "Detroit Lions",
                ],
                "team_conf": ["AFC", "NFC", "NFC"],
                "team_division": ["West", "West", "North"],
                "team_color": ["#E31837", "#AA0000", "#0076B6"],
            }
        )

    def test_successful_fetch(self, sample_team_desc_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_import:
            mock_import.return_value = sample_team_desc_df

            result = get_team_descriptions_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_pagination_with_offset(self, sample_team_desc_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_import:
            mock_import.return_value = sample_team_desc_df

            result = get_team_descriptions_impl(offset=1)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 2
            assert result.data[0]["team_abbr"] == "SF"

    def test_pagination_with_limit(self, sample_team_desc_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_import:
            mock_import.return_value = sample_team_desc_df

            result = get_team_descriptions_impl(limit=2)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 2

    def test_empty_dataframe_returns_success(self) -> None:
        """Test that empty DataFrame returns success with warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_import:
            mock_import.return_value = pd.DataFrame()

            result = get_team_descriptions_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_import:
            mock_import.side_effect = Exception("Network error")

            result = get_team_descriptions_impl()

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_team_desc_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_import:
            mock_import.return_value = sample_team_desc_df

            result = get_team_descriptions_impl()

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "team_abbr" in result.metadata.columns
            assert "team_name" in result.metadata.columns


class TestGetOfficials:
    """Tests for the get_officials tool."""

    @pytest.fixture
    def sample_officials_df(self) -> pd.DataFrame:
        """Create a sample officials DataFrame."""
        return pd.DataFrame(
            {
                "game_id": ["2024_01_KC_DET", "2024_01_KC_DET", "2024_01_SF_PIT"],
                "official_name": ["John Smith", "Jane Doe", "Mike Johnson"],
                "official_position": ["Referee", "Umpire", "Referee"],
            }
        )

    @pytest.fixture
    def large_officials_df(self) -> pd.DataFrame:
        """Create a large officials DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "game_id": [f"2024_{i:02d}_KC_DET" for i in range(150)],
                "official_name": [f"Official {i}" for i in range(150)],
                "official_position": ["Referee"] * 150,
            }
        )

    def test_successful_fetch(self, sample_officials_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_import:
            mock_import.return_value = sample_officials_df

            result = get_officials_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_truncation_at_max_rows(self, large_officials_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_import:
            mock_import.return_value = large_officials_df

            result = get_officials_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_pagination_with_offset(self, large_officials_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_import:
            mock_import.return_value = large_officials_df

            result = get_officials_impl(offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert result.data[0]["official_name"] == "Official 10"

    def test_pagination_with_limit(self, large_officials_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_import:
            mock_import.return_value = large_officials_df

            result = get_officials_impl(limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5

    def test_empty_dataframe_returns_success(self) -> None:
        """Test that empty DataFrame returns success with warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_import:
            mock_import.return_value = pd.DataFrame()

            result = get_officials_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_import:
            mock_import.side_effect = Exception("Connection refused")

            result = get_officials_impl()

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_officials_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_import:
            mock_import.return_value = sample_officials_df

            result = get_officials_impl()

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "game_id" in result.metadata.columns
            assert "official_name" in result.metadata.columns


class TestGetContracts:
    """Tests for the get_contracts tool."""

    @pytest.fixture
    def sample_contracts_df(self) -> pd.DataFrame:
        """Create a sample contracts DataFrame."""
        return pd.DataFrame(
            {
                "player": ["Patrick Mahomes", "Josh Allen", "Brock Purdy"],
                "team": ["KC", "BUF", "SF"],
                "value": [450000000, 258000000, 1200000],
                "apy": [45000000, 43000000, 400000],
                "guaranteed": [141481905, 150000000, 500000],
                "years": [10, 6, 4],
            }
        )

    @pytest.fixture
    def large_contracts_df(self) -> pd.DataFrame:
        """Create a large contracts DataFrame for truncation testing."""
        return pd.DataFrame(
            {
                "player": [f"Player {i}" for i in range(150)],
                "team": ["KC"] * 150,
                "value": [1000000] * 150,
                "apy": [250000] * 150,
            }
        )

    def test_successful_fetch(self, sample_contracts_df: pd.DataFrame) -> None:
        """Test successful data fetch returns SuccessResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.return_value = sample_contracts_df

            result = get_contracts_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 3
            assert result.metadata.row_count == 3

    def test_truncation_at_max_rows(self, large_contracts_df: pd.DataFrame) -> None:
        """Test that results are truncated at DEFAULT_MAX_ROWS."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.return_value = large_contracts_df

            result = get_contracts_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == DEFAULT_MAX_ROWS
            assert result.metadata.truncated is True
            assert result.metadata.total_available == 150

    def test_pagination_with_offset(self, large_contracts_df: pd.DataFrame) -> None:
        """Test that offset skips rows for pagination."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.return_value = large_contracts_df

            result = get_contracts_impl(offset=10)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert result.data[0]["player"] == "Player 10"

    def test_pagination_with_limit(self, large_contracts_df: pd.DataFrame) -> None:
        """Test that limit controls number of rows returned."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.return_value = large_contracts_df

            result = get_contracts_impl(limit=5)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 5

    def test_pagination_with_offset_and_limit(
        self, large_contracts_df: pd.DataFrame
    ) -> None:
        """Test pagination with both offset and limit."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.return_value = large_contracts_df

            result = get_contracts_impl(offset=20, limit=15)

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 15
            assert result.data[0]["player"] == "Player 20"

    def test_empty_dataframe_returns_success(self) -> None:
        """Test that empty DataFrame returns success with warning."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.return_value = pd.DataFrame()

            result = get_contracts_impl()

            assert isinstance(result, SuccessResponse)
            assert result.success is True
            assert len(result.data) == 0
            assert result.warning is not None

    def test_network_error_handling(self) -> None:
        """Test that network errors return ErrorResponse."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.side_effect = Exception("Timeout error")

            result = get_contracts_impl()

            assert isinstance(result, ErrorResponse)
            assert result.success is False
            assert "Error fetching" in result.error

    def test_columns_in_metadata(self, sample_contracts_df: pd.DataFrame) -> None:
        """Test that column names are included in metadata."""
        with patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_import:
            mock_import.return_value = sample_contracts_df

            result = get_contracts_impl()

            assert isinstance(result, SuccessResponse)
            assert result.metadata.columns is not None
            assert "player" in result.metadata.columns
            assert "value" in result.metadata.columns


class TestReferenceToolsIntegration:
    """Integration tests for reference data tools."""

    def test_all_tools_return_consistent_format(self) -> None:
        """Test that all reference tools return consistent response format."""
        # Create mock DataFrames for each tool
        mock_player_ids = pd.DataFrame({"gsis_id": ["00-0000001"], "name": ["Test"]})
        mock_team_desc = pd.DataFrame({"team_abbr": ["KC"], "team_name": ["Chiefs"]})
        mock_officials = pd.DataFrame(
            {"game_id": ["2024_01"], "official_name": ["Test"]}
        )
        mock_contracts = pd.DataFrame({"player": ["Test"], "value": [1000000]})

        with (
            patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_ids,
            patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_desc,
            patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_off,
            patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_con,
        ):
            mock_ids.return_value = mock_player_ids
            mock_desc.return_value = mock_team_desc
            mock_off.return_value = mock_officials
            mock_con.return_value = mock_contracts

            results = [
                get_player_ids_impl(),
                get_team_descriptions_impl(),
                get_officials_impl(),
                get_contracts_impl(),
            ]

            for result in results:
                assert isinstance(result, SuccessResponse)
                assert result.success is True
                assert hasattr(result, "data")
                assert hasattr(result, "metadata")
                assert result.metadata.columns is not None

    def test_no_required_parameters(self) -> None:
        """Test that all tools work without any parameters."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        with (
            patch("fast_nfl_mcp.schema_manager.nfl.import_ids") as mock_ids,
            patch("fast_nfl_mcp.schema_manager.nfl.import_team_desc") as mock_desc,
            patch("fast_nfl_mcp.schema_manager.nfl.import_officials") as mock_off,
            patch("fast_nfl_mcp.schema_manager.nfl.import_contracts") as mock_con,
        ):
            mock_ids.return_value = mock_df
            mock_desc.return_value = mock_df
            mock_off.return_value = mock_df
            mock_con.return_value = mock_df

            # All should work without arguments
            assert get_player_ids_impl().success is True
            assert get_team_descriptions_impl().success is True
            assert get_officials_impl().success is True
            assert get_contracts_impl().success is True
