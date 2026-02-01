"""Tests for the Kaggle data fetcher module.

This module tests the KaggleFetcher class including authentication checking,
data loading, caching, and error handling using mocked data.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fast_nfl_mcp.kaggle_fetcher import (
    KaggleAuthError,
    KaggleCompetitionError,
    KaggleFetcher,
    _convert_dataframe_to_records,
    fetch_bdb_data,
    get_fetcher,
)
from fast_nfl_mcp.models import ErrorResponse, SuccessResponse


class TestKaggleFetcherAuth:
    """Tests for KaggleFetcher authentication checking."""

    def test_auth_missing_kaggle_json(self, tmp_path: Path) -> None:
        """Test that missing kaggle.json raises KaggleAuthError."""
        fetcher = KaggleFetcher()
        fetcher._auth_checked = False

        with patch.object(Path, "home", return_value=tmp_path):
            with pytest.raises(KaggleAuthError) as exc_info:
                fetcher._check_auth()

            assert "kaggle.json" in str(exc_info.value)
            assert "https://www.kaggle.com/settings" in str(exc_info.value)

    def test_auth_kaggle_json_exists(self, tmp_path: Path) -> None:
        """Test that existing kaggle.json passes auth check."""
        # Create the kaggle directory and file
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text('{"username": "test"}')

        fetcher = KaggleFetcher()
        fetcher._auth_checked = False

        with patch.object(Path, "home", return_value=tmp_path):
            # Should not raise
            fetcher._check_auth()
            assert fetcher._auth_valid is True

    def test_auth_cached_invalid(self) -> None:
        """Test that cached invalid auth raises error without rechecking."""
        fetcher = KaggleFetcher()
        fetcher._auth_checked = True
        fetcher._auth_valid = False

        with pytest.raises(KaggleAuthError):
            fetcher._check_auth()

    def test_auth_cached_valid(self) -> None:
        """Test that cached valid auth passes without rechecking."""
        fetcher = KaggleFetcher()
        fetcher._auth_checked = True
        fetcher._auth_valid = True

        # Should not raise
        fetcher._check_auth()


class TestKaggleFetcherDownload:
    """Tests for KaggleFetcher data downloading."""

    def test_download_forbidden_raises_competition_error(self) -> None:
        """Test that 403 error provides helpful message about rules."""
        fetcher = KaggleFetcher()
        fetcher._auth_checked = True
        fetcher._auth_valid = True

        mock_kagglehub = MagicMock()
        mock_kagglehub.competition_download.side_effect = Exception(
            "403 Forbidden: Access denied"
        )

        with patch.dict("sys.modules", {"kagglehub": mock_kagglehub}):
            with pytest.raises(KaggleCompetitionError) as exc_info:
                fetcher._download_competition_data()

            assert "accept the competition rules" in str(exc_info.value)
            assert "rules" in str(exc_info.value)

    def test_download_not_found_raises_competition_error(self) -> None:
        """Test that 404 error provides helpful message."""
        fetcher = KaggleFetcher()
        fetcher._auth_checked = True
        fetcher._auth_valid = True

        mock_kagglehub = MagicMock()
        mock_kagglehub.competition_download.side_effect = Exception("404 Not Found")

        with patch.dict("sys.modules", {"kagglehub": mock_kagglehub}):
            with pytest.raises(KaggleCompetitionError) as exc_info:
                fetcher._download_competition_data()

            assert "not found" in str(exc_info.value)

    def test_download_unauthorized_raises_auth_error(self) -> None:
        """Test that 401 error raises auth error."""
        fetcher = KaggleFetcher()
        fetcher._auth_checked = True
        fetcher._auth_valid = True

        mock_kagglehub = MagicMock()
        mock_kagglehub.competition_download.side_effect = Exception("401 Unauthorized")

        with patch.dict("sys.modules", {"kagglehub": mock_kagglehub}):
            with pytest.raises(KaggleAuthError) as exc_info:
                fetcher._download_competition_data()

            assert "authentication failed" in str(exc_info.value)

    def test_download_caches_path(self, tmp_path: Path) -> None:
        """Test that download path is cached after first call."""
        fetcher = KaggleFetcher()
        fetcher._auth_checked = True
        fetcher._auth_valid = True

        mock_kagglehub = MagicMock()
        mock_kagglehub.competition_download.return_value = str(tmp_path)

        with patch.dict("sys.modules", {"kagglehub": mock_kagglehub}):
            # First call
            path1 = fetcher._download_competition_data()

            # Second call should use cache
            path2 = fetcher._download_competition_data()

            assert path1 == path2
            # Should only call download once
            mock_kagglehub.competition_download.assert_called_once()


class TestKaggleFetcherLoadCsv:
    """Tests for KaggleFetcher CSV loading."""

    def test_load_csv_caches_dataframe(self, tmp_path: Path) -> None:
        """Test that loaded DataFrames are cached."""
        fetcher = KaggleFetcher()
        fetcher._data_path = tmp_path

        # Create a test CSV
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_df.to_csv(tmp_path / "test.csv", index=False)

        # First load
        df1 = fetcher._load_csv("test.csv")

        # Second load should use cache
        df2 = fetcher._load_csv("test.csv")

        # Both should be the same object
        assert df1 is df2
        assert len(df1) == 3

    def test_load_csv_file_not_found(self, tmp_path: Path) -> None:
        """Test that missing file raises KaggleCompetitionError."""
        fetcher = KaggleFetcher()
        fetcher._data_path = tmp_path

        with pytest.raises(KaggleCompetitionError) as exc_info:
            fetcher._load_csv("nonexistent.csv")

        assert "not found" in str(exc_info.value)


class TestKaggleFetcherDataMethods:
    """Tests for KaggleFetcher data retrieval methods."""

    @pytest.fixture
    def mock_fetcher(self, tmp_path: Path) -> KaggleFetcher:
        """Create a KaggleFetcher with mocked data for BDB 2026 structure."""
        fetcher = KaggleFetcher()
        fetcher._data_path = tmp_path

        # Create a subdirectory like the real competition data
        data_subdir = tmp_path / "competition_data"
        data_subdir.mkdir()

        # Create supplementary_data.csv (contains both games and plays info)
        supplementary_df = pd.DataFrame(
            {
                "game_id": [2023090700, 2023090700, 2023091000],
                "season": [2023, 2023, 2023],
                "week": [1, 1, 1],
                "game_date": ["09/07/2023", "09/07/2023", "09/10/2023"],
                "game_time_eastern": ["20:20:00", "20:20:00", "13:00:00"],
                "home_team_abbr": ["KC", "KC", "BUF"],
                "visitor_team_abbr": ["DET", "DET", "LA"],
                "play_id": [1, 2, 1],
                "play_description": ["Play 1", "Play 2", "Play 3"],
                "quarter": [1, 1, 1],
            }
        )
        supplementary_df.to_csv(data_subdir / "supplementary_data.csv", index=False)

        # Create train directory for tracking data
        train_dir = data_subdir / "train"
        train_dir.mkdir()

        tracking_df = pd.DataFrame(
            {
                "game_id": [2023090700] * 10,
                "play_id": [1] * 10,
                "nfl_id": [43290] * 10,
                "player_name": ["Patrick Mahomes"] * 10,
                "player_position": ["QB"] * 10,
                "player_height": ["6-2"] * 10,
                "player_weight": [225] * 10,
                "player_birth_date": ["1995-09-17"] * 10,
                "frame_id": list(range(1, 11)),
                "x": [10.0 + i for i in range(10)],
                "y": [20.0] * 10,
                "s": [5.0] * 10,
            }
        )
        tracking_df.to_csv(train_dir / "input_2023_w01.csv", index=False)

        return fetcher

    def test_get_games(self, mock_fetcher: KaggleFetcher) -> None:
        """Test loading games data."""
        df = mock_fetcher.get_games()
        assert len(df) == 2  # 2 unique games
        assert "game_id" in df.columns
        assert "home_team_abbr" in df.columns

    def test_get_plays(self, mock_fetcher: KaggleFetcher) -> None:
        """Test loading plays data."""
        df = mock_fetcher.get_plays()
        assert len(df) == 3
        assert "play_id" in df.columns
        assert "play_description" in df.columns

    def test_get_players(self, mock_fetcher: KaggleFetcher) -> None:
        """Test loading players data."""
        df = mock_fetcher.get_players()
        assert len(df) == 1  # 1 unique player in tracking data
        assert "nfl_id" in df.columns
        assert "player_name" in df.columns

    def test_get_tracking_valid_week(self, mock_fetcher: KaggleFetcher) -> None:
        """Test loading tracking data for valid week."""
        df = mock_fetcher.get_tracking(1)
        assert len(df) == 10
        assert "frame_id" in df.columns
        assert "x" in df.columns

    def test_get_tracking_invalid_week(self, mock_fetcher: KaggleFetcher) -> None:
        """Test that invalid week raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            mock_fetcher.get_tracking(19)  # Week 19 is invalid (max is 18)

        assert "Invalid week" in str(exc_info.value)
        assert "19" in str(exc_info.value)

    def test_clear_cache(self, mock_fetcher: KaggleFetcher) -> None:
        """Test that clear_cache empties the cache."""
        # Load some data to populate cache
        # Note: games and plays both use supplementary_data.csv (1 file)
        # tracking uses input_2023_w01.csv (separate file)
        mock_fetcher.get_games()
        mock_fetcher.get_tracking(1)
        assert len(mock_fetcher._cache) == 2

        # Clear cache
        mock_fetcher.clear_cache()
        assert len(mock_fetcher._cache) == 0


class TestGetFetcher:
    """Tests for the get_fetcher singleton function."""

    def test_get_fetcher_returns_singleton(self) -> None:
        """Test that get_fetcher returns the same instance."""
        # Reset the singleton
        import fast_nfl_mcp.kaggle_fetcher as module

        module._fetcher = None

        fetcher1 = get_fetcher()
        fetcher2 = get_fetcher()

        assert fetcher1 is fetcher2


class TestConvertDataframeToRecords:
    """Tests for the _convert_dataframe_to_records function."""

    def test_empty_dataframe(self) -> None:
        """Test that empty DataFrame returns empty list."""
        df = pd.DataFrame()
        records, columns = _convert_dataframe_to_records(df)
        assert records == []
        assert columns == []

    def test_none_dataframe(self) -> None:
        """Test that None DataFrame returns empty list."""
        records, columns = _convert_dataframe_to_records(None)  # type: ignore
        assert records == []
        assert columns == []

    def test_converts_nan_to_none(self) -> None:
        """Test that NaN values are converted to None."""
        df = pd.DataFrame({"col": [1.0, None, 3.0]})
        records, _ = _convert_dataframe_to_records(df)
        assert records[1]["col"] is None

    def test_converts_numpy_types(self) -> None:
        """Test that numpy types are converted to Python native."""
        import numpy as np

        df = pd.DataFrame({"int_col": np.array([1, 2, 3], dtype=np.int64)})
        records, _ = _convert_dataframe_to_records(df)
        assert isinstance(records[0]["int_col"], int)


class TestFetchBdbData:
    """Tests for the fetch_bdb_data function."""

    @pytest.fixture
    def mock_fetcher_data(self, tmp_path: Path) -> None:
        """Set up mocked KaggleFetcher with test data for BDB 2026 structure."""
        import fast_nfl_mcp.kaggle_fetcher as module

        fetcher = KaggleFetcher()
        fetcher._data_path = tmp_path

        # Create a subdirectory like the real competition data
        data_subdir = tmp_path / "competition_data"
        data_subdir.mkdir()

        # Create supplementary_data.csv
        # 3 unique games: 2 with week=1, 1 with week=2
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
            }
        )
        supplementary_df.to_csv(data_subdir / "supplementary_data.csv", index=False)

        # Create train directory for tracking data
        train_dir = data_subdir / "train"
        train_dir.mkdir()

        tracking_df = pd.DataFrame(
            {
                "game_id": [2023090700] * 100,
                "play_id": [1] * 100,
                "nfl_id": [43290] * 100,
                "player_name": ["Patrick Mahomes"] * 100,
                "player_position": ["QB"] * 100,
                "player_height": ["6-2"] * 100,
                "player_weight": [225] * 100,
                "player_birth_date": ["1995-09-17"] * 100,
                "frame_id": list(range(1, 101)),
                "x": [10.0 + i for i in range(100)],
            }
        )
        tracking_df.to_csv(train_dir / "input_2023_w01.csv", index=False)

        module._fetcher = fetcher

    def test_fetch_games_success(self, mock_fetcher_data: None) -> None:
        """Test successful games fetch."""
        result = fetch_bdb_data("games")

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(result.data) == 3  # 3 unique games

    def test_fetch_with_filters(self, mock_fetcher_data: None) -> None:
        """Test fetch with filters."""
        result = fetch_bdb_data("games", filters={"week": [1]})

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 2
        for row in result.data:
            assert row["week"] == 1

    def test_fetch_with_columns(self, mock_fetcher_data: None) -> None:
        """Test fetch with column selection."""
        result = fetch_bdb_data("games", columns=["game_id", "week"])

        assert isinstance(result, SuccessResponse)
        assert result.metadata.columns == ["game_id", "week"]
        for row in result.data:
            assert set(row.keys()) == {"game_id", "week"}

    def test_fetch_with_pagination(self, mock_fetcher_data: None) -> None:
        """Test fetch with offset and limit."""
        result = fetch_bdb_data("games", offset=1, limit=1)

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 1
        assert result.metadata.total_available == 3

    def test_fetch_tracking_requires_week(self, mock_fetcher_data: None) -> None:
        """Test that tracking fetch requires week parameter."""
        result = fetch_bdb_data("tracking")

        assert isinstance(result, ErrorResponse)
        assert "week parameter is required" in result.error

    def test_fetch_tracking_success(self, mock_fetcher_data: None) -> None:
        """Test successful tracking fetch with week."""
        result = fetch_bdb_data("tracking", week=1, limit=10)

        assert isinstance(result, SuccessResponse)
        assert len(result.data) == 10

    def test_fetch_tracking_default_limit(self, mock_fetcher_data: None) -> None:
        """Test that tracking has lower default limit."""
        result = fetch_bdb_data("tracking", week=1)

        assert isinstance(result, SuccessResponse)
        # Default limit for tracking is 50
        assert len(result.data) == 50
        assert result.metadata.truncated is True

    def test_fetch_unknown_data_type(self, mock_fetcher_data: None) -> None:
        """Test that unknown data type returns error."""
        result = fetch_bdb_data("unknown")

        assert isinstance(result, ErrorResponse)
        assert "Unknown data type" in result.error

    def test_fetch_invalid_columns(self, mock_fetcher_data: None) -> None:
        """Test that invalid columns return error."""
        result = fetch_bdb_data("games", columns=["nonexistent"])

        assert isinstance(result, ErrorResponse)
        assert "None of the requested columns exist" in result.error

    def test_fetch_empty_columns_list(self, mock_fetcher_data: None) -> None:
        """Test that empty columns list returns error."""
        result = fetch_bdb_data("games", columns=[])

        assert isinstance(result, ErrorResponse)
        assert "Empty columns list" in result.error

    def test_fetch_auth_error(self) -> None:
        """Test that auth error is properly returned."""
        import fast_nfl_mcp.kaggle_fetcher as module

        # Create fetcher with auth failure
        fetcher = KaggleFetcher()
        fetcher._auth_checked = True
        fetcher._auth_valid = False
        module._fetcher = fetcher

        result = fetch_bdb_data("games")

        assert isinstance(result, ErrorResponse)
        assert "authentication" in result.error.lower()

    def test_fetch_invalid_filter_columns_warning(
        self, mock_fetcher_data: None
    ) -> None:
        """Test that invalid filter columns produce warning."""
        result = fetch_bdb_data("games", filters={"nonexistent": ["value"]})

        assert isinstance(result, SuccessResponse)
        # All rows returned since filter was ignored
        assert len(result.data) == 3
        assert result.warning is not None
        assert "nonexistent" in result.warning
