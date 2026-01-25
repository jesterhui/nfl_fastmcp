"""Shared pytest fixtures for Fast NFL MCP tests.

This module provides common fixtures used across test modules,
including mock data, schema managers, and DataFrame builders.
All fixtures are designed to avoid network dependencies.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fast_nfl_mcp.models import ColumnSchema, DatasetSchema
from fast_nfl_mcp.schema_manager import SchemaManager

# =============================================================================
# Sample DataFrames
# =============================================================================


@pytest.fixture
def sample_play_by_play_df() -> pd.DataFrame:
    """Create a sample play-by-play DataFrame for testing."""
    return pd.DataFrame(
        {
            "game_id": ["2024_01_KC_DET", "2024_01_KC_DET", "2024_01_SF_PIT"],
            "play_id": [1, 2, 3],
            "posteam": ["KC", "KC", "SF"],
            "defteam": ["DET", "DET", "PIT"],
            "yards_gained": [5, -2, 15],
            "epa": [0.5, -0.8, 1.2],
            "wpa": [0.02, -0.03, 0.05],
            "play_type": ["pass", "run", "pass"],
            "passer_player_name": ["P.Mahomes", None, "B.Purdy"],
            "receiver_player_name": ["T.Kelce", None, "D.Samuel"],
            "rusher_player_name": [None, "I.Pacheco", None],
        }
    )


@pytest.fixture
def sample_weekly_stats_df() -> pd.DataFrame:
    """Create a sample weekly stats DataFrame for testing."""
    return pd.DataFrame(
        {
            "player_id": ["00-0033873", "00-0036945", "00-0036389"],
            "player_name": ["Patrick Mahomes", "Josh Allen", "Brock Purdy"],
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "team": ["KC", "BUF", "SF"],
            "passing_yards": [320, 280, 310],
            "passing_tds": [3, 2, 4],
            "interceptions": [0, 1, 0],
            "rushing_yards": [25, 45, 15],
            "fantasy_points": [25.5, 22.0, 28.5],
        }
    )


@pytest.fixture
def sample_rosters_df() -> pd.DataFrame:
    """Create a sample rosters DataFrame for testing."""
    return pd.DataFrame(
        {
            "player_id": ["00-0033873", "00-0036945"],
            "player_name": ["Patrick Mahomes", "Josh Allen"],
            "team": ["KC", "BUF"],
            "position": ["QB", "QB"],
            "jersey_number": [15, 17],
            "status": ["ACT", "ACT"],
            "height": ["6-2", "6-5"],
            "weight": [225, 237],
            "college": ["Texas Tech", "Wyoming"],
            "draft_year": [2017, 2018],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Create an empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    """Create a DataFrame with various null patterns."""
    return pd.DataFrame(
        {
            "all_null": [np.nan, np.nan, np.nan],
            "some_null": [1.0, np.nan, 3.0],
            "no_null": [1, 2, 3],
            "string_null": ["a", None, "c"],
        }
    )


@pytest.fixture
def df_with_special_types() -> pd.DataFrame:
    """Create a DataFrame with special data types."""
    return pd.DataFrame(
        {
            "int_col": pd.array([1, 2, 3], dtype="Int64"),
            "float_col": [1.5, 2.5, 3.5],
            "bool_col": [True, False, True],
            "date_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "category_col": pd.Categorical(["A", "B", "A"]),
        }
    )


# =============================================================================
# Column and Dataset Schema Fixtures
# =============================================================================


@pytest.fixture
def sample_column_schema() -> ColumnSchema:
    """Create a sample ColumnSchema for testing."""
    return ColumnSchema(
        name="epa",
        dtype="float64",
        sample_values=[0.5, -0.3, 1.2, -0.8, 0.0],
        null_count=10,
        unique_count=450,
    )


@pytest.fixture
def sample_dataset_schema(sample_column_schema: ColumnSchema) -> DatasetSchema:
    """Create a sample DatasetSchema for testing."""
    columns = [
        sample_column_schema,
        ColumnSchema(name="game_id", dtype="object", sample_values=["2024_01_KC_DET"]),
        ColumnSchema(name="play_id", dtype="int64", sample_values=[1, 2, 3]),
    ]
    return DatasetSchema(
        name="play_by_play",
        description="Play-by-play data with EPA, WPA, and detailed play outcomes",
        columns=columns,
        row_count=50000,
        available_seasons=list(range(1999, 2025)),
    )


# =============================================================================
# SchemaManager Fixtures
# =============================================================================


@pytest.fixture
def empty_schema_manager() -> SchemaManager:
    """Create an empty SchemaManager (not preloaded)."""
    return SchemaManager()


@pytest.fixture
def mock_schema_manager(sample_play_by_play_df: pd.DataFrame) -> SchemaManager:
    """Create a SchemaManager with mocked data preloaded.

    This fixture patches DATASET_DEFINITIONS to use mock loaders,
    then preloads the manager for testing.
    """
    mock_definitions = {
        "play_by_play": (
            lambda _: sample_play_by_play_df,
            "Play-by-play data with EPA, WPA, and detailed play outcomes",
            True,
            2024,
        ),
        "team_descriptions": (
            lambda _: pd.DataFrame({"team": ["KC", "SF"], "name": ["Chiefs", "49ers"]}),
            "Team metadata and information",
            False,
            None,
        ),
    }

    with patch.dict(
        "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
        mock_definitions,
        clear=True,
    ):
        manager = SchemaManager()
        manager.preload_all()
        return manager


@pytest.fixture
def schema_manager_with_failures() -> SchemaManager:
    """Create a SchemaManager with some failed dataset loads."""

    def raise_error(_: object) -> pd.DataFrame:
        raise RuntimeError("Network error")

    mock_definitions = {
        "successful_dataset": (
            lambda _: pd.DataFrame({"col": [1, 2, 3]}),
            "Working dataset",
            True,
            2024,
        ),
        "failed_dataset": (
            raise_error,
            "Broken dataset",
            True,
            2024,
        ),
    }

    with patch.dict(
        "fast_nfl_mcp.schema_manager.DATASET_DEFINITIONS",
        mock_definitions,
        clear=True,
    ):
        manager = SchemaManager()
        manager.preload_all()
        return manager


# =============================================================================
# Mock Context Fixtures (for MCP tools)
# =============================================================================


@pytest.fixture
def mock_mcp_context(mock_schema_manager: SchemaManager) -> MagicMock:
    """Create a mock MCP context with a schema manager."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"schema_manager": mock_schema_manager}
    return ctx


@pytest.fixture
def mock_mcp_context_with_failures(
    schema_manager_with_failures: SchemaManager,
) -> MagicMock:
    """Create a mock MCP context with a schema manager that has failed datasets."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {
        "schema_manager": schema_manager_with_failures
    }
    return ctx


# =============================================================================
# NFL-specific test data
# =============================================================================


@pytest.fixture
def nfl_teams() -> list[str]:
    """Return list of current NFL team abbreviations."""
    return [
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


@pytest.fixture
def sample_seasons() -> list[int]:
    """Return a sample list of NFL seasons for testing."""
    return [2020, 2021, 2022, 2023, 2024]


@pytest.fixture
def sample_weeks() -> list[int]:
    """Return valid week numbers for NFL season."""
    return list(range(1, 19))  # Weeks 1-18


# =============================================================================
# Helper fixtures for assertions
# =============================================================================


@pytest.fixture
def assert_valid_success_response() -> Any:
    """Return a function that asserts a valid SuccessResponse structure."""
    from fast_nfl_mcp.models import SuccessResponse

    def _assert(response: SuccessResponse) -> None:
        assert response.success is True
        assert isinstance(response.data, list)
        assert response.metadata is not None
        assert response.metadata.row_count >= 0

    return _assert


@pytest.fixture
def assert_valid_error_response() -> Any:
    """Return a function that asserts a valid ErrorResponse structure."""
    from fast_nfl_mcp.models import ErrorResponse

    def _assert(response: ErrorResponse) -> None:
        assert response.success is False
        assert response.data == []
        assert response.error is not None
        assert len(response.error) > 0
        assert response.metadata.row_count == 0

    return _assert
