"""Data fetching and schema management for the Fast NFL MCP server.

This package contains modules for fetching NFL data from various sources
and managing dataset schemas.
"""

from fast_nfl_mcp.data.fetcher import DataFetchError, NFLDataPyFetcher
from fast_nfl_mcp.data.kaggle import (
    KaggleAuthError,
    KaggleCompetitionError,
    KaggleFetcher,
    fetch_bdb_data,
    get_fetcher,
)
from fast_nfl_mcp.data.schema import (
    DATASET_DEFINITIONS,
    SchemaManager,
)

__all__ = [
    # Fetcher
    "DataFetchError",
    "NFLDataPyFetcher",
    # Kaggle
    "KaggleAuthError",
    "KaggleCompetitionError",
    "KaggleFetcher",
    "fetch_bdb_data",
    "get_fetcher",
    # Schema
    "DATASET_DEFINITIONS",
    "SchemaManager",
]
