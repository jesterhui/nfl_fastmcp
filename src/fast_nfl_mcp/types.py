"""Type definitions for the Fast NFL MCP server.

This module provides typed structures for dataset definitions and other
reusable type aliases used throughout the package.
"""

from collections.abc import Callable
from typing import NamedTuple

import pandas as pd


class DatasetDefinition(NamedTuple):
    """Definition of an NFL dataset with its loader and metadata.

    Attributes:
        loader: A callable that takes an optional seasons list and returns a DataFrame.
        description: Human-readable description of the dataset.
        supports_seasons: Whether the dataset supports season-based filtering.
    """

    loader: Callable[[list[int] | None], pd.DataFrame]
    description: str
    supports_seasons: bool
