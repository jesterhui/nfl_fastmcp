"""Schema manager for NFL dataset schema preloading and caching.

This module provides the SchemaManager class that pre-loads and caches
dataset schemas on server startup for fast schema queries. Schemas include
column information, data types, sample values, and available seasons.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import nfl_data_py as nfl
import pandas as pd

from fast_nfl_mcp.constants import MIN_SEASON, get_current_season_year
from fast_nfl_mcp.models import ColumnSchema, DatasetSchema

logger = logging.getLogger(__name__)

# Dataset definitions with their loader functions and metadata
# Each entry contains: (loader_function, description, supports_seasons)
# For seasonal datasets, get_current_season_year() is called at runtime to get the default
DATASET_DEFINITIONS: dict[str, tuple[Callable[..., pd.DataFrame], str, bool]] = {
    "play_by_play": (
        lambda seasons: nfl.import_pbp_data(seasons),
        "Play-by-play data with EPA, WPA, and detailed play outcomes",
        True,
    ),
    "weekly_stats": (
        lambda seasons: nfl.import_weekly_data(seasons),
        "Weekly aggregated player statistics",
        True,
    ),
    "seasonal_stats": (
        lambda seasons: nfl.import_seasonal_data(seasons),
        "Season-level player statistics",
        True,
    ),
    "rosters": (
        lambda seasons: nfl.import_weekly_rosters(seasons),
        "Team rosters with player information",
        True,
    ),
    "player_ids": (
        lambda _: nfl.import_ids(),
        "Cross-platform player ID mappings",
        False,
    ),
    "draft_picks": (
        lambda seasons: nfl.import_draft_picks(seasons),
        "Historical NFL draft data",
        True,
    ),
    "schedules": (
        lambda seasons: nfl.import_schedules(seasons),
        "Game schedules and results",
        True,
    ),
    "team_descriptions": (
        lambda _: nfl.import_team_desc(),
        "Team metadata and information",
        False,
    ),
    "combine_data": (
        lambda seasons: nfl.import_combine_data(seasons),
        "NFL combine results and measurements",
        True,
    ),
    "scoring_lines": (
        lambda seasons: nfl.import_sc_lines(seasons),
        "Betting scoring lines",
        True,
    ),
    "win_totals": (
        lambda seasons: nfl.import_win_totals(seasons),
        "Season win total betting lines",
        True,
    ),
    "ngs_passing": (
        lambda seasons: nfl.import_ngs_data("passing", seasons),
        "Next Gen Stats - passing metrics",
        True,
    ),
    "ngs_rushing": (
        lambda seasons: nfl.import_ngs_data("rushing", seasons),
        "Next Gen Stats - rushing metrics",
        True,
    ),
    "ngs_receiving": (
        lambda seasons: nfl.import_ngs_data("receiving", seasons),
        "Next Gen Stats - receiving metrics",
        True,
    ),
    "snap_counts": (
        lambda seasons: nfl.import_snap_counts(seasons),
        "Player snap participation data",
        True,
    ),
    "injuries": (
        lambda seasons: nfl.import_injuries(seasons),
        "Injury reports and status",
        True,
    ),
    "depth_charts": (
        lambda seasons: nfl.import_depth_charts(seasons),
        "Team depth charts",
        True,
    ),
    "contracts": (
        lambda _: nfl.import_contracts(),
        "Player contract data",
        False,
    ),
    "officials": (
        lambda _: nfl.import_officials(),
        "Game officials data",
        False,
    ),
    "qbr": (
        lambda seasons: nfl.import_qbr(seasons),
        "ESPN QBR ratings",
        True,
    ),
}


def _extract_sample_values(series: pd.Series, max_samples: int = 2) -> list[Any]:
    """Extract sample values from a pandas Series.

    Args:
        series: The pandas Series to extract samples from.
        max_samples: Maximum number of samples to extract (default 2).

    Returns:
        A list of sample values, with NaN values excluded.
    """
    # Drop NaN values and get unique values for better representation
    non_null = series.dropna()
    if len(non_null) == 0:
        return []

    # Get unique values first for variety
    unique_vals = non_null.unique()
    samples = unique_vals[:max_samples].tolist()

    # Convert numpy types to Python native types for JSON serialization
    result = []
    for val in samples:
        if pd.isna(val):
            continue
        elif hasattr(val, "item"):
            # numpy scalar types
            result.append(val.item())
        elif isinstance(val, pd.Timestamp):
            result.append(str(val))
        else:
            result.append(val)

    return result[:max_samples]


def _extract_column_schema(df: pd.DataFrame, column: str) -> ColumnSchema:
    """Extract schema information for a single column.

    Args:
        df: The DataFrame containing the column.
        column: The column name to extract schema for.

    Returns:
        A ColumnSchema object with column metadata.
    """
    series = df[column]
    return ColumnSchema(
        name=column,
        dtype=str(series.dtype),
        sample_values=_extract_sample_values(series),
        null_count=int(series.isna().sum()),
        unique_count=int(series.nunique()),
    )


class SchemaManager:
    """Manages pre-loading and caching of NFL dataset schemas.

    The SchemaManager loads schema information for all NFL datasets on startup
    and caches them in memory for fast retrieval. Schema information includes
    column names, data types, sample values, null counts, and unique counts.

    Attributes:
        _schemas: Dictionary mapping dataset names to DatasetSchema objects.
        _failed_datasets: Set of dataset names that failed to load.
    """

    def __init__(self) -> None:
        """Initialize the SchemaManager with empty caches."""
        self._schemas: dict[str, DatasetSchema] = {}
        self._failed_datasets: set[str] = set()

    def _load_single_schema(self, dataset_name: str) -> DatasetSchema | None:
        """Load schema for a single dataset.

        Args:
            dataset_name: The name of the dataset to load.

        Returns:
            A DatasetSchema object if successful, None if the dataset failed to load.
        """
        if dataset_name not in DATASET_DEFINITIONS:
            logger.warning(f"Unknown dataset: {dataset_name}")
            return None

        loader, description, supports_seasons = DATASET_DEFINITIONS[dataset_name]

        try:
            logger.info(f"Loading schema for {dataset_name}...")

            # Load a small sample of data to extract schema
            # Get current season for seasonal datasets
            if supports_seasons:
                current_season = get_current_season_year()
                df = loader([current_season])
            else:
                current_season = None
                df = loader(None)

            # Handle case where loader returns None or empty DataFrame
            if df is None or df.empty:
                logger.warning(f"Dataset {dataset_name} returned empty DataFrame")
                empty_seasons: list[int] | None = None
                if supports_seasons and current_season is not None:
                    empty_seasons = [current_season]
                return DatasetSchema(
                    name=dataset_name,
                    description=description,
                    columns=[],
                    row_count=0,
                    available_seasons=empty_seasons,
                )

            # Extract column schemas
            columns = [_extract_column_schema(df, col) for col in df.columns]

            # Determine available seasons if applicable
            available_seasons: list[int] | None = None
            if supports_seasons and current_season is not None:
                # Dynamic range from MIN_SEASON to current season
                available_seasons = list(range(MIN_SEASON, current_season + 1))

            schema = DatasetSchema(
                name=dataset_name,
                description=description,
                columns=columns,
                row_count=len(df),
                available_seasons=available_seasons,
            )

            logger.info(
                f"Successfully loaded schema for {dataset_name}: "
                f"{len(columns)} columns, {len(df)} sample rows"
            )
            return schema

        except (OSError, ConnectionError, TimeoutError) as e:
            # Network and I/O errors are expected failure modes
            logger.error(f"Failed to load schema for {dataset_name}: {e}")
            return None

        except (ValueError, KeyError, RuntimeError) as e:
            # Data validation and runtime errors from the library
            logger.error(f"Failed to load schema for {dataset_name}: {e}")
            return None

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Failed to load schema for {dataset_name}: {e}")
            return None

    def preload_all(self, max_workers: int = 4, timeout: float = 60.0) -> None:
        """Pre-load schemas for all datasets.

        This method loads schema information for all known NFL datasets
        in parallel using a thread pool. Failed datasets are tracked
        and can be queried later.

        Args:
            max_workers: Maximum number of parallel workers (default 4).
            timeout: Maximum time in seconds for each dataset load (default 60).
        """
        logger.info(f"Starting schema preload for {len(DATASET_DEFINITIONS)} datasets")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all loading tasks
            future_to_dataset = {
                executor.submit(self._load_single_schema, name): name
                for name in DATASET_DEFINITIONS
            }

            # Collect results as they complete
            for future in as_completed(future_to_dataset, timeout=timeout):
                dataset_name = future_to_dataset[future]
                try:
                    schema = future.result(timeout=timeout)
                    if schema is not None:
                        self._schemas[dataset_name] = schema
                    else:
                        self._failed_datasets.add(dataset_name)
                except TimeoutError:
                    logger.error(f"Timeout loading schema for {dataset_name}")
                    self._failed_datasets.add(dataset_name)
                except (
                    OSError,
                    ConnectionError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                ) as e:
                    # Expected failure modes from data loading
                    logger.error(f"Error loading schema for {dataset_name}: {e}")
                    self._failed_datasets.add(dataset_name)
                except Exception as e:
                    # Catch-all for unexpected errors
                    logger.error(f"Error loading schema for {dataset_name}: {e}")
                    self._failed_datasets.add(dataset_name)

        logger.info(
            f"Schema preload complete: {len(self._schemas)} loaded, "
            f"{len(self._failed_datasets)} failed"
        )

    def get_schema(self, dataset: str) -> DatasetSchema | None:
        """Retrieve the cached schema for a dataset.

        Args:
            dataset: The name of the dataset to retrieve.

        Returns:
            The DatasetSchema if available, None if not found or failed to load.
        """
        return self._schemas.get(dataset)

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all available datasets with their basic information.

        Returns:
            A list of dictionaries containing dataset name, description,
            and availability status.
        """
        datasets = []
        for name, (_, description, supports_seasons) in DATASET_DEFINITIONS.items():
            status = "loaded"
            if name in self._failed_datasets:
                status = "failed"
            elif name not in self._schemas:
                status = "not_loaded"

            datasets.append(
                {
                    "name": name,
                    "description": description,
                    "supports_seasons": supports_seasons,
                    "status": status,
                }
            )

        return datasets

    def get_dataset_names(self) -> list[str]:
        """Get a list of all dataset names.

        Returns:
            A list of all known dataset names.
        """
        return list(DATASET_DEFINITIONS.keys())

    def get_loaded_count(self) -> int:
        """Get the number of successfully loaded schemas.

        Returns:
            The count of loaded schemas.
        """
        return len(self._schemas)

    def get_failed_datasets(self) -> set[str]:
        """Get the set of datasets that failed to load.

        Returns:
            A set of dataset names that failed to load.
        """
        return self._failed_datasets.copy()

    def is_loaded(self, dataset: str) -> bool:
        """Check if a dataset's schema is loaded.

        Args:
            dataset: The dataset name to check.

        Returns:
            True if the schema is loaded, False otherwise.
        """
        return dataset in self._schemas
