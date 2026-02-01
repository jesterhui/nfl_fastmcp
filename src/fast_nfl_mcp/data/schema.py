"""Schema manager for NFL dataset schema preloading and caching.

This module provides the SchemaManager class that pre-loads and caches
dataset schemas on server startup for fast schema queries. Schemas include
column information, data types, sample values, and available seasons.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

import nfl_data_py as nfl
import pandas as pd

from fast_nfl_mcp.core.models import ColumnSchema, DatasetSchema
from fast_nfl_mcp.utils.constants import MIN_SEASON, get_current_season_year
from fast_nfl_mcp.utils.enums import DatasetName, DatasetStatus
from fast_nfl_mcp.utils.serialization import extract_sample_values
from fast_nfl_mcp.utils.types import DatasetDefinition

logger = logging.getLogger(__name__)

# Dataset definitions with their loader functions and metadata
# Using DatasetDefinition NamedTuple for explicit field access
DATASET_DEFINITIONS: dict[str, DatasetDefinition] = {
    DatasetName.PLAY_BY_PLAY: DatasetDefinition(
        loader=lambda seasons: nfl.import_pbp_data(seasons),
        description="Play-by-play data with EPA, WPA, and detailed play outcomes",
        supports_seasons=True,
    ),
    DatasetName.WEEKLY_STATS: DatasetDefinition(
        loader=lambda seasons: nfl.import_weekly_data(seasons),
        description="Weekly aggregated player statistics",
        supports_seasons=True,
    ),
    DatasetName.SEASONAL_STATS: DatasetDefinition(
        loader=lambda seasons: nfl.import_seasonal_data(seasons),
        description="Season-level player statistics",
        supports_seasons=True,
    ),
    DatasetName.ROSTERS: DatasetDefinition(
        loader=lambda seasons: nfl.import_weekly_rosters(seasons),
        description="Team rosters with player information",
        supports_seasons=True,
    ),
    DatasetName.PLAYER_IDS: DatasetDefinition(
        loader=lambda _: nfl.import_ids(),
        description="Cross-platform player ID mappings",
        supports_seasons=False,
    ),
    DatasetName.DRAFT_PICKS: DatasetDefinition(
        loader=lambda seasons: nfl.import_draft_picks(seasons),
        description="Historical NFL draft data",
        supports_seasons=True,
    ),
    DatasetName.SCHEDULES: DatasetDefinition(
        loader=lambda seasons: nfl.import_schedules(seasons),
        description="Game schedules and results",
        supports_seasons=True,
    ),
    DatasetName.TEAM_DESCRIPTIONS: DatasetDefinition(
        loader=lambda _: nfl.import_team_desc(),
        description="Team metadata and information",
        supports_seasons=False,
    ),
    DatasetName.COMBINE_DATA: DatasetDefinition(
        loader=lambda seasons: nfl.import_combine_data(seasons),
        description="NFL combine results and measurements",
        supports_seasons=True,
    ),
    DatasetName.SCORING_LINES: DatasetDefinition(
        loader=lambda seasons: nfl.import_sc_lines(seasons),
        description="Betting scoring lines",
        supports_seasons=True,
    ),
    DatasetName.WIN_TOTALS: DatasetDefinition(
        loader=lambda seasons: nfl.import_win_totals(seasons),
        description="Season win total betting lines",
        supports_seasons=True,
    ),
    DatasetName.NGS_PASSING: DatasetDefinition(
        loader=lambda seasons: nfl.import_ngs_data("passing", seasons),
        description="Next Gen Stats - passing metrics",
        supports_seasons=True,
    ),
    DatasetName.NGS_RUSHING: DatasetDefinition(
        loader=lambda seasons: nfl.import_ngs_data("rushing", seasons),
        description="Next Gen Stats - rushing metrics",
        supports_seasons=True,
    ),
    DatasetName.NGS_RECEIVING: DatasetDefinition(
        loader=lambda seasons: nfl.import_ngs_data("receiving", seasons),
        description="Next Gen Stats - receiving metrics",
        supports_seasons=True,
    ),
    DatasetName.SNAP_COUNTS: DatasetDefinition(
        loader=lambda seasons: nfl.import_snap_counts(seasons),
        description="Player snap participation data",
        supports_seasons=True,
    ),
    DatasetName.INJURIES: DatasetDefinition(
        loader=lambda seasons: nfl.import_injuries(seasons),
        description="Injury reports and status",
        supports_seasons=True,
    ),
    DatasetName.DEPTH_CHARTS: DatasetDefinition(
        loader=lambda seasons: nfl.import_depth_charts(seasons),
        description="Team depth charts",
        supports_seasons=True,
    ),
    DatasetName.CONTRACTS: DatasetDefinition(
        loader=lambda _: nfl.import_contracts(),
        description="Player contract data",
        supports_seasons=False,
    ),
    DatasetName.OFFICIALS: DatasetDefinition(
        loader=lambda _: nfl.import_officials(),
        description="Game officials data",
        supports_seasons=False,
    ),
    DatasetName.QBR: DatasetDefinition(
        loader=lambda seasons: nfl.import_qbr(seasons),
        description="ESPN QBR ratings",
        supports_seasons=True,
    ),
}


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
        sample_values=extract_sample_values(series),
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

        definition = DATASET_DEFINITIONS[dataset_name]

        try:
            logger.info(f"Loading schema for {dataset_name}...")

            # Load a small sample of data to extract schema
            # Get current season for seasonal datasets
            if definition.supports_seasons:
                current_season = get_current_season_year()
                df = definition.loader([current_season])
            else:
                current_season = None
                df = definition.loader(None)

            # Handle case where loader returns None or empty DataFrame
            if df is None or df.empty:
                logger.warning(f"Dataset {dataset_name} returned empty DataFrame")
                empty_seasons: list[int] | None = None
                if definition.supports_seasons and current_season is not None:
                    empty_seasons = [current_season]
                return DatasetSchema(
                    name=dataset_name,
                    description=definition.description,
                    columns=[],
                    row_count=0,
                    available_seasons=empty_seasons,
                )

            # Extract column schemas
            columns = [_extract_column_schema(df, col) for col in df.columns]

            # Determine available seasons if applicable
            available_seasons: list[int] | None = None
            if definition.supports_seasons and current_season is not None:
                # Dynamic range from MIN_SEASON to current season
                available_seasons = list(range(MIN_SEASON, current_season + 1))

            schema = DatasetSchema(
                name=dataset_name,
                description=definition.description,
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

        This method never raises on timeout - instead, pending datasets
        are marked as failed and logged.

        Args:
            max_workers: Maximum number of parallel workers (default 4).
            timeout: Maximum time in seconds for the overall preload operation (default 60).
        """
        logger.info(f"Starting schema preload for {len(DATASET_DEFINITIONS)} datasets")

        # Track which futures have been processed
        processed_futures: set[int] = set()
        timed_out = False

        # Don't use context manager - we need control over shutdown behavior
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            # Submit all loading tasks
            future_to_dataset = {
                executor.submit(self._load_single_schema, name): name
                for name in DATASET_DEFINITIONS
            }

            try:
                # Collect results as they complete
                for future in as_completed(future_to_dataset, timeout=timeout):
                    processed_futures.add(id(future))
                    dataset_name = future_to_dataset[future]
                    self._process_future_result(future, dataset_name)
            except TimeoutError:
                # Global timeout reached - mark remaining datasets as failed
                timed_out = True
                logger.warning(
                    "Global preload timeout reached, marking pending datasets as failed"
                )
                for future, dataset_name in future_to_dataset.items():
                    if id(future) not in processed_futures:
                        logger.error(
                            f"Timeout loading schema for {dataset_name} "
                            "(global timeout reached)"
                        )
                        self._failed_datasets.add(dataset_name)
        finally:
            # On timeout, don't wait for running futures - return immediately
            # On normal completion, wait for any stragglers
            executor.shutdown(wait=not timed_out, cancel_futures=timed_out)

        logger.info(
            f"Schema preload complete: {len(self._schemas)} loaded, "
            f"{len(self._failed_datasets)} failed"
        )

    def _process_future_result(
        self, future: "Future[DatasetSchema | None]", dataset_name: str
    ) -> None:
        """Process the result of a completed future.

        Args:
            future: The completed Future object.
            dataset_name: The name of the dataset this future was loading.
        """
        try:
            schema = future.result(timeout=0)  # Already complete, no wait needed
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
        for name, definition in DATASET_DEFINITIONS.items():
            if name in self._failed_datasets:
                status = DatasetStatus.UNAVAILABLE
            elif name not in self._schemas:
                status = DatasetStatus.NOT_LOADED
            else:
                status = DatasetStatus.AVAILABLE

            datasets.append(
                {
                    "name": name,
                    "description": definition.description,
                    "supports_seasons": definition.supports_seasons,
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
