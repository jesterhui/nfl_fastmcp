"""Kaggle data fetcher for Big Data Bowl tracking data.

This module provides the KaggleFetcher class for accessing NFL Big Data Bowl
competition data via the Kaggle API. It handles authentication, downloading
competition files, and in-memory caching of DataFrames.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fast_nfl_mcp.constants import (
    BDB_AVAILABLE_WEEKS,
    BDB_COMPETITION_HANDLE,
    DEFAULT_MAX_ROWS_TRACKING,
    MAX_ROWS_TRACKING,
)
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)

logger = logging.getLogger(__name__)


class KaggleAuthError(Exception):
    """Exception raised when Kaggle authentication fails."""

    pass


class KaggleCompetitionError(Exception):
    """Exception raised when there's an issue with the Kaggle competition."""

    pass


class KaggleFetcher:
    """Fetches NFL Big Data Bowl data from Kaggle.

    This class handles:
    - Checking for Kaggle authentication (kaggle.json)
    - Downloading competition data via kagglehub
    - In-memory caching of DataFrames per session
    - Clear error messages for auth/competition issues

    Attributes:
        _data_path: Path to downloaded competition data.
        _cache: Dictionary caching loaded DataFrames by filename.
    """

    def __init__(self) -> None:
        """Initialize the KaggleFetcher."""
        self._data_path: Path | None = None
        self._cache: dict[str, pd.DataFrame] = {}
        self._auth_checked: bool = False
        self._auth_valid: bool = False

    def _check_auth(self) -> None:
        """Check if Kaggle authentication is configured.

        Raises:
            KaggleAuthError: If neither ~/.kaggle/kaggle.json nor
                ~/.kaggle/access_token is found.
        """
        if self._auth_checked:
            if not self._auth_valid:
                raise KaggleAuthError(
                    "Kaggle authentication not configured. "
                    "Please create ~/.kaggle/kaggle.json or ~/.kaggle/access_token. "
                    "Get your API token from: https://www.kaggle.com/settings"
                )
            return

        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        access_token = kaggle_dir / "access_token"

        if not kaggle_json.exists() and not access_token.exists():
            self._auth_checked = True
            self._auth_valid = False
            raise KaggleAuthError(
                "Kaggle authentication not configured. "
                f"Please create {kaggle_json} or {access_token} with your API credentials. "
                "Get your API token from: https://www.kaggle.com/settings"
            )

        self._auth_checked = True
        self._auth_valid = True

    def _download_competition_data(self) -> Path:
        """Download competition data using kagglehub.

        Returns:
            Path to the downloaded competition data directory.

        Raises:
            KaggleAuthError: If authentication fails.
            KaggleCompetitionError: If download fails or rules not accepted.
        """
        if self._data_path is not None:
            return self._data_path

        self._check_auth()

        try:
            import kagglehub

            logger.info(f"Downloading competition data: {BDB_COMPETITION_HANDLE}")
            self._data_path = Path(
                kagglehub.competition_download(BDB_COMPETITION_HANDLE)
            )
            logger.info(f"Competition data downloaded to: {self._data_path}")
            return self._data_path

        except Exception as e:
            error_msg = str(e).lower()
            if "403" in error_msg or "forbidden" in error_msg:
                raise KaggleCompetitionError(
                    f"Access denied to competition '{BDB_COMPETITION_HANDLE}'. "
                    "You must accept the competition rules first. "
                    f"Visit: https://www.kaggle.com/competitions/{BDB_COMPETITION_HANDLE}/rules"
                ) from e
            if "404" in error_msg or "not found" in error_msg:
                raise KaggleCompetitionError(
                    f"Competition '{BDB_COMPETITION_HANDLE}' not found."
                ) from e
            if "401" in error_msg or "unauthorized" in error_msg:
                raise KaggleAuthError(
                    "Kaggle API authentication failed. "
                    "Please check your API credentials in ~/.kaggle/kaggle.json. "
                    "Get a new API token from: https://www.kaggle.com/settings"
                ) from e
            raise KaggleCompetitionError(
                f"Failed to download competition data: {e}"
            ) from e

    def _find_data_subdir(self, data_path: Path) -> Path:
        """Find the actual data subdirectory within the competition folder.

        BDB 2026 extracts to a numbered subdirectory like
        '114239_nfl_competition_files_published_analytics_final/'.

        Args:
            data_path: The competition data root path.

        Returns:
            The path to the actual data directory.
        """
        # If root has CSV files, use it directly (don't descend into train/ etc.)
        root_csvs = list(data_path.glob("*.csv"))
        if root_csvs:
            return data_path

        # Only descend into subdirectory if root has no CSV files
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
        return data_path

    def _load_csv(self, filename: str, subdir: str | None = None) -> pd.DataFrame:
        """Load a CSV file from the competition data directory.

        Args:
            filename: Name of the CSV file to load.
            subdir: Optional subdirectory within the data folder (e.g., 'train').

        Returns:
            DataFrame containing the file contents.

        Raises:
            KaggleCompetitionError: If the file cannot be loaded.
        """
        cache_key = f"{subdir}/{filename}" if subdir else filename
        if cache_key in self._cache:
            return self._cache[cache_key]

        data_path = self._download_competition_data()
        actual_data_path = self._find_data_subdir(data_path)

        if subdir:
            file_path = actual_data_path / subdir / filename
        else:
            file_path = actual_data_path / filename

        if not file_path.exists():
            search_path = actual_data_path / subdir if subdir else actual_data_path
            available = list(search_path.glob("*.csv")) if search_path.exists() else []
            raise KaggleCompetitionError(
                f"File '{filename}' not found in competition data. "
                f"Available files: {[f.name for f in available[:10]]}"
            )

        try:
            logger.info(f"Loading {cache_key}")
            df = pd.read_csv(file_path)
            self._cache[cache_key] = df
            logger.info(f"Loaded {cache_key}: {len(df)} rows")
            return df
        except Exception as e:
            raise KaggleCompetitionError(f"Failed to load {filename}: {e}") from e

    def get_games(self) -> pd.DataFrame:
        """Get unique games data from the supplementary data.

        Returns:
            DataFrame containing game metadata (deduplicated by game_id).
        """
        df = self._load_csv("supplementary_data.csv")
        # Extract unique games
        game_cols = [
            "game_id",
            "season",
            "week",
            "game_date",
            "game_time_eastern",
            "home_team_abbr",
            "visitor_team_abbr",
        ]
        available_cols = [c for c in game_cols if c in df.columns]
        return df[available_cols].drop_duplicates(subset=["game_id"])

    def get_plays(self) -> pd.DataFrame:
        """Get plays data from the supplementary data.

        Returns:
            DataFrame containing play-level data.
        """
        return self._load_csv("supplementary_data.csv")

    def get_players(self) -> pd.DataFrame:
        """Get unique players from tracking data across all weeks.

        Aggregates player information from all available weeks to ensure
        players who didn't appear in week 1 (bye, injury, late-season
        signing, etc.) are included.

        Returns:
            DataFrame containing player information.
        """
        # Check cache first
        cache_key = "_players_aggregated"
        if cache_key in self._cache:
            return self._cache[cache_key]

        player_cols = [
            "nfl_id",
            "player_name",
            "player_height",
            "player_weight",
            "player_birth_date",
            "player_position",
        ]

        # Aggregate players from all weeks
        all_players: list[pd.DataFrame] = []
        for week in BDB_AVAILABLE_WEEKS:
            try:
                df = self._load_csv(f"input_2023_w{week:02d}.csv", subdir="train")
                available_cols = [c for c in player_cols if c in df.columns]
                if available_cols and "nfl_id" in available_cols:
                    week_players = df[available_cols].drop_duplicates(subset=["nfl_id"])
                    all_players.append(week_players)
            except KaggleCompetitionError:
                # Week file doesn't exist, skip it
                continue

        if not all_players:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=player_cols)

        # Combine and deduplicate
        combined = pd.concat(all_players, ignore_index=True)
        result = combined.drop_duplicates(subset=["nfl_id"])

        # Cache the result
        self._cache[cache_key] = result
        return result

    def get_tracking(self, week: int) -> pd.DataFrame:
        """Get tracking data for a specific week.

        Args:
            week: Week number (1-18 for BDB 2026).

        Returns:
            DataFrame containing per-frame tracking data.

        Raises:
            ValueError: If week is not in valid range.
        """
        if week not in BDB_AVAILABLE_WEEKS:
            raise ValueError(
                f"Invalid week: {week}. "
                f"Valid weeks for BDB 2026 are: {list(BDB_AVAILABLE_WEEKS)}"
            )
        return self._load_csv(f"input_2023_w{week:02d}.csv", subdir="train")

    def clear_cache(self) -> None:
        """Clear the in-memory DataFrame cache."""
        self._cache.clear()
        logger.info("Cleared KaggleFetcher cache")


# Module-level singleton instance
_fetcher: KaggleFetcher | None = None


def get_fetcher() -> KaggleFetcher:
    """Get the module-level KaggleFetcher singleton.

    Returns:
        The singleton KaggleFetcher instance.
    """
    global _fetcher
    if _fetcher is None:
        _fetcher = KaggleFetcher()
    return _fetcher


def _convert_value(val: Any) -> Any:
    """Convert non-serializable values to JSON-safe types."""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return str(val)
    if hasattr(val, "item"):
        return val.item()
    return val


def _convert_dataframe_to_records(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert a DataFrame to a list of dictionaries.

    Handles special cases like NaN values and numpy types to ensure
    JSON serialization compatibility.

    Args:
        df: The DataFrame to convert.

    Returns:
        A tuple of (list of row dictionaries, list of column names).
    """
    if df is None or df.empty:
        return [], []

    columns: list[str] = [str(col) for col in df.columns]
    result_df = df.copy()

    for col in result_df.columns:
        dtype = result_df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            values = [None if pd.isna(v) else str(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)
        elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            values = [_convert_value(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)
        elif pd.api.types.is_bool_dtype(dtype):
            values = [_convert_value(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)
        elif pd.api.types.is_string_dtype(dtype) or dtype is object:
            values = [_convert_value(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)

    records = result_df.to_dict(orient="records")
    cleaned_records: list[dict[str, Any]] = [
        {str(k): v for k, v in record.items()} for record in records
    ]

    return cleaned_records, columns


def fetch_bdb_data(
    data_type: str,
    week: int | None = None,
    filters: dict[str, list[Any]] | None = None,
    offset: int = 0,
    limit: int | None = None,
    columns: list[str] | None = None,
) -> SuccessResponse | ErrorResponse:
    """Fetch BDB data with filtering and pagination.

    Args:
        data_type: Type of data to fetch ('games', 'plays', 'players', 'tracking').
        week: Week number (required for tracking data).
        filters: Optional dict mapping column names to lists of acceptable values.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return.
        columns: Optional list of column names to include.

    Returns:
        SuccessResponse with data and metadata, or ErrorResponse on error.
    """
    if filters is None:
        filters = {}

    # Determine default and max limits
    if data_type == "tracking":
        default_limit = DEFAULT_MAX_ROWS_TRACKING
        max_limit = MAX_ROWS_TRACKING
    else:
        default_limit = 50
        max_limit = 100

    effective_limit = limit if limit is not None else default_limit
    if effective_limit > max_limit:
        effective_limit = max_limit

    try:
        fetcher = get_fetcher()

        if data_type == "games":
            df = fetcher.get_games()
        elif data_type == "plays":
            df = fetcher.get_plays()
        elif data_type == "players":
            df = fetcher.get_players()
        elif data_type == "tracking":
            if week is None:
                return create_error_response(
                    error="week parameter is required for tracking data. "
                    f"Valid weeks are: {list(BDB_AVAILABLE_WEEKS)}"
                )
            df = fetcher.get_tracking(week)
        else:
            return create_error_response(
                error=f"Unknown data type: '{data_type}'. "
                "Valid types are: games, plays, players, tracking"
            )

        if df is None or df.empty:
            return create_success_response(
                data=[],
                total_available=0,
                truncated=False,
                columns=[],
                warning=f"No data found for {data_type}.",
            )

        # Apply filters
        invalid_filter_cols: list[str] = []
        for column, values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(values)]
            else:
                invalid_filter_cols.append(column)

        if df.empty:
            filter_warning = "No data matched the specified filters."
            if invalid_filter_cols:
                filter_warning += (
                    f" Note: The following filter columns do not exist "
                    f"in the dataset and were ignored: {invalid_filter_cols}"
                )
            return create_success_response(
                data=[],
                total_available=0,
                truncated=False,
                columns=[str(col) for col in df.columns],
                warning=filter_warning,
            )

        # Select specific columns if requested
        if columns is not None:
            if not columns:
                return create_error_response(
                    error="Empty columns list provided. "
                    "Please specify at least one column name."
                )
            available_cols = set(df.columns)
            valid_cols = [c for c in columns if c in available_cols]
            invalid_cols = [c for c in columns if c not in available_cols]
            if not valid_cols:
                return create_error_response(
                    error=f"None of the requested columns exist: {invalid_cols}. "
                    f"Available columns: {sorted(available_cols)}"
                )
            df = df[valid_cols]
            if invalid_cols:
                logger.warning(f"Requested columns not found: {invalid_cols}")

        # Get total count before pagination
        total_available = len(df)

        # Apply offset
        if offset > 0:
            df = df.iloc[offset:]

        # Check if results will be truncated
        rows_after_offset = len(df)
        truncated = rows_after_offset > effective_limit

        # Apply limit
        if truncated:
            df = df.head(effective_limit)

        # Convert to records
        records, columns_list = _convert_dataframe_to_records(df)

        # Build warning message
        warnings: list[str] = []
        if invalid_filter_cols:
            warnings.append(
                f"The following filter columns do not exist in the dataset "
                f"and were ignored: {invalid_filter_cols}"
            )
        if truncated:
            next_offset = offset + effective_limit
            warnings.append(
                f"Results truncated. Showing {effective_limit} of {total_available} total rows "
                f"(offset: {offset}). Use offset={next_offset} to get the next page."
            )
        warning = " ".join(warnings) if warnings else None

        return create_success_response(
            data=records,
            total_available=total_available,
            truncated=truncated,
            columns=columns_list,
            warning=warning,
        )

    except KaggleAuthError as e:
        return create_error_response(error=str(e))

    except KaggleCompetitionError as e:
        return create_error_response(error=str(e))

    except ValueError as e:
        return create_error_response(error=str(e))

    except Exception as e:
        logger.error(f"Unexpected error fetching BDB {data_type}: {e}")
        return create_error_response(error=f"Error fetching BDB {data_type}: {str(e)}")
