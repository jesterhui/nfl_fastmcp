"""FastMCP server for NFL data access.

This module initializes the FastMCP application and provides the entry point
for running the MCP server via stdio transport.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from fast_nfl_mcp.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.schema_manager import SchemaManager
from fast_nfl_mcp.tools.play_by_play import get_play_by_play_impl
from fast_nfl_mcp.tools.utilities import describe_dataset_impl, list_datasets_impl


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage server lifecycle.

    This context manager handles startup and shutdown operations,
    including preloading dataset schemas for fast retrieval.

    Args:
        server: The FastMCP server instance.

    Yields:
        A dictionary containing shared resources including the SchemaManager.
    """
    # Initialize and preload the schema manager
    schema_manager = SchemaManager()
    schema_manager.preload_all()

    yield {"schema_manager": schema_manager}


# Initialize the FastMCP application
mcp = FastMCP(
    name="fast-nfl-mcp",
    lifespan=lifespan,
)


@mcp.tool()
def list_datasets(ctx: Context) -> SuccessResponse:
    """List all available NFL datasets.

    Returns a list of all available datasets that can be queried,
    including their names, descriptions, and whether they support
    season-based filtering. Use this to discover what data is available.

    Returns:
        A list of datasets with name, description, supports_seasons, and status.
    """
    schema_manager: SchemaManager = ctx.request_context.lifespan_context[
        "schema_manager"
    ]
    return list_datasets_impl(schema_manager)


@mcp.tool()
def describe_dataset(ctx: Context, dataset: str) -> SuccessResponse | ErrorResponse:
    """Get detailed schema information for a specific NFL dataset.

    Returns column names, data types, sample values, and other metadata
    for the specified dataset. Use this to understand what data is available
    and how to query it before making data requests.

    Args:
        dataset: The name of the dataset to describe (e.g., "play_by_play",
                "weekly_stats", "rosters"). Use list_datasets to see all options.

    Returns:
        Schema information including columns, types, and sample values,
        or an error if the dataset is not found.
    """
    schema_manager: SchemaManager = ctx.request_context.lifespan_context[
        "schema_manager"
    ]
    return describe_dataset_impl(schema_manager, dataset)


@mcp.tool()
def get_play_by_play(
    seasons: list[int],
    columns: Annotated[
        list[str],
        Field(
            description="List of column names to include in output (REQUIRED). "
            "Use describe_dataset('play_by_play') to see available columns. "
            "Common useful columns: play_id, game_id, desc, play_type, yards_gained, "
            "posteam, defteam, down, ydstogo, epa, wpa"
        ),
    ],
    weeks: list[int] | None = None,
    filters: Annotated[
        dict[str, Any] | None,
        Field(
            description="Filter on any column. Keys are column names, values are "
            "either a single value or list of acceptable values. "
            'Example: {"posteam": "TB", "play_type": ["pass", "run"]}'
        ),
    ] = None,
    offset: Annotated[
        int,
        Field(
            description="Number of rows to skip for pagination. Use with limit to page through results.",
            ge=0,
        ),
    ] = 0,
    limit: Annotated[
        int | None,
        Field(
            description="Maximum number of rows to return (default 100, max 100).",
            ge=1,
            le=100,
        ),
    ] = None,
) -> SuccessResponse | ErrorResponse:
    """Get NFL play-by-play data including EPA, WPA, and detailed play outcomes.

    Retrieves play-level data for specified seasons and weeks. This is the most
    comprehensive dataset, containing detailed information about every play.

    Key columns include:
    - Play identifiers: game_id, play_id, drive, play_type
    - Teams: posteam (possession), defteam (defense), home_team, away_team
    - Players: passer, receiver, rusher names and IDs
    - Outcomes: yards_gained, touchdown, interception, fumble
    - Advanced metrics: epa (Expected Points Added), wpa (Win Probability Added)
    - Situational: down, ydstogo, yardline_100, score_differential

    Args:
        seasons: List of seasons (e.g., [2023, 2024]). Maximum 3 seasons allowed.
                 Play-by-play data is available from 1999 onwards.
        columns: List of column names to include (REQUIRED to manage response size).
        weeks: Optional list of weeks to filter (1-18 for regular season).
               If not specified, returns data for all weeks.
        filters: Optional dict to filter on any column. Keys are column names,
                 values can be a single value or list of acceptable values.
                 Use describe_dataset("play_by_play") to see available columns.
        offset: Number of rows to skip for pagination (default 0).
        limit: Maximum number of rows to return (default 100, max 100).

    Returns:
        Play-by-play data as JSON with up to 100 rows by default. Use offset and limit
        to paginate through larger result sets.

    Examples:
        Get basic plays: get_play_by_play([2024], ["play_id", "desc", "epa"])
        Filter by week: get_play_by_play([2024], ["desc", "yards_gained"], weeks=[1, 2])
        Filter by team: get_play_by_play([2024], ["desc", "epa"], filters={"posteam": "TB"})
        Paginate: get_play_by_play([2024], ["play_id", "desc"], offset=100, limit=50)
    """
    return get_play_by_play_impl(seasons, weeks, filters, offset, limit, columns)


def main() -> None:
    """Run the MCP server.

    This is the entry point for the `fast-nfl-mcp` command defined in pyproject.toml.
    It starts the server using stdio transport for MCP communication.
    """
    mcp.run()


if __name__ == "__main__":
    main()
