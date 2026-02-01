"""Tool registration module for the Fast NFL MCP server.

This module contains all MCP tool definitions (decorated functions with docstrings)
and provides a `register_tools()` function to centralize tool registration with the
FastMCP server instance.

The tool implementations are in their respective modules (e.g., play_by_play.py,
player_stats.py). This module provides the thin wrappers that:
- Define the tool interface (parameters with Field annotations)
- Provide detailed docstrings for MCP tool discovery
- Delegate to the implementation functions
"""

from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from fast_nfl_mcp.core.models import ErrorResponse, SuccessResponse
from fast_nfl_mcp.data.schema import SchemaManager
from fast_nfl_mcp.utils.constants import BDB_AVAILABLE_WEEKS
from fast_nfl_mcp.tools.bdb import (
    get_bdb_games_impl,
    get_bdb_players_impl,
    get_bdb_plays_impl,
    get_bdb_tracking_impl,
)
from fast_nfl_mcp.tools.draft import get_draft_picks_impl
from fast_nfl_mcp.tools.play_by_play import get_play_by_play_impl
from fast_nfl_mcp.tools.player_stats import (
    get_seasonal_stats_impl,
    get_weekly_stats_impl,
)
from fast_nfl_mcp.tools.reference import (
    get_contracts_impl,
    get_officials_impl,
    get_player_ids_impl,
    get_team_descriptions_impl,
    lookup_player_impl,
)
from fast_nfl_mcp.tools.rosters import get_rosters_impl
from fast_nfl_mcp.tools.schedules import get_schedules_impl
from fast_nfl_mcp.tools.utilities import describe_dataset_impl, list_datasets_impl


def register_tools(mcp: FastMCP) -> None:
    """Register all MCP tools with the FastMCP server.

    This function registers all tool wrappers with the provided FastMCP instance.
    Each tool is a thin wrapper that defines the interface (parameters, docstrings)
    and delegates to the implementation in its respective module.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """

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

    @mcp.tool()
    def get_weekly_stats(
        seasons: list[int],
        columns: Annotated[
            list[str],
            Field(
                description="List of column names to include in output (REQUIRED). "
                "Use describe_dataset('weekly_stats') to see available columns. "
                "Common useful columns: player_id, player_name, season, week, team, "
                "passing_yards, passing_tds, rushing_yards, receiving_yards, fantasy_points"
            ),
        ],
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"team": "KC", "week": [1, 2, 3]}'
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
        """Get weekly aggregated player statistics.

        Retrieves player-level statistics aggregated by week for specified seasons.
        Includes passing, rushing, receiving, and fantasy statistics for each player's
        weekly performance.

        Key columns include:
        - Player info: player_id, player_name, team, position
        - Game context: season, week
        - Passing: passing_yards, passing_tds, interceptions, completions, attempts
        - Rushing: rushing_yards, rushing_tds, carries
        - Receiving: receiving_yards, receiving_tds, receptions, targets
        - Fantasy: fantasy_points, fantasy_points_ppr

        Args:
            seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
                     Weekly stats data is available from 1999 onwards.
            columns: List of column names to include (REQUIRED to manage response size).
            filters: Optional dict to filter on any column. Keys are column names,
                     values can be a single value or list of acceptable values.
                     Use describe_dataset("weekly_stats") to see available columns.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Weekly stats data as JSON with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get QB stats: get_weekly_stats([2024], ["player_name", "passing_yards", "passing_tds"])
            Filter by team: get_weekly_stats([2024], ["player_name", "rushing_yards"], filters={"team": "KC"})
            Filter by week: get_weekly_stats([2024], ["player_name", "fantasy_points"], filters={"week": [1, 2]})
            Paginate: get_weekly_stats([2024], ["player_name", "team"], offset=100, limit=50)
        """
        return get_weekly_stats_impl(seasons, columns, filters, offset, limit)

    @mcp.tool()
    def get_seasonal_stats(
        seasons: list[int],
        columns: Annotated[
            list[str],
            Field(
                description="List of column names to include in output (REQUIRED). "
                "Use describe_dataset('seasonal_stats') to see available columns. "
                "Common useful columns: player_id, season, passing_yards, passing_tds, "
                "rushing_yards, rushing_tds, receiving_yards, fantasy_points. "
                "NOTE: This dataset only has player_id, not player_name or team. "
                "To get player names, use weekly_stats or rosters to map player_id to names."
            ),
        ],
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"player_id": "00-0023459", "season": 2024}'
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
        """Get season-level player statistics.

        Retrieves player-level statistics aggregated for entire seasons. Useful for
        analyzing full-season performance, comparing players across seasons, and
        identifying statistical leaders.

        Key columns include:
        - Player info: player_id (NOTE: no player_name, team, or position in this dataset)
        - Season context: season
        - Passing: passing_yards, passing_tds, interceptions, completions, attempts
        - Rushing: rushing_yards, rushing_tds, carries
        - Receiving: receiving_yards, receiving_tds, receptions, targets
        - Fantasy: fantasy_points, fantasy_points_ppr
        - Games: games

        To get player names, first look up the player in weekly_stats or rosters,
        then use the player_id to query this dataset.

        Args:
            seasons: List of seasons (e.g., [2020, 2021, 2022]). Maximum 10 seasons allowed.
                     Seasonal stats data is available from 1999 onwards.
            columns: List of column names to include (REQUIRED to manage response size).
            filters: Optional dict to filter on any column. Keys are column names,
                     values can be a single value or list of acceptable values.
                     Use describe_dataset("seasonal_stats") to see available columns.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Seasonal stats data as JSON with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get stats: get_seasonal_stats([2024], ["player_id", "passing_yards", "rushing_yards"])
            Filter by player: get_seasonal_stats([2020, 2021], ["player_id", "fantasy_points"], filters={"player_id": "00-0023459"})
            Paginate: get_seasonal_stats([2024], ["player_id", "rushing_tds"], offset=100, limit=50)
        """
        return get_seasonal_stats_impl(seasons, columns, filters, offset, limit)

    @mcp.tool()
    def get_rosters(
        seasons: list[int],
        columns: Annotated[
            list[str],
            Field(
                description="List of column names to include in output (REQUIRED). "
                "Use describe_dataset('rosters') to see available columns. "
                "Common useful columns: player_id, player_name, team, position, "
                "jersey_number, status, height, weight, college"
            ),
        ],
        weeks: list[int] | None = None,
        teams: Annotated[
            list[str] | None,
            Field(
                description="Filter by team abbreviations (e.g., ['KC', 'SF']). "
                "Valid abbreviations: ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, "
                "DET, GB, HOU, IND, JAX, KC, LA, LAC, LV, MIA, MIN, NE, NO, NYG, NYJ, "
                "PHI, PIT, SEA, SF, TB, TEN, WAS"
            ),
        ] = None,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"position": "QB", "status": ["ACT", "RES"]}'
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit "
                "to page through results.",
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
        """Get NFL team roster data with player information.

        Retrieves roster data for specified seasons with optional filtering by
        weeks and teams. Roster data includes player details, positions, physical
        attributes, and roster status.

        Key columns include:
        - Player info: player_id, player_name, position, jersey_number
        - Team info: team, status (ACT=Active, RES=Reserve, etc.)
        - Physical: height, weight, age
        - Background: college, draft_year, draft_round, draft_pick
        - Seasonal: season, week

        Args:
            seasons: List of seasons (e.g., [2023, 2024]). Maximum 5 seasons allowed.
                     Roster data is available from 1999 onwards.
            columns: List of column names to include (REQUIRED to manage response size).
            weeks: Optional list of weeks to filter (1-18 for regular season).
                   If not specified, returns data for all weeks.
            teams: Optional list of team abbreviations to filter (e.g., ["KC", "SF"]).
                   Use standard NFL team abbreviations.
            filters: Optional dict to filter on any column. Keys are column names,
                     values can be a single value or list of acceptable values.
                     Use describe_dataset("rosters") to see available columns.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Roster data as JSON with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get roster data: get_rosters([2024], ["player_name", "team", "position"])
            Filter by team: get_rosters([2024], ["player_name", "position"], teams=["KC"])
            Filter by week: get_rosters([2024], ["player_name", "team"], weeks=[1])
            Filter QBs: get_rosters([2024], ["player_name", "team"], filters={"position": "QB"})
        """
        return get_rosters_impl(seasons, weeks, teams, filters, offset, limit, columns)

    @mcp.tool()
    def get_player_ids(
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit to "
                "page through results.",
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
        """Get cross-platform player ID mappings.

        Returns player identifiers across multiple platforms (ESPN, Yahoo, Sleeper,
        PFF, etc.) to help correlate player data from different sources. This is
        essential for joining data across different NFL data providers.

        Key columns include:
        - gsis_id: NFL's official player ID
        - espn_id: ESPN player identifier
        - yahoo_id: Yahoo fantasy player ID
        - sleeper_id: Sleeper fantasy player ID
        - pff_id: Pro Football Focus player ID
        - name: Player full name

        Args:
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Player ID mappings with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get first 100 players: get_player_ids()
            Paginate: get_player_ids(offset=100, limit=50)
        """
        return get_player_ids_impl(offset=offset, limit=limit)

    @mcp.tool()
    def lookup_player(
        name: Annotated[
            str,
            Field(
                description="Player name to search for (case-insensitive, partial match "
                "supported). Examples: 'Mahomes', 'Patrick Mahomes', 'jameis winston'"
            ),
        ],
        limit: Annotated[
            int | None,
            Field(
                description="Maximum number of results to return (default 10, max 100).",
                ge=1,
                le=100,
            ),
        ] = None,
    ) -> SuccessResponse | ErrorResponse:
        """Search for players by name and return their player_id (gsis_id).

        This utility tool enables searching for players by name to find their gsis_id,
        which can then be used with datasets that only contain player_id (like
        seasonal_stats). Useful when you need to look up a player's statistics but
        only know their name.

        Supports case-insensitive partial matching, so searching for "mahomes" will
        find "Patrick Mahomes", and "winston" will find "Jameis Winston".

        Returns:
        - gsis_id: The player_id used in other datasets (e.g., seasonal_stats)
        - name: Full player name
        - team: Current team abbreviation
        - position: Player position
        - merge_name: Lowercase normalized name for matching

        Args:
            name: Player name to search for (case-insensitive, partial match supported).
            limit: Maximum number of results to return (default 10, max 100).

        Returns:
            Matching players with their gsis_id and metadata. Use the gsis_id in
            other dataset queries that require player_id.

        Examples:
            Find a player: lookup_player("Mahomes")
            Case-insensitive: lookup_player("jameis winston")
            Partial match: lookup_player("Hill") -> finds Tyreek Hill, etc.

        Workflow:
            1. lookup_player("Jameis Winston") -> get gsis_id "00-0031355"
            2. get_seasonal_stats([2024], ["player_id", "passing_yards"],
               filters={"player_id": "00-0031355"})
        """
        return lookup_player_impl(name=name, limit=limit)

    @mcp.tool()
    def get_team_descriptions(
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit to "
                "page through results.",
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
        """Get NFL team metadata and information.

        Returns team details including abbreviations, full names, divisions,
        conferences, and other organizational information for all 32 NFL teams.
        Useful for mapping team abbreviations to full names and understanding
        team organization.

        Key columns include:
        - team_abbr: Standard team abbreviation (e.g., "KC", "SF")
        - team_name: Full team name (e.g., "Kansas City Chiefs")
        - team_conf: Conference (AFC or NFC)
        - team_division: Division within conference
        - team_color: Primary team color (hex)

        Args:
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Team descriptions with up to 100 rows by default.

        Examples:
            Get all teams: get_team_descriptions()
        """
        return get_team_descriptions_impl(offset=offset, limit=limit)

    @mcp.tool()
    def get_officials(
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit to "
                "page through results.",
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
        """Get NFL game officials data.

        Returns information about game officials including their names, positions,
        and the games they have officiated. Can be used to analyze officiating
        patterns or identify officials for specific games.

        Key columns include:
        - game_id: Unique game identifier
        - official_name: Name of the official
        - official_position: Position (referee, umpire, line judge, etc.)

        Args:
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Officials data with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get first 100 officials records: get_officials()
            Paginate: get_officials(offset=100, limit=50)
        """
        return get_officials_impl(offset=offset, limit=limit)

    @mcp.tool()
    def get_contracts(
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit to "
                "page through results.",
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
        """Get NFL player contract data.

        Returns player contract information including salary details, contract
        length, guarantees, and signing bonuses. Useful for salary cap analysis
        and understanding team financial commitments.

        Key columns include:
        - player: Player name
        - team: Current team
        - value: Total contract value
        - apy: Average per year salary
        - guaranteed: Guaranteed money
        - years: Contract length

        Args:
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Contract data with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get first 100 contracts: get_contracts()
            Paginate: get_contracts(offset=100, limit=50)
        """
        return get_contracts_impl(offset=offset, limit=limit)

    @mcp.tool()
    def get_draft_picks(
        seasons: list[int],
        columns: Annotated[
            list[str] | None,
            Field(
                description="List of column names to include in output (optional). "
                "Use describe_dataset('draft_picks') to see available columns. "
                "Common useful columns: season, round, pick, team, pfr_player_name, position"
            ),
        ] = None,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"team": "KC", "round": [1, 2]}'
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit to "
                "page through results.",
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
        """Get historical NFL draft picks data.

        Retrieves draft pick information for the specified seasons, including
        player selection details, team information, and draft positions.

        Key columns include:
        - season: Draft year
        - round: Draft round (1-7)
        - pick: Overall pick number
        - team: Team that made the selection
        - pfr_player_name: Player name
        - position: Player position
        - age: Player age at draft time
        - college: College attended

        Args:
            seasons: List of seasons (e.g., [2020, 2021, 2022]). Maximum 20 seasons allowed.
                     Draft data is available from 1999 onwards.
            columns: List of column names to include in output (optional).
            filters: Optional dict to filter on any column. Keys are column names,
                     values can be a single value or list of acceptable values.
                     Use describe_dataset("draft_picks") to see available columns.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Draft picks data as JSON with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get draft picks: get_draft_picks([2024])
            Filter by team: get_draft_picks([2024], filters={"team": "KC"})
            First round only: get_draft_picks([2024], filters={"round": 1})
            Paginate: get_draft_picks([2024], offset=100, limit=50)
        """
        return get_draft_picks_impl(seasons, columns, filters, offset, limit)

    @mcp.tool()
    def get_schedules(
        seasons: list[int],
        columns: Annotated[
            list[str] | None,
            Field(
                description="List of column names to include in output (optional). "
                "Use describe_dataset('schedules') to see available columns. "
                "Common useful columns: game_id, season, week, home_team, away_team, "
                "home_score, away_score, gameday"
            ),
        ] = None,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"home_team": "KC", "week": [1, 2, 3]}'
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit to "
                "page through results.",
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
        """Get NFL game schedules and results.

        Retrieves game schedule information for the specified seasons, including
        matchups, dates, scores, and game outcomes.

        Key columns include:
        - game_id: Unique game identifier
        - season: Season year
        - week: Week number
        - gameday: Date of the game
        - home_team: Home team abbreviation
        - away_team: Away team abbreviation
        - home_score: Home team final score
        - away_score: Away team final score

        Args:
            seasons: List of seasons (e.g., [2020, 2021, 2022]). Maximum 10 seasons allowed.
                     Schedule data is available from 1999 onwards.
            columns: List of column names to include in output (optional).
            filters: Optional dict to filter on any column. Keys are column names,
                     values can be a single value or list of acceptable values.
                     Use describe_dataset("schedules") to see available columns.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 100, max 100).

        Returns:
            Schedule data as JSON with up to 100 rows by default. Use offset and limit
            to paginate through larger result sets.

        Examples:
            Get schedules: get_schedules([2024])
            Filter by team: get_schedules([2024], filters={"home_team": "KC"})
            Filter by week: get_schedules([2024], filters={"week": [1, 2]})
            Paginate: get_schedules([2024], offset=100, limit=50)
        """
        return get_schedules_impl(seasons, columns, filters, offset, limit)

    @mcp.tool()
    def get_bdb_games(
        columns: Annotated[
            list[str] | None,
            Field(
                description="List of column names to include in output (optional). "
                "Common columns: game_id, season, week, home_team_abbr, visitor_team_abbr, "
                "game_date, game_time_eastern"
            ),
        ] = None,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"home_team_abbr": "KC", "week": [1, 2, 3]}'
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit "
                "to page through results.",
                ge=0,
            ),
        ] = 0,
        limit: Annotated[
            int | None,
            Field(
                description="Maximum number of rows to return (default 50, max 100).",
                ge=1,
                le=100,
            ),
        ] = None,
    ) -> SuccessResponse | ErrorResponse:
        """Get NFL Big Data Bowl game metadata.

        Retrieves game information from the NFL Big Data Bowl 2026 dataset.
        This dataset contains tracking data from the 2023 NFL season.

        IMPORTANT: Requires Kaggle authentication. You must:
        1. Create ~/.kaggle/access_token with your API token
        2. Accept competition rules at: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules

        Key columns include:
        - game_id: Unique game identifier
        - season: Season year
        - week: Week number (1-18)
        - home_team_abbr: Home team abbreviation
        - visitor_team_abbr: Away team abbreviation
        - game_date: Date of the game
        - game_time_eastern: Game start time (Eastern)

        Args:
            columns: List of column names to include in output (optional).
            filters: Optional dict to filter on any column.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 50, max 100).

        Returns:
            Game metadata as JSON with up to 50 rows by default.

        Examples:
            Get all games: get_bdb_games()
            Filter by team: get_bdb_games(filters={"home_team_abbr": "KC"})
            Filter by week: get_bdb_games(filters={"week": [1, 2, 3]})
        """
        return get_bdb_games_impl(columns, filters, offset, limit)

    @mcp.tool()
    def get_bdb_plays(
        game_id: Annotated[
            int | None,
            Field(
                description="Game ID to filter plays for a specific game. "
                "Use get_bdb_games to find game IDs."
            ),
        ] = None,
        columns: Annotated[
            list[str] | None,
            Field(
                description="List of column names to include in output (optional). "
                "Common columns: game_id, play_id, play_description, quarter, down, "
                "yards_to_go, possession_team, defensive_team, yardline_number"
            ),
        ] = None,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"possession_team": "KC", "down": [3, 4]}'
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit "
                "to page through results.",
                ge=0,
            ),
        ] = 0,
        limit: Annotated[
            int | None,
            Field(
                description="Maximum number of rows to return (default 50, max 100).",
                ge=1,
                le=100,
            ),
        ] = None,
    ) -> SuccessResponse | ErrorResponse:
        """Get NFL Big Data Bowl play-level data.

        Retrieves play information from the NFL Big Data Bowl 2026 dataset.
        Includes play descriptions, game situation, advanced metrics, and outcomes.

        IMPORTANT: Requires Kaggle authentication. You must:
        1. Create ~/.kaggle/access_token with your API token
        2. Accept competition rules at: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules

        Key columns include:
        - game_id: Game identifier
        - play_id: Play identifier (unique within a game)
        - play_description: Text description of the play
        - quarter, down, yards_to_go: Game situation
        - possession_team, defensive_team: Teams involved
        - pass_result, yards_gained: Play outcome
        - expected_points_added: EPA metric

        Args:
            game_id: Optional game ID to filter plays for a specific game.
            columns: List of column names to include in output (optional).
            filters: Optional dict to filter on any column.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 50, max 100).

        Returns:
            Play data as JSON with up to 50 rows by default.

        Examples:
            Get plays for a game: get_bdb_plays(game_id=2023090700)
            Filter by down: get_bdb_plays(filters={"down": [3, 4]})
            Filter by team: get_bdb_plays(filters={"possession_team": "KC"})
        """
        return get_bdb_plays_impl(game_id, columns, filters, offset, limit)

    @mcp.tool()
    def get_bdb_players(
        columns: Annotated[
            list[str] | None,
            Field(
                description="List of column names to include in output (optional). "
                "Common columns: nfl_id, player_name, player_position, player_height, player_weight"
            ),
        ] = None,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"player_position": ["QB", "WR"]}'
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit "
                "to page through results.",
                ge=0,
            ),
        ] = 0,
        limit: Annotated[
            int | None,
            Field(
                description="Maximum number of rows to return (default 50, max 100).",
                ge=1,
                le=100,
            ),
        ] = None,
    ) -> SuccessResponse | ErrorResponse:
        """Get NFL Big Data Bowl player information.

        Retrieves player data from the NFL Big Data Bowl 2026 dataset.
        Includes player names, positions, and physical attributes.

        IMPORTANT: Requires Kaggle authentication. You must:
        1. Create ~/.kaggle/access_token with your API token
        2. Accept competition rules at: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules

        Key columns include:
        - nfl_id: NFL player identifier
        - player_name: Player name
        - player_position: Player position
        - player_height: Player height
        - player_weight: Player weight in pounds

        Args:
            columns: List of column names to include in output (optional).
            filters: Optional dict to filter on any column.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum number of rows to return (default 50, max 100).

        Returns:
            Player data as JSON with up to 50 rows by default.

        Examples:
            Get all players: get_bdb_players()
            Filter by position: get_bdb_players(filters={"player_position": ["QB", "WR"]})
            Select columns: get_bdb_players(columns=["nfl_id", "player_name", "player_position"])
        """
        return get_bdb_players_impl(columns, filters, offset, limit)

    @mcp.tool()
    def get_bdb_tracking(
        week: Annotated[
            int,
            Field(
                description=f"Week number ({min(BDB_AVAILABLE_WEEKS)}-{max(BDB_AVAILABLE_WEEKS)}). "
                "Required. Each week is stored in a separate file.",
                ge=min(BDB_AVAILABLE_WEEKS),
                le=max(BDB_AVAILABLE_WEEKS),
            ),
        ],
        game_id: Annotated[
            int | None,
            Field(
                description="Game ID to filter. STRONGLY RECOMMENDED to avoid slow queries. "
                "Use get_bdb_games to find game IDs."
            ),
        ] = None,
        play_id: Annotated[
            int | None,
            Field(
                description="Play ID to filter. Requires game_id to be specified. "
                "Use get_bdb_plays to find play IDs within a game."
            ),
        ] = None,
        nfl_id: Annotated[
            int | None,
            Field(
                description="NFL player ID to filter for a specific player. "
                "Use get_bdb_players to find player IDs."
            ),
        ] = None,
        columns: Annotated[
            list[str] | None,
            Field(
                description="List of column names to include in output (optional). "
                "Common columns: game_id, play_id, nfl_id, player_name, frame_id, "
                "x, y, s (speed), a (acceleration), o (orientation), dir (direction)"
            ),
        ] = None,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description="Filter on any column. Keys are column names, values are "
                "either a single value or list of acceptable values. "
                'Example: {"event": "pass_forward"}'
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                description="Number of rows to skip for pagination. Use with limit "
                "to page through results.",
                ge=0,
            ),
        ] = 0,
        limit: Annotated[
            int | None,
            Field(
                description="Maximum number of rows to return (default 50, max 100). "
                "Tracking data is very large - use filters to narrow results.",
                ge=1,
                le=100,
            ),
        ] = None,
    ) -> SuccessResponse | ErrorResponse:
        """Get NFL Big Data Bowl per-frame tracking data.

        Retrieves high-frequency player tracking data from the NFL Big Data Bowl
        2026 dataset. Each row represents a single frame with player position,
        speed, acceleration, and orientation.

        IMPORTANT:
        - Requires Kaggle authentication (~/.kaggle/access_token)
        - Accept competition rules: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules
        - Tracking data is VERY LARGE. Always filter by game_id and/or play_id.

        Key columns include:
        - game_id: Game identifier
        - play_id: Play identifier
        - nfl_id: Player identifier
        - player_name: Player name
        - frame_id: Frame number within the play
        - x: X position on field (0-120 yards, including end zones)
        - y: Y position on field (0-53.3 yards)
        - s: Speed in yards/second
        - a: Acceleration in yards/second^2
        - o: Orientation (0-360 degrees, direction player is facing)
        - dir: Direction of movement (0-360 degrees)
        - player_position, player_role: Player context

        Args:
            week: Week number (1-18). Required.
            game_id: Game ID to filter. Strongly recommended.
            play_id: Play ID to filter. Requires game_id.
            nfl_id: NFL player ID to filter for a specific player.
            columns: List of column names to include in output (optional).
            filters: Optional dict to filter on any column.
            offset: Number of rows to skip for pagination (default 0).
            limit: Maximum rows to return (default 50, max 100).

        Returns:
            Tracking data as JSON with up to 50 rows by default.

        Examples:
            Get tracking for a play: get_bdb_tracking(week=1, game_id=2023090700, play_id=101)
            Get specific player: get_bdb_tracking(week=1, game_id=2023090700, nfl_id=43290)
            Filter by position: get_bdb_tracking(week=1, game_id=2023090700, filters={"player_position": "QB"})
        """
        return get_bdb_tracking_impl(
            week, game_id, play_id, nfl_id, columns, filters, offset, limit
        )
