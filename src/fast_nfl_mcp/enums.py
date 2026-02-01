"""Enumerations for the Fast NFL MCP server.

This module provides StrEnum classes for commonly used string constants,
improving type safety and IDE support while maintaining backward compatibility
with string-based JSON responses.
"""

from enum import StrEnum


class DatasetName(StrEnum):
    """NFL dataset names available through the MCP server."""

    PLAY_BY_PLAY = "play_by_play"
    WEEKLY_STATS = "weekly_stats"
    SEASONAL_STATS = "seasonal_stats"
    ROSTERS = "rosters"
    PLAYER_IDS = "player_ids"
    DRAFT_PICKS = "draft_picks"
    SCHEDULES = "schedules"
    TEAM_DESCRIPTIONS = "team_descriptions"
    COMBINE_DATA = "combine_data"
    SCORING_LINES = "scoring_lines"
    WIN_TOTALS = "win_totals"
    NGS_PASSING = "ngs_passing"
    NGS_RUSHING = "ngs_rushing"
    NGS_RECEIVING = "ngs_receiving"
    SNAP_COUNTS = "snap_counts"
    INJURIES = "injuries"
    DEPTH_CHARTS = "depth_charts"
    CONTRACTS = "contracts"
    OFFICIALS = "officials"
    QBR = "qbr"


class DatasetStatus(StrEnum):
    """Status values for dataset loading state."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_LOADED = "not_loaded"
