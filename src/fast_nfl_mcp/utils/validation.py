"""Shared validation utilities for NFL data tools.

This module provides common validation functions used across multiple
tool modules to reduce code duplication.
"""

from typing import Any

from fast_nfl_mcp.utils.constants import (
    MAX_WEEK,
    MIN_SEASON,
    MIN_WEEK,
    VALID_TEAM_ABBREVIATIONS,
)


def validate_seasons(
    seasons: list[int],
    max_seasons: int,
    dataset_name: str = "Data",
) -> tuple[list[int], str | None]:
    """Validate the seasons parameter with a specified limit.

    Args:
        seasons: List of season years to validate.
        max_seasons: Maximum number of seasons allowed.
        dataset_name: Name of the dataset for error messages.

    Returns:
        A tuple of (valid_seasons, warning_message).
        warning_message is None if all seasons are valid.
    """
    if not seasons:
        return [], "No seasons provided. Please specify at least one season."

    warnings: list[str] = []

    # First filter out invalid seasons (before applying max limit)
    valid_seasons = []
    invalid_seasons = []
    for season in seasons:
        if season >= MIN_SEASON:
            valid_seasons.append(season)
        else:
            invalid_seasons.append(season)

    if invalid_seasons:
        warnings.append(
            f"Invalid seasons removed: {invalid_seasons}. "
            f"{dataset_name} is available from {MIN_SEASON} onwards."
        )

    # Then apply max_seasons limit to valid seasons only
    if len(valid_seasons) > max_seasons:
        warnings.append(
            f"Too many seasons requested ({len(valid_seasons)} valid). "
            f"Limited to {max_seasons} seasons: {valid_seasons[:max_seasons]}"
        )
        valid_seasons = valid_seasons[:max_seasons]

    warning = " ".join(warnings) if warnings else None
    return valid_seasons, warning


def validate_weeks(weeks: list[int] | None) -> tuple[list[int] | None, str | None]:
    """Validate the weeks parameter.

    Args:
        weeks: Optional list of week numbers to validate.

    Returns:
        A tuple of (valid_weeks, warning_message).
        warning_message is None if all weeks are valid.
    """
    if weeks is None:
        return None, None

    if not weeks:
        return None, None

    # Filter out invalid weeks
    valid_weeks = []
    invalid_weeks = []
    for week in weeks:
        if MIN_WEEK <= week <= MAX_WEEK:
            valid_weeks.append(week)
        else:
            invalid_weeks.append(week)

    warning = None
    if invalid_weeks:
        warning = (
            f"Invalid weeks removed: {invalid_weeks}. "
            f"Valid week range is {MIN_WEEK}-{MAX_WEEK}."
        )

    return valid_weeks if valid_weeks else None, warning


def validate_teams(teams: list[str] | None) -> tuple[list[str] | None, str | None]:
    """Validate team abbreviations.

    Args:
        teams: Optional list of team abbreviations to validate.

    Returns:
        A tuple of (valid_teams, warning_message).
        warning_message is None if all teams are valid.
    """
    if teams is None:
        return None, None

    if not teams:
        return None, None

    # Normalize to uppercase and validate
    valid_teams = []
    invalid_teams = []
    for team in teams:
        team_upper = team.upper()
        if team_upper in VALID_TEAM_ABBREVIATIONS:
            valid_teams.append(team_upper)
        else:
            invalid_teams.append(team)

    warning = None
    if invalid_teams:
        warning = (
            f"Invalid team abbreviations removed: {invalid_teams}. "
            f"Use standard NFL team abbreviations (e.g., KC, SF, TB)."
        )

    return valid_teams if valid_teams else None, warning


def normalize_filters(
    filters: dict[str, Any] | None,
) -> dict[str, list[Any]]:
    """Normalize filter values to lists.

    Args:
        filters: Dict mapping column names to filter values.
                 Values can be single items or lists.

    Returns:
        Dict with all values normalized to lists.
    """
    if filters is None:
        return {}

    normalized: dict[str, list[Any]] = {}
    for column, value in filters.items():
        if isinstance(value, list):
            normalized[column] = value
        else:
            normalized[column] = [value]
    return normalized
