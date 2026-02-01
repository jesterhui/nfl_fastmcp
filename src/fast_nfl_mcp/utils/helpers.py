"""Utility functions for the Fast NFL MCP server.

This module provides shared utility functions used across multiple modules,
including response building helpers and warning merging utilities.
"""

from fast_nfl_mcp.core.models import SuccessResponse, create_success_response


def merge_warnings(existing_warning: str | None, *warnings: str) -> str | None:
    """Merge multiple warning messages into a single string.

    Combines an existing warning (if any) with additional warning messages,
    filtering out empty strings and None values.

    Args:
        existing_warning: The existing warning message (may be None).
        *warnings: Additional warning messages to merge.

    Returns:
        A combined warning string, or None if no warnings.

    Examples:
        >>> merge_warnings(None)
        None
        >>> merge_warnings("Warning 1", "Warning 2")
        "Warning 1 Warning 2"
        >>> merge_warnings(None, "Warning 1")
        "Warning 1"
        >>> merge_warnings("Existing", "")
        "Existing"
    """
    all_warnings = [existing_warning or "", *warnings]
    combined = " ".join(filter(None, all_warnings))
    return combined if combined else None


def add_warnings_to_response(
    result: SuccessResponse,
    warnings: list[str],
) -> SuccessResponse:
    """Add validation warnings to a SuccessResponse.

    Creates a new SuccessResponse with the combined warning messages
    from the original response and the provided warnings list.

    Args:
        result: The original SuccessResponse.
        warnings: List of warning messages to add.

    Returns:
        A new SuccessResponse with combined warnings.
    """
    if not warnings:
        return result

    combined_warning = merge_warnings(result.warning, *warnings)
    return create_success_response(
        data=result.data,
        total_available=result.metadata.total_available,
        truncated=result.metadata.truncated,
        columns=result.metadata.columns,
        warning=combined_warning,
    )
