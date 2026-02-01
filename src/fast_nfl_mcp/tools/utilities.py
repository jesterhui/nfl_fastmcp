"""Utility MCP tools for dataset discovery and schema inspection.

This module provides tools that allow AI assistants to discover available
NFL datasets and inspect their schemas. These are foundational tools that
support the data query workflow.
"""

from typing import Any

from fast_nfl_mcp.enums import DatasetStatus
from fast_nfl_mcp.models import (
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from fast_nfl_mcp.schema_manager import DATASET_DEFINITIONS, SchemaManager


def list_datasets_impl(schema_manager: SchemaManager) -> SuccessResponse:
    """List all available NFL datasets with their descriptions.

    Returns a list of all available datasets that can be queried,
    including their names, descriptions, and whether they support
    season-based filtering.

    Args:
        schema_manager: The SchemaManager instance with preloaded schemas.

    Returns:
        A SuccessResponse containing dataset information.
    """
    datasets = []
    for name, definition in DATASET_DEFINITIONS.items():
        # Determine loading status
        if schema_manager.is_loaded(name):
            status = DatasetStatus.AVAILABLE
        elif name in schema_manager.get_failed_datasets():
            status = DatasetStatus.UNAVAILABLE
        else:
            status = DatasetStatus.NOT_LOADED

        dataset_info: dict[str, Any] = {
            "name": name,
            "description": definition.description,
            "supports_seasons": definition.supports_seasons,
            "status": status,
        }

        datasets.append(dataset_info)

    # Sort alphabetically by name for consistent ordering
    datasets.sort(key=lambda d: d["name"])

    return create_success_response(
        data=datasets,
        columns=["name", "description", "supports_seasons", "status"],
    )


def describe_dataset_impl(
    schema_manager: SchemaManager, dataset: str
) -> SuccessResponse | ErrorResponse:
    """Get detailed schema information for a specific dataset.

    Returns column names, data types, sample values, and other metadata
    for the specified dataset. Use this to understand what data is available
    before querying.

    Args:
        schema_manager: The SchemaManager instance with preloaded schemas.
        dataset: The name of the dataset to describe (e.g., "play_by_play").

    Returns:
        A SuccessResponse with schema information, or ErrorResponse if dataset not found.
    """
    # Check if dataset exists in definitions
    if dataset not in DATASET_DEFINITIONS:
        available = list(DATASET_DEFINITIONS.keys())
        return create_error_response(
            error=f"Unknown dataset: '{dataset}'",
            warning=f"Available datasets: {', '.join(sorted(available))}",
        )

    # Get the cached schema
    schema = schema_manager.get_schema(dataset)

    if schema is None:
        # Dataset exists but schema wasn't loaded
        if dataset in schema_manager.get_failed_datasets():
            return create_error_response(
                error=f"Dataset '{dataset}' failed to load during startup",
                warning="This dataset may be temporarily unavailable. Try again later.",
            )
        else:
            return create_error_response(
                error=f"Schema for '{dataset}' is not loaded",
                warning="The server may still be initializing. Try again shortly.",
            )

    # Build a more AI-friendly response structure
    response_data: dict[str, Any] = {
        "dataset": schema.name,
        "description": schema.description,
        "row_count_sample": schema.row_count,
        "available_seasons": schema.available_seasons,
        "column_count": len(schema.columns),
        "columns": [],
    }

    # Add column details
    for col in schema.columns:
        col_info: dict[str, Any] = {
            "name": col.name,
            "dtype": col.dtype,
            "sample_values": col.sample_values,
            "null_count": col.null_count,
            "unique_count": col.unique_count,
        }
        response_data["columns"].append(col_info)

    return create_success_response(
        data=[response_data],
        columns=list(response_data.keys()),
    )
