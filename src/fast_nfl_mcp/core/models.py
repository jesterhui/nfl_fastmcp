"""Pydantic models for standardized API responses.

This module defines the response schemas used by all MCP tools in the
Fast NFL MCP server. All responses follow a consistent format optimized
for token efficiency when used with AI assistants.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class ColumnSchema:
    """Schema information for a single column in a dataset.

    Attributes:
        name: The column name.
        dtype: The pandas dtype as a string (e.g., "int64", "float64").
        sample_values: A list of example values from the column (up to 5).
        null_count: The number of null/NaN values in the column.
        unique_count: The number of unique values in the column.
    """

    name: str
    dtype: str
    sample_values: list[Any] = field(default_factory=list)
    null_count: int = 0
    unique_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "sample_values": self.sample_values,
            "null_count": self.null_count,
            "unique_count": self.unique_count,
        }


@dataclass
class DatasetSchema:
    """Schema information for an entire dataset.

    Attributes:
        name: The dataset name (e.g., "play_by_play", "weekly_stats").
        description: A brief description of the dataset.
        columns: List of ColumnSchema objects describing each column.
        row_count: Total number of rows in the dataset.
        available_seasons: List of available seasons, or None if N/A.
    """

    name: str
    description: str
    columns: list[ColumnSchema] = field(default_factory=list)
    row_count: int = 0
    available_seasons: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "columns": [col.to_dict() for col in self.columns],
            "row_count": self.row_count,
            "available_seasons": self.available_seasons,
        }


class ResponseMetadata(BaseModel):
    """Metadata included in all API responses.

    Attributes:
        row_count: Number of rows returned in this response.
        total_available: Total rows available before truncation (optional).
        truncated: Whether the response was truncated due to row limits.
        columns: List of column names in the response data (optional).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    row_count: int = Field(default=0, ge=0, description="Number of rows returned")
    total_available: int | None = Field(
        default=None,
        ge=0,
        description="Total rows available before truncation",
    )
    truncated: bool = Field(
        default=False,
        description="Whether the response was truncated",
    )
    columns: list[str] | None = Field(
        default=None,
        description="List of column names in the data",
    )


class SuccessResponse(BaseModel):
    """Response model for successful API calls.

    This model is used when a tool successfully retrieves data,
    even if the result set is empty (e.g., no matching rows).

    Attributes:
        success: Always True for successful responses.
        data: List of row dictionaries containing the response data.
        metadata: Response metadata including row counts and column info.
        warning: Optional warning message (e.g., for invalid parameters).
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    success: bool = Field(default=True, description="Indicates successful response")
    data: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of row dictionaries",
    )
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Response metadata",
    )
    warning: str | None = Field(
        default=None,
        description="Optional warning message",
    )


class ErrorResponse(BaseModel):
    """Response model for failed API calls.

    This model is used when a tool encounters an error that prevents
    it from returning data (e.g., network errors, invalid dataset names).

    Attributes:
        success: Always False for error responses.
        data: Always an empty list for error responses.
        metadata: Minimal metadata with row_count=0.
        error: Description of what went wrong.
        warning: Optional additional warning message.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    success: bool = Field(default=False, description="Always False for errors")
    data: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Always empty for error responses",
    )
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Response metadata (row_count=0)",
    )
    error: str = Field(description="Error message describing what went wrong")
    warning: str | None = Field(
        default=None,
        description="Optional additional warning",
    )


def create_success_response(
    data: list[dict[str, Any]],
    total_available: int | None = None,
    truncated: bool = False,
    columns: list[str] | None = None,
    warning: str | None = None,
) -> SuccessResponse:
    """Factory function to create a SuccessResponse.

    Args:
        data: List of row dictionaries to include in the response.
        total_available: Total number of rows before truncation.
        truncated: Whether the data was truncated.
        columns: List of column names in the data.
        warning: Optional warning message.

    Returns:
        A properly constructed SuccessResponse.
    """
    metadata = ResponseMetadata(
        row_count=len(data),
        total_available=total_available,
        truncated=truncated,
        columns=columns,
    )
    return SuccessResponse(
        success=True,
        data=data,
        metadata=metadata,
        warning=warning,
    )


def create_error_response(
    error: str,
    warning: str | None = None,
) -> ErrorResponse:
    """Factory function to create an ErrorResponse.

    Args:
        error: Description of the error that occurred.
        warning: Optional additional warning message.

    Returns:
        A properly constructed ErrorResponse.
    """
    metadata = ResponseMetadata(row_count=0)
    return ErrorResponse(
        success=False,
        data=[],
        metadata=metadata,
        error=error,
        warning=warning,
    )
