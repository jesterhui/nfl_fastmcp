"""Serialization utilities for DataFrame to JSON conversion.

This module provides shared utilities for converting pandas DataFrames
to JSON-serializable records, handling NaN values, numpy scalars, and
timestamps consistently across the codebase.
"""

from typing import Any

import pandas as pd


def convert_value(val: Any) -> Any:
    """Convert a single value to a JSON-serializable type.

    Handles the following conversions:
    - NaN/NA/None -> None
    - pd.Timestamp -> string
    - numpy scalars (via .item()) -> native Python types

    Args:
        val: The value to convert.

    Returns:
        A JSON-serializable value.
    """
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return str(val)
    if hasattr(val, "item"):
        # numpy scalar types (int64, float64, bool_, etc.)
        return val.item()
    return val


def convert_dataframe_to_records(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert a DataFrame to a list of JSON-serializable dictionaries.

    Handles special cases like NaN values, numpy types, and timestamps
    to ensure JSON serialization compatibility.

    Args:
        df: The DataFrame to convert.

    Returns:
        A tuple of (list of row dictionaries, list of column names).
        Returns ([], []) if df is None or empty.
    """
    if df is None or df.empty:
        return [], []

    # Pre-compute column names once (avoid repeated str() calls in loops)
    columns: list[str] = [str(col) for col in df.columns]

    # Create a working copy for vectorized transformations
    result_df = df.copy()

    # Vectorized type conversions per column (much faster than per-cell checks)
    for col in result_df.columns:
        dtype = result_df[col].dtype

        # Convert Timestamp columns to strings, handling NaT properly
        if pd.api.types.is_datetime64_any_dtype(dtype):
            # Use list comprehension to properly handle NaT as None
            values = [None if pd.isna(v) else str(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)
        # Convert numpy integer/float types to Python native (preserving NaN as None)
        # Must use list comprehension + .item() to ensure native Python types
        elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            values = [convert_value(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)
        # Convert boolean columns - handles nullable BooleanDtype with pd.NA
        # Must convert to object dtype to preserve None values
        elif pd.api.types.is_bool_dtype(dtype):
            values = [convert_value(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)
        # Handle string and object dtype columns - replace NaN with None
        # and convert any embedded Timestamps/numpy scalars
        # Must rebuild Series from list to preserve None (pandas converts None to nan)
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            values = [convert_value(v) for v in result_df[col]]
            result_df[col] = pd.Series(values, index=result_df.index, dtype=object)

    # Convert to records - now all values should be JSON-serializable
    records = result_df.to_dict(orient="records")

    # Ensure all keys are strings (handle non-string column names)
    cleaned_records: list[dict[str, Any]] = [
        {str(k): v for k, v in record.items()} for record in records
    ]

    return cleaned_records, columns


def extract_sample_values(series: pd.Series, max_samples: int = 2) -> list[Any]:
    """Extract sample values from a pandas Series.

    Args:
        series: The pandas Series to extract samples from.
        max_samples: Maximum number of samples to extract (default 2).

    Returns:
        A list of JSON-serializable sample values, with NaN values excluded.
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
        converted = convert_value(val)
        if converted is not None:
            result.append(converted)

    return result[:max_samples]
