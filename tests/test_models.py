"""Tests for response models.

This module tests the Pydantic models and dataclasses used for
standardized API responses in the Fast NFL MCP server.
"""

import json

import pytest
from pydantic import ValidationError

from fast_nfl_mcp.models import (
    ColumnSchema,
    DatasetSchema,
    ErrorResponse,
    ResponseMetadata,
    SuccessResponse,
    create_error_response,
    create_success_response,
)


class TestColumnSchema:
    """Tests for the ColumnSchema dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating a ColumnSchema with only required fields."""
        col = ColumnSchema(name="player_id", dtype="object")

        assert col.name == "player_id"
        assert col.dtype == "object"
        assert col.sample_values == []
        assert col.null_count == 0
        assert col.unique_count == 0

    def test_create_with_all_fields(self) -> None:
        """Test creating a ColumnSchema with all fields."""
        col = ColumnSchema(
            name="rushing_yards",
            dtype="int64",
            sample_values=[100, 85, 120, 45, 200],
            null_count=5,
            unique_count=150,
        )

        assert col.name == "rushing_yards"
        assert col.dtype == "int64"
        assert col.sample_values == [100, 85, 120, 45, 200]
        assert col.null_count == 5
        assert col.unique_count == 150

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        col = ColumnSchema(
            name="epa",
            dtype="float64",
            sample_values=[0.5, -0.3, 1.2],
            null_count=10,
            unique_count=500,
        )

        result = col.to_dict()

        assert result == {
            "name": "epa",
            "dtype": "float64",
            "sample_values": [0.5, -0.3, 1.2],
            "null_count": 10,
            "unique_count": 500,
        }

    def test_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        col = ColumnSchema(
            name="test",
            dtype="object",
            sample_values=["a", "b", "c"],
        )

        # Should not raise
        json_str = json.dumps(col.to_dict())
        assert isinstance(json_str, str)


class TestDatasetSchema:
    """Tests for the DatasetSchema dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating a DatasetSchema with only required fields."""
        schema = DatasetSchema(
            name="play_by_play",
            description="Play-by-play data with EPA, WPA",
        )

        assert schema.name == "play_by_play"
        assert schema.description == "Play-by-play data with EPA, WPA"
        assert schema.columns == []
        assert schema.row_count == 0
        assert schema.available_seasons is None

    def test_create_with_all_fields(self) -> None:
        """Test creating a DatasetSchema with all fields."""
        columns = [
            ColumnSchema(name="game_id", dtype="object"),
            ColumnSchema(name="play_id", dtype="int64"),
        ]
        schema = DatasetSchema(
            name="weekly_stats",
            description="Weekly player statistics",
            columns=columns,
            row_count=50000,
            available_seasons=[2020, 2021, 2022, 2023, 2024],
        )

        assert schema.name == "weekly_stats"
        assert len(schema.columns) == 2
        assert schema.row_count == 50000
        assert schema.available_seasons == [2020, 2021, 2022, 2023, 2024]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        columns = [
            ColumnSchema(name="col1", dtype="int64"),
        ]
        schema = DatasetSchema(
            name="test_dataset",
            description="Test description",
            columns=columns,
            row_count=100,
            available_seasons=[2023],
        )

        result = schema.to_dict()

        assert result["name"] == "test_dataset"
        assert result["description"] == "Test description"
        assert len(result["columns"]) == 1
        assert result["columns"][0]["name"] == "col1"
        assert result["row_count"] == 100
        assert result["available_seasons"] == [2023]

    def test_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        schema = DatasetSchema(
            name="test",
            description="Test",
            columns=[ColumnSchema(name="a", dtype="int64")],
            available_seasons=[2023, 2024],
        )

        json_str = json.dumps(schema.to_dict())
        assert isinstance(json_str, str)


class TestResponseMetadata:
    """Tests for the ResponseMetadata Pydantic model."""

    def test_create_default(self) -> None:
        """Test creating ResponseMetadata with defaults."""
        meta = ResponseMetadata()

        assert meta.row_count == 0
        assert meta.total_available is None
        assert meta.truncated is False
        assert meta.columns is None

    def test_create_with_values(self) -> None:
        """Test creating ResponseMetadata with all values."""
        meta = ResponseMetadata(
            row_count=100,
            total_available=45000,
            truncated=True,
            columns=["game_id", "play_id", "epa"],
        )

        assert meta.row_count == 100
        assert meta.total_available == 45000
        assert meta.truncated is True
        assert meta.columns == ["game_id", "play_id", "epa"]

    def test_immutable(self) -> None:
        """Test that ResponseMetadata is frozen/immutable."""
        meta = ResponseMetadata(row_count=50)

        with pytest.raises(ValidationError):
            meta.row_count = 100  # type: ignore[misc]

    def test_negative_row_count_rejected(self) -> None:
        """Test that negative row counts are rejected."""
        with pytest.raises(ValidationError):
            ResponseMetadata(row_count=-1)

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ResponseMetadata(row_count=10, extra_field="test")  # type: ignore[call-arg]

    def test_json_serialization(self) -> None:
        """Test JSON serialization via model_dump_json."""
        meta = ResponseMetadata(
            row_count=50,
            total_available=1000,
            truncated=True,
            columns=["a", "b"],
        )

        json_str = meta.model_dump_json()
        data = json.loads(json_str)

        assert data["row_count"] == 50
        assert data["total_available"] == 1000
        assert data["truncated"] is True
        assert data["columns"] == ["a", "b"]


class TestSuccessResponse:
    """Tests for the SuccessResponse Pydantic model."""

    def test_create_default(self) -> None:
        """Test creating SuccessResponse with defaults."""
        resp = SuccessResponse()

        assert resp.success is True
        assert resp.data == []
        assert resp.metadata.row_count == 0
        assert resp.warning is None

    def test_create_with_data(self) -> None:
        """Test creating SuccessResponse with data."""
        data = [
            {"player": "Patrick Mahomes", "yards": 300},
            {"player": "Josh Allen", "yards": 280},
        ]
        meta = ResponseMetadata(row_count=2, columns=["player", "yards"])

        resp = SuccessResponse(data=data, metadata=meta)

        assert resp.success is True
        assert len(resp.data) == 2
        assert resp.data[0]["player"] == "Patrick Mahomes"
        assert resp.metadata.row_count == 2

    def test_create_with_warning(self) -> None:
        """Test creating SuccessResponse with a warning."""
        resp = SuccessResponse(
            data=[],
            metadata=ResponseMetadata(row_count=0),
            warning="Season 1950 is not available. Valid range: 1999-2024",
        )

        assert resp.success is True
        assert resp.data == []
        expected = "Season 1950 is not available. Valid range: 1999-2024"
        assert resp.warning == expected

    def test_json_serialization(self) -> None:
        """Test JSON serialization matches expected schema."""
        data = [{"col1": "value1", "col2": 123}]
        meta = ResponseMetadata(
            row_count=1,
            total_available=45000,
            truncated=True,
            columns=["col1", "col2"],
        )
        resp = SuccessResponse(data=data, metadata=meta, warning=None)

        json_str = resp.model_dump_json()
        result = json.loads(json_str)

        # Verify matches design doc schema
        assert result["success"] is True
        assert result["data"] == [{"col1": "value1", "col2": 123}]
        assert result["metadata"]["row_count"] == 1
        assert result["metadata"]["total_available"] == 45000
        assert result["metadata"]["truncated"] is True
        assert result["metadata"]["columns"] == ["col1", "col2"]
        assert result["warning"] is None

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            SuccessResponse(extra="not allowed")  # type: ignore[call-arg]


class TestErrorResponse:
    """Tests for the ErrorResponse Pydantic model."""

    def test_create_with_error(self) -> None:
        """Test creating ErrorResponse with an error message."""
        error_msg = "Network error: Unable to reach data source"
        resp = ErrorResponse(error=error_msg)

        assert resp.success is False
        assert resp.data == []
        assert resp.metadata.row_count == 0
        assert resp.error == error_msg
        assert resp.warning is None

    def test_create_with_warning(self) -> None:
        """Test creating ErrorResponse with both error and warning."""
        resp = ErrorResponse(
            error="Failed to fetch data",
            warning="Retry may succeed",
        )

        assert resp.success is False
        assert resp.error == "Failed to fetch data"
        assert resp.warning == "Retry may succeed"

    def test_json_serialization(self) -> None:
        """Test JSON serialization matches expected schema."""
        resp = ErrorResponse(
            error="Network error: Unable to reach data source",
        )

        json_str = resp.model_dump_json()
        result = json.loads(json_str)

        # Verify matches design doc error schema
        assert result["success"] is False
        assert result["data"] == []
        assert result["metadata"]["row_count"] == 0
        assert result["error"] == "Network error: Unable to reach data source"
        assert result["warning"] is None

    def test_error_required(self) -> None:
        """Test that error field is required."""
        with pytest.raises(ValidationError):
            ErrorResponse()  # type: ignore[call-arg]


class TestFactoryFunctions:
    """Tests for the factory functions."""

    def test_create_success_response_minimal(self) -> None:
        """Test create_success_response with minimal args."""
        resp = create_success_response(data=[])

        assert resp.success is True
        assert resp.data == []
        assert resp.metadata.row_count == 0
        assert resp.metadata.truncated is False
        assert resp.warning is None

    def test_create_success_response_full(self) -> None:
        """Test create_success_response with all args."""
        data = [{"id": 1}, {"id": 2}]
        resp = create_success_response(
            data=data,
            total_available=1000,
            truncated=True,
            columns=["id"],
            warning="Data may be incomplete",
        )

        assert resp.success is True
        assert len(resp.data) == 2
        assert resp.metadata.row_count == 2
        assert resp.metadata.total_available == 1000
        assert resp.metadata.truncated is True
        assert resp.metadata.columns == ["id"]
        assert resp.warning == "Data may be incomplete"

    def test_create_success_response_row_count_auto(self) -> None:
        """Test that row_count is automatically set from data length."""
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        resp = create_success_response(data=data)

        assert resp.metadata.row_count == 3

    def test_create_error_response_minimal(self) -> None:
        """Test create_error_response with minimal args."""
        resp = create_error_response(error="Something went wrong")

        assert resp.success is False
        assert resp.data == []
        assert resp.metadata.row_count == 0
        assert resp.error == "Something went wrong"
        assert resp.warning is None

    def test_create_error_response_with_warning(self) -> None:
        """Test create_error_response with warning."""
        resp = create_error_response(
            error="Connection timeout",
            warning="Server may be overloaded",
        )

        assert resp.error == "Connection timeout"
        assert resp.warning == "Server may be overloaded"


class TestJsonInteroperability:
    """Tests for JSON serialization and deserialization."""

    def test_success_response_roundtrip(self) -> None:
        """Test SuccessResponse can be serialized and deserialized."""
        original = create_success_response(
            data=[{"player": "Mahomes", "td": 5}],
            total_available=500,
            truncated=True,
            columns=["player", "td"],
        )

        json_str = original.model_dump_json()
        restored = SuccessResponse.model_validate_json(json_str)

        assert restored.success == original.success
        assert restored.data == original.data
        assert restored.metadata.row_count == original.metadata.row_count
        assert restored.metadata.truncated == original.metadata.truncated

    def test_error_response_roundtrip(self) -> None:
        """Test ErrorResponse can be serialized and deserialized."""
        original = create_error_response(
            error="Test error",
            warning="Test warning",
        )

        json_str = original.model_dump_json()
        restored = ErrorResponse.model_validate_json(json_str)

        assert restored.success == original.success
        assert restored.error == original.error
        assert restored.warning == original.warning

    def test_model_dump_produces_json_compatible_dict(self) -> None:
        """Test that model_dump produces JSON-compatible dictionaries."""
        resp = create_success_response(
            data=[{"a": 1, "b": "text", "c": 3.14, "d": None}],
            columns=["a", "b", "c", "d"],
        )

        # model_dump should produce a dict that can be JSON serialized
        dump = resp.model_dump()
        json_str = json.dumps(dump)
        restored = json.loads(json_str)

        assert restored["success"] is True
        assert restored["data"][0]["a"] == 1
        assert restored["data"][0]["b"] == "text"
        assert restored["data"][0]["c"] == 3.14
        assert restored["data"][0]["d"] is None
