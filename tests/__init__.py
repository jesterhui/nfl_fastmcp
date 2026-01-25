"""Tests for fast-nfl-mcp package.

Test modules:
- conftest.py: Shared fixtures for all test modules
- test_models.py: Tests for response models (ColumnSchema, DatasetSchema, etc.)
- test_schema_manager.py: Tests for SchemaManager and schema preloading
- test_server.py: Tests for MCP server initialization and lifespan

Run tests with:
    pytest tests/ -v --cov=fast_nfl_mcp

All tests use mocked nfl_data_py calls to avoid network dependencies.
"""
