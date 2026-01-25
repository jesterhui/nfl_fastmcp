# Implementation Plan: JAC-25 - Basic FastMCP Server Skeleton

## Summary

Create a minimal FastMCP server that:
1. Initializes correctly with the name "fast-nfl-mcp"
2. Starts without errors via stdio transport
3. Includes a placeholder hook for schema preloading
4. Can be run using `uv run fast-nfl-mcp`

## Technical Approach

### Files to Create/Modify

1. **Create `src/fast_nfl_mcp/server.py`**
   - Initialize FastMCP application with name "fast-nfl-mcp"
   - Add a lifespan context manager for future schema preloading
   - Define `main()` entry point function
   - Use stdio transport for MCP communication

2. **Verify `pyproject.toml`**
   - Already has entry point: `fast-nfl-mcp = "fast_nfl_mcp.server:main"`
   - No changes needed

### Implementation Details

The server will use FastMCP's recommended patterns:
- Use `FastMCP("fast-nfl-mcp")` for initialization
- Use `@asynccontextmanager` for lifespan management (placeholder for schema preloading)
- Call `mcp.run()` in main() to start the stdio server

### Testing Strategy

1. Create `tests/test_server.py` with:
   - Test that FastMCP app initializes correctly
   - Test that the app has the correct name
   - Test that main() function exists and is callable

2. Manual verification:
   - Run `uv run fast-nfl-mcp` and verify no errors on startup

## Estimated Scope

Small - approximately 30-50 lines of implementation code plus tests.

## Risks/Considerations

- Need to verify FastMCP version 2.0+ API for lifespan management
- Keep server.py minimal - tool registrations will be added in future tasks
