# Fast NFL MCP

A Model Context Protocol (MCP) server that exposes comprehensive NFL data from the `nfl_data_py` library to AI assistants. Query play-by-play data, player statistics, rosters, contracts, and more through natural language.

## Features

- **Play-by-Play Data** - Detailed play outcomes with EPA (Expected Points Added) and WPA (Win Probability Added) metrics
- **Player Statistics** - Weekly and seasonal aggregated stats for passing, rushing, receiving, and fantasy
- **Roster Information** - Team rosters with player details, positions, physical attributes, and draft history
- **Contract Data** - Player salary information including guarantees and signing bonuses
- **Reference Data** - Cross-platform player ID mappings, team metadata, and game officials
- **Smart Pagination** - Browse large datasets with offset/limit support
- **Column Selection** - Request only the columns you need to reduce response size
- **Flexible Filtering** - Filter on any column with single values or lists

## Requirements

- Docker and Docker Compose (for production use)
- Python 3.11+ and [UV](https://docs.astral.sh/uv/) (for development)

## Installation

### For Users

```bash
git clone https://github.com/jesterhui/fast-nfl-mcp.git
cd fast-nfl-mcp
docker-compose up --build -d
```

To stop the server:

```bash
docker-compose down
```

The server runs on port 8000 with a persistent cache volume for NFL data. To reset cached data, remove the volume and rebuild (e.g. `docker-compose down -v`).

### For Development

```bash
git clone https://github.com/jesterhui/fast-nfl-mcp.git
cd fast-nfl-mcp
uv pip install -e ".[dev]"
uv run fast-nfl-mcp
```

This runs the server in stdio mode for local development and allows rapid iteration without container rebuilds.

## Configuration

### Claude Code

Add the server to your Claude Code settings:

**Option 1: Project-level configuration (recommended)**

Create or edit `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "fast-nfl-mcp": {
      "command": "uv",
      "args": ["run", "fast-nfl-mcp"],
      "cwd": "/path/to/fast-nfl-mcp"
    }
  }
}
```

**Option 2: Global configuration**

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "fast-nfl-mcp": {
      "command": "uv",
      "args": ["run", "fast-nfl-mcp"],
      "cwd": "/path/to/fast-nfl-mcp"
    }
  }
}
```

### Docker (SSE mode)

If running via Docker Compose, use SSE transport:

```json
{
  "mcpServers": {
    "fast-nfl-mcp": {
      "type": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### Verifying the Configuration

After adding the configuration, restart Claude Code. You should see "fast-nfl-mcp" in the available MCP servers. Test by asking:

- "List available NFL datasets"
- "Look up Patrick Mahomes"
- "Get team descriptions"

See `examples/` for ready-to-use configuration files.

## Available Tools

### Data Query Tools

| Tool | Description |
|------|-------------|
| `get_play_by_play` | Play-by-play data with EPA, WPA, and detailed play outcomes (1999+) |
| `get_weekly_stats` | Weekly aggregated player statistics |
| `get_seasonal_stats` | Season-level player statistics |
| `get_rosters` | NFL team roster data with player information |

### Reference Data Tools

| Tool | Description |
|------|-------------|
| `get_player_ids` | Cross-platform player ID mappings (GSIS, ESPN, Yahoo, Sleeper, PFF) |
| `lookup_player` | Search for players by name to find their GSIS ID |
| `get_team_descriptions` | NFL team metadata (names, divisions, conferences, colors) |
| `get_officials` | Game officials data with positions and game assignments |
| `get_contracts` | Player contract information |

### Discovery Tools

| Tool | Description |
|------|-------------|
| `list_datasets` | List all available datasets with descriptions |
| `describe_dataset` | Get detailed schema information for a dataset |

## Usage Examples

Once configured, ask your AI assistant questions like:

- "What were Patrick Mahomes' passing stats in 2024?"
- "Show me the play-by-play for Chiefs vs Bills in week 1"
- "Who are the top 10 rushers this season?"
- "What's Travis Kelce's contract worth?"
- "List all QBs on the 49ers roster"

## Development

After following the "For Development" installation steps, install pre-commit hooks:

```bash
pre-commit install
```

### Running Tests

```bash
pytest                    # Run all tests
pytest --cov             # With coverage report
pytest -v                # Verbose output
```

### Code Quality

The project uses pre-commit hooks for:
- **Black** - Code formatting
- **Ruff** - Linting and import sorting
- **Mypy** - Static type checking
- **Flake8** - Additional linting

## Project Structure

```
src/fast_nfl_mcp/
├── server.py          # FastMCP server initialization and tool registration
├── schema_manager.py  # Dataset schema preloading and caching
├── data_fetcher.py    # NFL data retrieval with validation and pagination
├── models.py          # Pydantic models for request/response schemas
├── constants.py       # Configuration constants
└── tools/
    ├── play_by_play.py   # Play-by-play data tool
    ├── player_stats.py   # Weekly and seasonal stats tools
    ├── rosters.py        # Roster data tool
    ├── reference.py      # Reference data tools
    └── utilities.py      # Dataset discovery tools
```

## Limitations

- **Row Limits** - Maximum 100 rows per response (use pagination for larger datasets)
- **Season Limits** - Play-by-play: 3 seasons, Rosters/Weekly: 5 seasons, Seasonal: 10 seasons per request
- **Data Availability** - Most datasets available from 1999 onward

## License

MIT

## Author

Jacques Esterhuizen
