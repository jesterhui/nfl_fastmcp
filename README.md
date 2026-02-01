# Fast NFL MCP

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A Model Context Protocol (MCP) server that exposes comprehensive NFL data from the `nfl_data_py` library to AI assistants. Query play-by-play data, player statistics, rosters, contracts, and more through natural language.

## Features

- **Play-by-Play Data** - Detailed play outcomes with EPA (Expected Points Added) and WPA (Win Probability Added) metrics
- **Player Statistics** - Weekly and seasonal aggregated stats for passing, rushing, receiving, and fantasy
- **Roster Information** - Team rosters with player details, positions, physical attributes, and draft history
- **Draft Data** - Historical draft picks with player selection details and team information
- **Game Schedules** - Complete game schedules with matchups, dates, and scores
- **Contract Data** - Player salary information including guarantees and signing bonuses
- **Big Data Bowl** - NFL tracking data with per-frame player positions, speeds, and accelerations
- **Reference Data** - Cross-platform player ID mappings, team metadata, and game officials
- **Smart Pagination** - Browse large datasets with offset/limit support
- **Column Selection** - Request only the columns you need to reduce response size
- **Flexible Filtering** - Filter on any column with single values or lists

## Requirements

- Docker and Docker Compose (for production use)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (for development)
- Kaggle account (optional, for Big Data Bowl tracking data)

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/jesterhui/fast-nfl-mcp.git
cd fast-nfl-mcp

# Start the server
docker-compose up --build -d

# Verify it's running
docker-compose logs fast-nfl-mcp
```

The server runs on port 8000 with SSE transport and includes:
- Redis cache for improved performance
- Persistent volume for NFL data cache
- Automatic container restart

To stop the server:

```bash
docker-compose down
```

To reset cached data:

```bash
docker-compose down -v
```

### Option 2: Local Development with uv

```bash
# Clone the repository
git clone https://github.com/jesterhui/fast-nfl-mcp.git
cd fast-nfl-mcp

# Install dependencies and run
uv pip install -e ".[dev]"
uv run fast-nfl-mcp
```

This runs the server in stdio mode for local development.

## Claude Code Configuration

Add the MCP server to your Claude Code settings to enable NFL data queries.

### Project-Level Configuration (Recommended)

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

### Global Configuration

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

### Docker Configuration (SSE Mode)

If running via Docker Compose, use SSE transport.

**Important:** The Docker server must be running before starting Claude Code.

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

### Discovery Tools

| Tool | Description |
|------|-------------|
| `list_datasets` | List all available datasets with descriptions |
| `describe_dataset` | Get detailed schema information for a dataset |

### Data Query Tools

| Tool | Description |
|------|-------------|
| `get_play_by_play` | Play-by-play data with EPA, WPA, and detailed play outcomes (1999+) |
| `get_weekly_stats` | Weekly aggregated player statistics |
| `get_seasonal_stats` | Season-level player statistics |
| `get_rosters` | NFL team roster data with player information |
| `get_schedules` | Game schedules and results |
| `get_draft_picks` | Historical draft picks data |

### Reference Data Tools

| Tool | Description |
|------|-------------|
| `get_player_ids` | Cross-platform player ID mappings (GSIS, ESPN, Yahoo, Sleeper, PFF) |
| `lookup_player` | Search for players by name to find their GSIS ID |
| `get_team_descriptions` | NFL team metadata (names, divisions, conferences, colors) |
| `get_officials` | Game officials data with positions and game assignments |
| `get_contracts` | Player contract information |

### Big Data Bowl Tools

These tools access NFL tracking data from the Big Data Bowl 2026 competition. Requires Kaggle authentication (see [Kaggle Setup](#kaggle-setup-for-big-data-bowl)).

| Tool | Description |
|------|-------------|
| `get_bdb_games` | Game metadata from the tracking dataset |
| `get_bdb_plays` | Play-level data with descriptions and game situation |
| `get_bdb_players` | Player information (names, positions, physical attributes) |
| `get_bdb_tracking` | Per-frame tracking data with positions, speeds, and accelerations |

## Usage Examples

Once configured, ask your AI assistant questions like:

### Basic Queries

- "What were Patrick Mahomes' passing stats in 2024?"
- "Show me the Kansas City Chiefs roster"
- "Who are the top 10 rushers this season?"
- "What's Travis Kelce's contract worth?"

### Advanced Queries

- "Compare Lamar Jackson's rushing yards per game in 2023 vs 2024"
- "Find all 4th quarter comebacks by Josh Allen in 2024"
- "Show me first-round draft picks who played QB in 2024"
- "What's the average EPA on 3rd down passes for each team?"

### Big Data Bowl Queries

- "Show me tracking data for play 123 in game 2023090700"
- "Get player positions on a specific passing play"

## Kaggle Setup for Big Data Bowl

To access Big Data Bowl tracking data, you need Kaggle authentication:

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com/)

2. Generate an API token:
   - Go to Account Settings > API
   - Click "Create New Token"
   - Save the downloaded token

3. Set up credentials:
   ```bash
   mkdir -p ~/.kaggle
   # Copy your token to ~/.kaggle/access_token
   chmod 600 ~/.kaggle/access_token
   ```

4. Accept competition rules at:
   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/rules

For Docker users, the `docker-compose.yml` mounts `~/.kaggle` automatically.

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/jesterhui/fast-nfl-mcp.git
cd fast-nfl-mcp
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov

# Verbose output
uv run pytest -v
```

### Code Quality

The project uses pre-commit hooks for:
- **Black** - Code formatting
- **Ruff** - Linting and import sorting
- **Mypy** - Static type checking

Run manually:

```bash
uv run black .
uv run ruff check .
uv run mypy src/
```

## Project Structure

```
src/fast_nfl_mcp/
├── core/
│   ├── server.py          # FastMCP server initialization
│   └── models.py          # Pydantic request/response schemas
├── data/
│   ├── fetcher.py         # NFL data retrieval with validation
│   ├── schema.py          # Dataset schema management
│   └── kaggle.py          # Kaggle Big Data Bowl integration
├── tools/
│   ├── registry.py        # Tool registration with FastMCP
│   ├── play_by_play.py    # Play-by-play data tool
│   ├── player_stats.py    # Weekly and seasonal stats tools
│   ├── rosters.py         # Roster data tool
│   ├── schedules.py       # Game schedules tool
│   ├── draft.py           # Draft picks tool
│   ├── reference.py       # Reference data tools
│   ├── utilities.py       # Dataset discovery tools
│   └── bdb.py             # Big Data Bowl tools
└── utils/
    ├── constants.py       # Configuration constants
    ├── cache.py           # Redis caching utilities
    ├── validation.py      # Input validation
    └── helpers.py         # Utility functions
```

## Limitations

| Constraint | Limit |
|------------|-------|
| Rows per response | 100 max |
| Play-by-play seasons | 3 per request |
| Rosters/Weekly stats seasons | 5 per request |
| Seasonal stats seasons | 10 per request |
| Schedules seasons | 10 per request |
| Draft picks seasons | 20 per request |
| Data availability | 1999 onwards |
| Big Data Bowl tracking rows | 100 max (50 default) |

## License

MIT

## Author

Jacques Esterhuizen
