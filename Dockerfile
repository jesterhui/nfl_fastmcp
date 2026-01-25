# Fast NFL MCP Server Dockerfile
# Containerized deployment with non-root user and cache volume support

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    NFL_CACHE_DIR=/app/cache

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Create non-root user for security
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set up application directory
WORKDIR /app

# Create cache directory with proper permissions
RUN mkdir -p /app/cache && chown -R appuser:appgroup /app

# Copy project files for installation
COPY --chown=appuser:appgroup pyproject.toml uv.lock ./
COPY --chown=appuser:appgroup src/ ./src/

# Install the package (non-editable for production)
RUN uv pip install --no-cache .

# Switch to non-root user
USER appuser

# Set the entrypoint for the MCP server
ENTRYPOINT ["python", "-m", "fast_nfl_mcp.server"]
