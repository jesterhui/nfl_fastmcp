"""End-to-end tests for Docker deployment.

This module contains integration tests that verify the complete Docker
deployment works correctly, including:
- Docker build
- Container startup with schema preloading
- MCP SSE endpoint accessibility
- MCP tool execution via JSON-RPC
- Non-root user execution
- Cache volume persistence across restarts

These tests require Docker to be running and should be run separately
from unit tests using: pytest tests/test_docker_e2e.py -v -m e2e

Note: These tests make real network calls and may take several minutes.
"""

import json
import select
import subprocess
import time
import uuid
from collections.abc import Generator
from typing import Any

import pytest
import requests

# Mark all tests in this module as e2e (end-to-end)
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def docker_compose_up() -> Generator[None, None, None]:
    """Start docker compose before tests and clean up after.

    This fixture:
    1. Builds the Docker image
    2. Starts the container
    3. Waits for the server to be ready
    4. Yields control to tests
    5. Stops and removes containers after tests complete
    """
    # Build the image
    build_result = subprocess.run(
        ["docker", "compose", "build"],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minutes for build
    )
    if build_result.returncode != 0:
        pytest.fail(f"Docker build failed:\n{build_result.stderr}")

    # Start containers in detached mode
    up_result = subprocess.run(
        ["docker", "compose", "up", "-d"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if up_result.returncode != 0:
        pytest.fail(f"Docker compose up failed:\n{up_result.stderr}")

    # Wait for server to be ready (up to 90 seconds for schema preloading)
    max_wait = 90
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            # Use stream=True for SSE endpoint - it returns immediately with 200
            # but streams data, so we need to handle it as a stream
            response = requests.get("http://localhost:8000/sse", timeout=5, stream=True)
            # SSE endpoint returns 200 with streaming response
            if response.status_code == 200:
                server_ready = True
                response.close()  # Close the streaming connection
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)

    if not server_ready:
        # Get logs for debugging
        logs = subprocess.run(
            ["docker", "compose", "logs", "--tail=50"],
            capture_output=True,
            text=True,
        )
        pytest.fail(
            f"Server did not become ready within {max_wait}s.\nLogs:\n{logs.stdout}"
        )

    yield

    # Cleanup: stop and remove containers
    subprocess.run(
        ["docker", "compose", "down", "-v"],
        capture_output=True,
        timeout=60,
    )


def read_json_line(stdout: Any, timeout: float = 60.0) -> dict[str, Any]:
    """Read lines until we get a valid JSON response.

    The server may print log messages to stdout which we need to skip.
    Uses select() for timeout-based reading to prevent indefinite hangs.

    Args:
        stdout: The stdout stream to read from
        timeout: Maximum seconds to wait for a response (default 60)

    Returns:
        The parsed JSON response

    Raises:
        RuntimeError: If no valid JSON is found within timeout or max attempts
        TimeoutError: If reading times out
    """
    max_attempts = 50
    start_time = time.time()

    for _ in range(max_attempts):
        # Check if we've exceeded the timeout
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining <= 0:
            raise TimeoutError(f"Timed out after {timeout}s waiting for JSON response")

        # Use select to wait for data with timeout (Unix-only, but Docker tests run on Unix)
        fd = stdout.fileno()
        ready, _, _ = select.select([fd], [], [], min(remaining, 5.0))

        if not ready:
            # No data available yet, continue waiting
            continue

        line = stdout.readline()
        if not line:
            raise RuntimeError("EOF reached without valid JSON response")
        line = line.strip()
        if not line:
            continue
        try:
            result: dict[str, Any] = json.loads(line)
            return result
        except json.JSONDecodeError:
            # Skip non-JSON lines (log messages, etc.)
            continue

    raise RuntimeError("No valid JSON response found after max attempts")


def create_mcp_session() -> tuple[subprocess.Popen[str], str]:
    """Create an MCP session by starting the server and doing handshake.

    Returns:
        Tuple of (process, session_id) for the running MCP server.
    """
    # Start the MCP server in stdio mode inside Docker
    # Redirect stderr to devnull to prevent pipe buffer backpressure
    # which can cause the subprocess to hang if stderr buffer fills up
    proc = subprocess.Popen(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "fast-nfl-mcp",
            "fast-nfl-mcp",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # Prevent backpressure from unread stderr
        text=True,
    )

    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "1.0.0"},
        },
    }
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(init_request) + "\n")
    proc.stdin.flush()

    # Read initialize response (skip non-JSON lines)
    assert proc.stdout is not None
    init_response = read_json_line(proc.stdout)

    # Send initialized notification
    initialized_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }
    proc.stdin.write(json.dumps(initialized_notif) + "\n")
    proc.stdin.flush()

    session_id = init_response.get("id", "unknown")
    return proc, str(session_id)


def send_mcp_request(
    proc: subprocess.Popen[str], method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Send a JSON-RPC request to an MCP session.

    Args:
        proc: The subprocess running the MCP server
        method: The JSON-RPC method name
        params: Optional parameters for the method

    Returns:
        The JSON-RPC response as a dictionary
    """
    request_id = str(uuid.uuid4())
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params:
        payload["params"] = params

    assert proc.stdin is not None
    assert proc.stdout is not None

    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()

    # Read response (skip non-JSON lines)
    return read_json_line(proc.stdout)


def close_mcp_session(proc: subprocess.Popen[str]) -> None:
    """Close an MCP session."""
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def mcp_session(
    docker_compose_up: None,
) -> Generator[subprocess.Popen[str], None, None]:
    """Create an MCP session for testing tools.

    This fixture creates an MCP session with proper handshake,
    yields the process for tests to use, and cleans up after.
    """
    proc, _ = create_mcp_session()
    yield proc
    close_mcp_session(proc)


class TestDockerBuild:
    """Tests for Docker image build."""

    def test_docker_build_succeeds(self) -> None:
        """Test that docker compose build completes successfully."""
        result = subprocess.run(
            ["docker", "compose", "build"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"Build failed:\n{result.stderr}"

    def test_image_exists_after_build(self) -> None:
        """Test that the image is created after build."""
        # Build first
        subprocess.run(["docker", "compose", "build"], capture_output=True, timeout=600)

        # Check image exists
        result = subprocess.run(
            ["docker", "images", "--filter", "reference=*fast-nfl*", "-q"],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() != "", "Docker image was not created"


class TestContainerStartup:
    """Tests for container startup behavior."""

    def test_server_starts_within_timeout(self, docker_compose_up: None) -> None:
        """Test that the server starts and is accessible within 90 seconds.

        This implicitly tests schema preloading completes successfully.
        """
        # The fixture handles startup - if we get here, it succeeded
        response = requests.get("http://localhost:8000/sse", timeout=10, stream=True)
        assert response.status_code == 200
        response.close()

    def test_sse_endpoint_returns_session_info(self, docker_compose_up: None) -> None:
        """Test that SSE endpoint returns session information."""
        # SSE returns event stream, check we get data
        response = requests.get("http://localhost:8000/sse", timeout=10, stream=True)
        assert response.status_code == 200

        # Read first chunk of streaming response
        for line in response.iter_lines(decode_unicode=True):
            if line:
                # Should contain 'event:' or 'data:' for SSE
                assert "event" in line.lower() or "data" in line.lower()
                break


class TestContainerSecurity:
    """Tests for container security configuration."""

    def test_runs_as_non_root_user(self, docker_compose_up: None) -> None:
        """Test that the container runs as non-root user."""
        result = subprocess.run(
            ["docker", "compose", "exec", "fast-nfl-mcp", "whoami"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "appuser"

    def test_user_has_correct_uid(self, docker_compose_up: None) -> None:
        """Test that the user has expected UID 1000."""
        result = subprocess.run(
            ["docker", "compose", "exec", "fast-nfl-mcp", "id", "-u"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "1000"


class TestCacheVolume:
    """Tests for cache volume functionality."""

    def test_cache_directory_exists(self, docker_compose_up: None) -> None:
        """Test that the cache directory exists in the container."""
        result = subprocess.run(
            ["docker", "compose", "exec", "fast-nfl-mcp", "test", "-d", "/app/cache"],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, "Cache directory does not exist"

    def test_cache_directory_is_writable(self, docker_compose_up: None) -> None:
        """Test that the cache directory is writable by appuser."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "fast-nfl-mcp",
                "touch",
                "/app/cache/test_write",
            ],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, "Cache directory is not writable"

        # Clean up test file
        subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "fast-nfl-mcp",
                "rm",
                "/app/cache/test_write",
            ],
            capture_output=True,
            timeout=30,
        )


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_nfl_cache_dir_is_set(self, docker_compose_up: None) -> None:
        """Test that NFL_CACHE_DIR environment variable is set."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "fast-nfl-mcp",
                "printenv",
                "NFL_CACHE_DIR",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "/app/cache"


class TestServerHealth:
    """Tests for server health and responsiveness."""

    def test_server_accepts_connections(self, docker_compose_up: None) -> None:
        """Test that the server accepts HTTP connections."""
        response = requests.get("http://localhost:8000/sse", timeout=10, stream=True)
        assert response.status_code == 200
        response.close()

    def test_multiple_concurrent_connections(self, docker_compose_up: None) -> None:
        """Test that the server can handle multiple connections."""
        import concurrent.futures

        def make_request() -> int:
            response = requests.get(
                "http://localhost:8000/sse", timeout=10, stream=True
            )
            status: int = response.status_code
            response.close()
            return status

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(status == 200 for status in results)


class TestContainerLogs:
    """Tests for container logging."""

    def test_logs_show_server_startup(self, docker_compose_up: None) -> None:
        """Test that container logs show successful server startup."""
        result = subprocess.run(
            ["docker", "compose", "logs", "--tail=100"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Check for FastMCP startup indicators
        assert "FastMCP" in result.stdout or "fast-nfl-mcp" in result.stdout

    def test_logs_show_transport_type(self, docker_compose_up: None) -> None:
        """Test that logs show SSE transport mode."""
        result = subprocess.run(
            ["docker", "compose", "logs", "--tail=100"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should indicate SSE transport
        assert "sse" in result.stdout.lower() or "8000" in result.stdout


class TestMCPToolExecution:
    """Tests for MCP tool execution via stdio transport.

    These tests verify that all MCP tools respond correctly when called
    via the JSON-RPC protocol. We use stdio mode for reliable synchronous
    testing within the Docker container.
    """

    def test_list_tools_returns_all_tools(
        self, mcp_session: subprocess.Popen[str]
    ) -> None:
        """Test that tools/list returns all available tools."""
        response = send_mcp_request(mcp_session, "tools/list")

        assert "result" in response, f"Expected result in response: {response}"
        tools = response["result"].get("tools", [])

        # Should have at least 11 tools (the ones we know about)
        assert len(tools) >= 11, f"Expected at least 11 tools, got {len(tools)}"

        # Check for specific expected tools
        tool_names = {tool["name"] for tool in tools}
        expected_tools = {
            "list_datasets",
            "describe_dataset",
            "get_play_by_play",
            "get_weekly_stats",
            "get_seasonal_stats",
            "get_rosters",
            "get_player_ids",
            "lookup_player",
            "get_team_descriptions",
            "get_officials",
            "get_contracts",
        }
        assert expected_tools.issubset(
            tool_names
        ), f"Missing tools: {expected_tools - tool_names}"

    def test_list_datasets_tool(self, mcp_session: subprocess.Popen[str]) -> None:
        """Test that list_datasets tool returns dataset information."""
        response = send_mcp_request(
            mcp_session, "tools/call", {"name": "list_datasets", "arguments": {}}
        )

        assert "result" in response, f"Expected result in response: {response}"
        # The tool returns content with text containing JSON
        content = response["result"].get("content", [])
        assert len(content) > 0, "Expected content in response"

        # Parse the response - it should contain dataset info
        text_content = content[0].get("text", "")
        result_data = json.loads(text_content)
        assert result_data.get("success") is True
        assert "data" in result_data

    def test_describe_dataset_tool(self, mcp_session: subprocess.Popen[str]) -> None:
        """Test that describe_dataset tool returns schema information."""
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {"name": "describe_dataset", "arguments": {"dataset": "play_by_play"}},
        )

        assert "result" in response, f"Expected result in response: {response}"
        content = response["result"].get("content", [])
        assert len(content) > 0, "Expected content in response"

        text_content = content[0].get("text", "")
        result_data = json.loads(text_content)
        assert result_data.get("success") is True
        assert "data" in result_data

    def test_get_team_descriptions_tool(
        self, mcp_session: subprocess.Popen[str]
    ) -> None:
        """Test that get_team_descriptions returns team data."""
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {"name": "get_team_descriptions", "arguments": {}},
        )

        assert "result" in response, f"Expected result in response: {response}"
        content = response["result"].get("content", [])
        assert len(content) > 0, "Expected content in response"

        text_content = content[0].get("text", "")
        result_data = json.loads(text_content)
        assert result_data.get("success") is True
        assert "data" in result_data
        # Should have data for NFL teams
        assert len(result_data["data"]) > 0

    def test_get_player_ids_tool(self, mcp_session: subprocess.Popen[str]) -> None:
        """Test that get_player_ids returns player ID mappings."""
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {"name": "get_player_ids", "arguments": {"limit": 10}},
        )

        assert "result" in response, f"Expected result in response: {response}"
        content = response["result"].get("content", [])
        assert len(content) > 0, "Expected content in response"

        text_content = content[0].get("text", "")
        result_data = json.loads(text_content)
        assert result_data.get("success") is True
        assert "data" in result_data

    def test_lookup_player_tool(self, mcp_session: subprocess.Popen[str]) -> None:
        """Test that lookup_player can find players by name."""
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {"name": "lookup_player", "arguments": {"name": "Mahomes"}},
        )

        assert "result" in response, f"Expected result in response: {response}"
        content = response["result"].get("content", [])
        assert len(content) > 0, "Expected content in response"

        text_content = content[0].get("text", "")
        result_data = json.loads(text_content)
        assert result_data.get("success") is True


class TestCachePersistence:
    """Tests for cache persistence across container restarts.

    Note: This test class uses the docker_compose_up fixture and tests
    cache persistence using restart (not down -v).
    """

    def test_cache_persists_across_restart(self, docker_compose_up: None) -> None:
        """Test that cache data persists when container is restarted.

        This test:
        1. Uses the running container from docker_compose_up fixture
        2. Writes a test file to the cache
        3. Restarts the container (without removing volumes)
        4. Verifies the file still exists
        """
        # Write a test file to the cache
        test_content = f"test-{uuid.uuid4()}"
        subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "fast-nfl-mcp",
                "sh",
                "-c",
                f'echo "{test_content}" > /app/cache/persistence_test',
            ],
            capture_output=True,
            timeout=30,
        )

        # Restart the container (not down -v which removes volumes)
        subprocess.run(
            ["docker", "compose", "restart"],
            capture_output=True,
            timeout=60,
        )

        # Wait for server to be ready again
        max_wait = 90
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    "http://localhost:8000/sse", timeout=5, stream=True
                )
                if response.status_code == 200:
                    response.close()
                    break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(2)

        # Verify the file still exists with correct content
        result = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "fast-nfl-mcp",
                "cat",
                "/app/cache/persistence_test",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, "Cache file did not persist"
        assert (
            test_content in result.stdout
        ), f"Content mismatch: expected {test_content}, got {result.stdout}"

        # Clean up test file (don't remove the container - fixture handles that)
        subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "fast-nfl-mcp",
                "rm",
                "-f",
                "/app/cache/persistence_test",
            ],
            capture_output=True,
            timeout=30,
        )


class TestErrorScenarios:
    """Tests for error handling scenarios."""

    def test_invalid_tool_name_returns_error(
        self, mcp_session: subprocess.Popen[str]
    ) -> None:
        """Test that calling a non-existent tool returns an error."""
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {"name": "nonexistent_tool", "arguments": {}},
        )

        # Should have an error response
        assert "error" in response or (
            "result" in response and response["result"].get("isError", False)
        ), f"Expected error for invalid tool: {response}"

    def test_invalid_dataset_name_returns_error(
        self, mcp_session: subprocess.Popen[str]
    ) -> None:
        """Test that describe_dataset with invalid name returns error."""
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {"name": "describe_dataset", "arguments": {"dataset": "invalid_dataset"}},
        )

        assert "result" in response, f"Expected result in response: {response}"
        content = response["result"].get("content", [])
        assert len(content) > 0

        text_content = content[0].get("text", "")
        result_data = json.loads(text_content)
        # Should indicate failure for invalid dataset
        assert result_data.get("success") is False

    def test_missing_required_parameter(
        self, mcp_session: subprocess.Popen[str]
    ) -> None:
        """Test that missing required parameters return error."""
        # get_play_by_play requires 'seasons' and 'columns' parameters
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {"name": "get_play_by_play", "arguments": {}},
        )

        # Should have an error for missing parameters
        assert "error" in response or (
            "result" in response
            and (
                response["result"].get("isError", False)
                or any(
                    "error" in str(c.get("text", "")).lower()
                    for c in response["result"].get("content", [])
                )
            )
        ), f"Expected error for missing parameters: {response}"

    def test_invalid_season_format(self, mcp_session: subprocess.Popen[str]) -> None:
        """Test that invalid season format returns appropriate error."""
        response = send_mcp_request(
            mcp_session,
            "tools/call",
            {
                "name": "get_play_by_play",
                "arguments": {
                    "seasons": ["invalid"],  # Should be integers
                    "columns": ["play_id"],
                },
            },
        )

        # Should have an error for invalid type
        has_error = "error" in response or (
            "result" in response
            and (
                response["result"].get("isError", False)
                or any(
                    "error" in str(c.get("text", "")).lower()
                    for c in response["result"].get("content", [])
                )
            )
        )
        assert has_error, f"Expected error for invalid season format: {response}"
