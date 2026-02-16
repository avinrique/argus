"""Tests for the Judge0 executor (mocked, no real server needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alethia.executor_judge0 import Judge0Config, Judge0Executor
from alethia.models import TestCase


def _make_response(status_id: int, stdout: str = "", stderr: str = "", compile_output: str = ""):
    """Build a mock Judge0 API response."""
    resp = MagicMock()
    resp.json.return_value = {
        "status": {"id": status_id, "description": ""},
        "stdout": stdout,
        "stderr": stderr,
        "compile_output": compile_output,
        "token": "abc123",
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestJudge0Success:
    @patch("alethia.executor_judge0.httpx.post")
    def test_successful_execution(self, mock_post):
        mock_post.return_value = _make_response(3, stdout="hello\n")
        executor = Judge0Executor(Judge0Config(base_url="http://fake:2358"))
        result = executor.execute_code("print('hello')")
        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        assert not result.timed_out

    @patch("alethia.executor_judge0.httpx.post")
    def test_payload_fields(self, mock_post):
        mock_post.return_value = _make_response(3)
        executor = Judge0Executor(Judge0Config(base_url="http://fake:2358", api_key="secret"))
        executor.execute_code("print(1)", stdin_input="42\n", timeout=5, max_memory_mb=128)
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["source_code"] == "print(1)"
        assert payload["stdin"] == "42\n"
        assert payload["cpu_time_limit"] == 5
        assert payload["memory_limit"] == 128 * 1024
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["X-Auth-Token"] == "secret"


class TestJudge0Errors:
    @patch("alethia.executor_judge0.httpx.post")
    def test_time_limit_exceeded(self, mock_post):
        mock_post.return_value = _make_response(5)
        executor = Judge0Executor(Judge0Config(base_url="http://fake:2358"))
        result = executor.execute_code("while True: pass")
        assert result.timed_out
        assert result.exit_code == -1

    @patch("alethia.executor_judge0.httpx.post")
    def test_compilation_error(self, mock_post):
        mock_post.return_value = _make_response(6, compile_output="SyntaxError")
        executor = Judge0Executor(Judge0Config(base_url="http://fake:2358"))
        result = executor.execute_code("def :")
        assert result.exit_code == -1
        assert "SyntaxError" in result.stderr

    @patch("alethia.executor_judge0.httpx.post")
    def test_runtime_error_nzec(self, mock_post):
        mock_post.return_value = _make_response(11, stderr="ValueError: boom")
        executor = Judge0Executor(Judge0Config(base_url="http://fake:2358"))
        result = executor.execute_code("raise ValueError('boom')")
        assert result.exit_code == 1
        assert "ValueError" in result.stderr


class TestJudge0PollFallback:
    @patch("alethia.executor_judge0.httpx.get")
    @patch("alethia.executor_judge0.httpx.post")
    def test_poll_when_not_ready(self, mock_post, mock_get):
        # Initial response: still processing
        initial = MagicMock()
        initial.json.return_value = {
            "token": "tok123",
            "status": {"id": 2, "description": "Processing"},
        }
        initial.raise_for_status = MagicMock()
        mock_post.return_value = initial

        # Poll response: done
        poll_resp = _make_response(3, stdout="done\n")
        mock_get.return_value = poll_resp

        executor = Judge0Executor(
            Judge0Config(base_url="http://fake:2358", poll_interval=0.01, max_poll_attempts=3)
        )
        result = executor.execute_code("print('done')")
        assert result.exit_code == 0
        assert result.stdout == "done\n"
        mock_get.assert_called_once()


class TestJudge0Harness:
    def test_build_test_harness_function(self):
        executor = Judge0Executor()
        tc = TestCase(input="3\n4", expected_output="7")
        code, stdin = executor.build_test_harness("def add(a, b):\n    return a + b", tc, "def add(a, b)")
        assert "add" in code
        assert stdin == ""

    def test_build_test_harness_stdin(self):
        executor = Judge0Executor()
        tc = TestCase(input="5", expected_output="10")
        code, stdin = executor.build_test_harness("n = int(input()); print(n*2)", tc, None)
        assert stdin == "5"

    def test_outputs_match_delegates(self):
        executor = Judge0Executor()
        assert executor.outputs_match("[0, 1]", "[1, 0]")
        assert not executor.outputs_match("42", "43")
