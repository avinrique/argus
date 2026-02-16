"""Judge0 REST API executor for sandboxed remote code execution."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from alethia.executor import _build_class_harness, _build_function_harness, outputs_match
from alethia.models import ExecutionResult, TestCase

# Judge0 status codes
_STATUS_IN_QUEUE = 1
_STATUS_PROCESSING = 2
_STATUS_ACCEPTED = 3  # ran successfully (exit code 0)
_STATUS_WRONG_ANSWER = 4
_STATUS_TLE = 5
_STATUS_COMPILATION_ERROR = 6
_STATUS_RUNTIME_ERROR_SIGSEGV = 7
_STATUS_RUNTIME_ERROR_SIGXFSZ = 8
_STATUS_RUNTIME_ERROR_SIGFPE = 9
_STATUS_RUNTIME_ERROR_SIGABRT = 10
_STATUS_RUNTIME_ERROR_NZEC = 11
_STATUS_RUNTIME_ERROR_OTHER = 12
_STATUS_INTERNAL_ERROR = 13
_STATUS_EXEC_FORMAT_ERROR = 14


@dataclass
class Judge0Config:
    base_url: str = "http://localhost:2358"
    api_key: str = ""
    language_id: int = 71  # Python 3
    poll_interval: float = 0.5
    max_poll_attempts: int = 60


class Judge0Executor:
    """Executes code via the Judge0 REST API."""

    def __init__(self, config: Judge0Config | None = None) -> None:
        self._config = config or Judge0Config()

    def execute_code(
        self,
        code: str,
        stdin_input: str = "",
        timeout: int = 10,
        max_memory_mb: int = 256,
    ) -> ExecutionResult:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["X-Auth-Token"] = self._config.api_key

        payload: dict = {
            "source_code": code,
            "language_id": self._config.language_id,
            "stdin": stdin_input,
            "cpu_time_limit": timeout,
            "memory_limit": max_memory_mb * 1024,  # Judge0 expects KB
        }

        base = self._config.base_url.rstrip("/")

        try:
            # Try synchronous submission (wait=true)
            resp = httpx.post(
                f"{base}/submissions?base64_encoded=false&wait=true",
                json=payload,
                headers=headers,
                timeout=timeout + 30,  # extra margin for network
            )
            resp.raise_for_status()
            data = resp.json()

            # If we got a token but no status, the server didn't wait — poll
            if "status" not in data or data.get("status", {}).get("id") in (
                _STATUS_IN_QUEUE,
                _STATUS_PROCESSING,
            ):
                token = data.get("token", "")
                if token:
                    data = self._poll(token, headers, base)

            return self._parse_response(data)

        except httpx.TimeoutException:
            return ExecutionResult(
                stdout="", stderr="Judge0 request timed out", exit_code=-1, timed_out=True
            )
        except Exception as e:
            return ExecutionResult(stdout="", stderr=str(e), exit_code=-1)

    def _poll(self, token: str, headers: dict[str, str], base: str) -> dict:
        for _ in range(self._config.max_poll_attempts):
            time.sleep(self._config.poll_interval)
            resp = httpx.get(
                f"{base}/submissions/{token}?base64_encoded=false",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            status_id = data.get("status", {}).get("id", 0)
            if status_id not in (_STATUS_IN_QUEUE, _STATUS_PROCESSING):
                return data
        return {"status": {"id": _STATUS_TLE}, "stdout": "", "stderr": "Poll timeout"}

    def _parse_response(self, data: dict) -> ExecutionResult:
        status_id = data.get("status", {}).get("id", 0)
        stdout = data.get("stdout") or ""
        stderr = data.get("stderr") or ""
        compile_output = data.get("compile_output") or ""

        if status_id == _STATUS_ACCEPTED:
            return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=0)

        if status_id == _STATUS_TLE:
            return ExecutionResult(
                stdout=stdout, stderr=stderr or "Time limit exceeded", exit_code=-1, timed_out=True
            )

        if status_id == _STATUS_COMPILATION_ERROR:
            return ExecutionResult(
                stdout="", stderr=compile_output or "Compilation error", exit_code=-1
            )

        # Runtime errors (7-12) and other failures
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr or data.get("status", {}).get("description", "Unknown error"),
            exit_code=1,
        )

    def build_test_harness(
        self,
        solution_code: str,
        test_case: TestCase,
        function_signature: str | None,
    ) -> tuple[str, str]:
        if function_signature:
            if function_signature.strip().startswith("class "):
                return _build_class_harness(solution_code, test_case), ""
            return _build_function_harness(solution_code, test_case, function_signature), ""
        return solution_code, test_case.input

    def outputs_match(self, expected: str, actual: str) -> bool:
        return outputs_match(expected, actual)
