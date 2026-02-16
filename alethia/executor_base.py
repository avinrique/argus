"""Abstract executor interface for running code."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from alethia.models import ExecutionResult, TestCase


@runtime_checkable
class CodeExecutor(Protocol):
    def execute_code(
        self,
        code: str,
        stdin_input: str = "",
        timeout: int = 10,
        max_memory_mb: int = 256,
    ) -> ExecutionResult: ...

    def build_test_harness(
        self,
        solution_code: str,
        test_case: TestCase,
        function_signature: str | None,
    ) -> tuple[str, str]: ...

    def outputs_match(self, expected: str, actual: str) -> bool: ...
