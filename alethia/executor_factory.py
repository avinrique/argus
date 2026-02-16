"""Factory for creating code executors based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alethia.executor import LocalExecutor
from alethia.executor_base import CodeExecutor

if TYPE_CHECKING:
    from alethia.config import Config


def create_executor(config: Config) -> CodeExecutor:
    """Create an executor based on config.executor_type."""
    if config.executor_type == "judge0":
        from alethia.executor_judge0 import Judge0Config, Judge0Executor

        return Judge0Executor(
            Judge0Config(
                base_url=config.judge0_url,
                api_key=config.judge0_api_key,
            )
        )
    return LocalExecutor()
