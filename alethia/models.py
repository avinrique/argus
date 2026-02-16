"""Data models for Alethia."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Verdict(enum.Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass
class TestCase:
    input: str
    expected_output: str
    description: str = ""
    is_generated: bool = False


@dataclass
class Problem:
    title: str
    description: str
    test_cases: list[TestCase]
    function_signature: str | None = None  # e.g. "def two_sum(nums, target)"
    constraints: str = ""
    code_template: str = ""  # Platform boilerplate, e.g. "class Solution:\n    def ..."
    images: list[str] = field(default_factory=list)  # Absolute paths to problem images


@dataclass
class Solution:
    code: str
    attempt: int
    raw_response: str = ""
    plan: str = ""
    strategy: str = ""  # name of generation strategy used


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@dataclass
class TestCaseResult:
    test_case: TestCase
    execution: ExecutionResult
    passed: bool
    actual_output: str = ""


@dataclass
class VerificationResult:
    verdict: Verdict
    test_results: list[TestCaseResult] = field(default_factory=list)
    llm_reasoning: str = ""
    summary: str = ""
    score: float = 0.0


@dataclass
class RevisionFeedback:
    root_cause: str
    suggested_fix: str
    focus_areas: list[str] = field(default_factory=list)
    raw_response: str = ""
    should_change_approach: bool = False
    alternative_approach: str = ""
