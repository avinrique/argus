"""Tests for backtracking orchestrator logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alethia.config import Config
from alethia.models import (
    Problem,
    RevisionFeedback,
    Solution,
    TestCase,
    Verdict,
    VerificationResult,
)
from alethia.orchestrator import AttemptRecord, Orchestrator


def _make_config(**overrides) -> Config:
    defaults = {
        "openai_api_key": "test-key",
        "max_attempts": 3,
        "enable_backtracking": True,
        "candidates_per_attempt": 1,
        "diverse_generation": False,
    }
    defaults.update(overrides)
    return Config(**defaults)


def _make_record(attempt: int, score: float, code: str = "pass") -> AttemptRecord:
    return AttemptRecord(
        attempt=attempt,
        solution=Solution(code=code, attempt=attempt),
        verification=VerificationResult(verdict=Verdict.REJECTED, score=score),
        feedback=RevisionFeedback(root_cause="bug", suggested_fix="fix it"),
        score=score,
    )


PROBLEM = Problem(
    title="Test",
    description="Test problem",
    test_cases=[TestCase(input="1", expected_output="2")],
)


class TestSelectBestAttempt:
    def test_picks_highest_score(self):
        history = [_make_record(1, 0.33), _make_record(2, 0.67), _make_record(3, 0.33)]
        best = Orchestrator._select_best_attempt(history)
        assert best.attempt == 2
        assert best.score == 0.67

    def test_breaks_ties_by_recency(self):
        history = [_make_record(1, 0.5), _make_record(2, 0.5), _make_record(3, 0.5)]
        best = Orchestrator._select_best_attempt(history)
        assert best.attempt == 3  # most recent among tied scores


class TestBacktrackingOrchestrator:
    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_backtracking_reverts_to_better_attempt(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=3, enable_backtracking=True)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        # Attempt 1: score 0.67
        # Attempt 2: score 0.33 (regression)
        # Attempt 3: should revise from attempt 1 (higher score)
        solutions = [
            Solution(code="code_v1", attempt=1),
            Solution(code="code_v2", attempt=2),
            Solution(code="code_v3", attempt=3),
        ]
        gen.generate.side_effect = solutions

        verifications = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.67, summary="1/3 failed"),
            VerificationResult(verdict=Verdict.REJECTED, score=0.33, summary="2/3 failed"),
            VerificationResult(verdict=Verdict.REJECTED, score=0.33, summary="2/3 failed"),
        ]
        ver.verify.side_effect = verifications

        # Use different root causes to avoid triggering stuck loop detection
        rev.revise.side_effect = [
            RevisionFeedback(root_cause="off by one error in loop", suggested_fix="fix loop bounds"),
            RevisionFeedback(root_cause="wrong data structure choice", suggested_fix="use a hash map"),
            RevisionFeedback(root_cause="missing base case", suggested_fix="add base case"),
        ]

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert not result.solved
        # Best solution returned should be from attempt 1 (score 0.67)
        assert result.solution is not None
        assert result.solution.code == "code_v1"

        # Attempt 3 should have parent_attempt=1 (backtracked to best)
        assert result.history[2].parent_attempt == 1

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_linear_mode_always_uses_latest(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=3, enable_backtracking=False)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        solutions = [
            Solution(code="v1", attempt=1),
            Solution(code="v2", attempt=2),
            Solution(code="v3", attempt=3),
        ]
        gen.generate.side_effect = solutions

        verifications = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.67, summary="fail"),
            VerificationResult(verdict=Verdict.REJECTED, score=0.33, summary="fail"),
            VerificationResult(verdict=Verdict.REJECTED, score=0.0, summary="fail"),
        ]
        ver.verify.side_effect = verifications

        # Use different root causes to avoid triggering stuck loop detection
        rev.revise.side_effect = [
            RevisionFeedback(root_cause="off by one error in loop", suggested_fix="fix loop bounds"),
            RevisionFeedback(root_cause="wrong data structure choice", suggested_fix="use a hash map"),
            RevisionFeedback(root_cause="missing base case", suggested_fix="add base case"),
        ]

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert not result.solved
        # In linear mode, attempt 3 should have parent_attempt=2 (latest, not best)
        assert result.history[2].parent_attempt == 2

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_multi_candidate_keeps_best(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=1, candidates_per_attempt=3)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        # 3 candidates, all rejected but with different scores
        gen.generate.side_effect = [
            Solution(code="c1", attempt=1),
            Solution(code="c2", attempt=1),
            Solution(code="c3", attempt=1),
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.33, summary="fail"),
            VerificationResult(verdict=Verdict.REJECTED, score=0.67, summary="fail"),
            VerificationResult(verdict=Verdict.REJECTED, score=0.33, summary="fail"),
        ]
        rev.revise.return_value = RevisionFeedback(root_cause="bug", suggested_fix="fix")

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert not result.solved
        # The best candidate (c2, score 0.67) should be kept
        assert result.history[0].solution.code == "c2"
        assert result.history[0].score == 0.67

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_multi_candidate_accepts_early(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=2, candidates_per_attempt=3)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        # Second candidate is accepted
        gen.generate.side_effect = [
            Solution(code="c1", attempt=1),
            Solution(code="c2_good", attempt=1),
            Solution(code="c3", attempt=1),  # should not be reached
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.33, summary="fail"),
            VerificationResult(verdict=Verdict.ACCEPTED, score=1.0, summary="pass"),
        ]

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert result.solved
        assert result.solution.code == "c2_good"
        assert result.attempts == 1

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_failed_solve_returns_best_solution(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=2)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        gen.generate.side_effect = [
            Solution(code="v1", attempt=1),
            Solution(code="v2", attempt=2),
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.67, summary="fail"),
            VerificationResult(verdict=Verdict.REJECTED, score=0.33, summary="fail"),
        ]
        rev.revise.return_value = RevisionFeedback(root_cause="bug", suggested_fix="fix")

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert not result.solved
        # Should return the best solution (v1 with score 0.67), not None
        assert result.solution is not None
        assert result.solution.code == "v1"
