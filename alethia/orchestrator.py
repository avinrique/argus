"""Main Generator -> Verifier -> Reviser loop with backtracking."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from alethia.agents.generator import GeneratorAgent
from alethia.agents.reviser import ReviserAgent
from alethia.agents.verifier import VerifierAgent
from alethia.config import Config
from alethia.executor_base import CodeExecutor
from alethia.executor_factory import create_executor
from alethia.models import (
    Problem,
    RevisionFeedback,
    Solution,
    Verdict,
    VerificationResult,
)


@dataclass
class AttemptRecord:
    attempt: int
    solution: Solution
    verification: VerificationResult
    feedback: RevisionFeedback | None = None
    score: float = 0.0
    parent_attempt: int | None = None  # which attempt this was revised from


@dataclass
class SolveResult:
    solved: bool
    solution: Solution | None
    attempts: int
    history: list[AttemptRecord] = field(default_factory=list)


class Orchestrator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._executor: CodeExecutor = create_executor(config)
        self.generator = GeneratorAgent(config)
        self.verifier = VerifierAgent(config, executor=self._executor)
        self.reviser = ReviserAgent(config)

    def solve(self, problem: Problem) -> SolveResult:
        self._log(f"\n{'='*60}")
        self._log(f"Solving: {problem.title}")
        self._log(f"Max attempts: {self.config.max_attempts}")
        self._log(f"Model: {self.config.generator_model}")
        self._log(f"Backtracking: {'enabled' if self.config.enable_backtracking else 'disabled'}")
        self._log(f"Candidates per attempt: {self.config.candidates_per_attempt}")
        self._log(f"{'='*60}\n")

        history: list[AttemptRecord] = []
        force_fresh_start = False

        for attempt in range(1, self.config.max_attempts + 1):
            self._log(f"--- Attempt {attempt}/{self.config.max_attempts} ---")

            # Select revision base
            if force_fresh_start or not history:
                feedback = None
                previous_code = None
                previous_plan = None
                parent_attempt = None
                fresh_start_hint = force_fresh_start
                force_fresh_start = False
            elif self.config.enable_backtracking:
                base = self._select_best_attempt(history)
                self._log(f"Backtracking: revising from attempt {base.attempt} (score {base.score:.2f})")
                feedback = base.feedback
                previous_code = base.solution.code
                previous_plan = base.solution.plan
                parent_attempt = base.attempt
                fresh_start_hint = False
            else:
                # Linear mode: always use the latest attempt
                base = history[-1]
                feedback = base.feedback
                previous_code = base.solution.code
                previous_plan = base.solution.plan
                parent_attempt = base.attempt
                fresh_start_hint = False

            # Generate candidate(s)
            candidates = self._generate_candidates(
                problem, attempt, feedback, previous_code,
                self.config.candidates_per_attempt,
                previous_plan=previous_plan,
                fresh_start_hint=fresh_start_hint,
            )

            # Verify each candidate and keep the best
            best_solution: Solution | None = None
            best_verification: VerificationResult | None = None
            best_score = -1.0

            for i, candidate in enumerate(candidates):
                if len(candidates) > 1:
                    self._log(f"Verifying candidate {i+1}/{len(candidates)}...")
                else:
                    self._log("Verifying solution...")
                verification = self.verifier.verify(problem, candidate)
                self._log(f"Verdict: {verification.verdict.value} (score: {verification.score:.2f})")

                if verification.verdict == Verdict.ACCEPTED:
                    self._log(f"\nSolution ACCEPTED on attempt {attempt}!")
                    record = AttemptRecord(
                        attempt=attempt,
                        solution=candidate,
                        verification=verification,
                        score=verification.score,
                        parent_attempt=parent_attempt,
                    )
                    history.append(record)
                    return SolveResult(
                        solved=True,
                        solution=candidate,
                        attempts=attempt,
                        history=history,
                    )

                if verification.score > best_score:
                    best_score = verification.score
                    best_solution = candidate
                    best_verification = verification

            assert best_solution is not None
            assert best_verification is not None

            # Stagnation detection: has the score improved recently?
            request_diversity = self._detect_stagnation(history, best_score)
            if request_diversity:
                self._log("Score stagnation detected — requesting approach diversity from reviser.")

            # Best candidate was rejected — generate revision feedback
            self._log("Solution rejected. Generating revision feedback...")
            rev_feedback = self.reviser.revise(
                problem, best_solution, best_verification,
                request_diversity=request_diversity,
            )
            self._log(f"Root cause: {rev_feedback.root_cause[:200]}")
            self._log(f"Suggested fix: {rev_feedback.suggested_fix[:200]}")
            if rev_feedback.should_change_approach:
                self._log(f"Approach change suggested: {rev_feedback.alternative_approach[:200]}")

            # Stuck loop detection: same root cause repeated
            if self._detect_stuck_loop(history, rev_feedback):
                self._log("Stuck loop detected — forcing fresh start on next attempt.")
                force_fresh_start = True

            record = AttemptRecord(
                attempt=attempt,
                solution=best_solution,
                verification=best_verification,
                feedback=rev_feedback,
                score=best_verification.score,
                parent_attempt=parent_attempt,
            )
            history.append(record)

        # Return the best solution found across all attempts
        best = self._select_best_attempt(history)
        self._log(f"\nFailed after {self.config.max_attempts} attempts. Best score: {best.score:.2f}")
        return SolveResult(
            solved=False,
            solution=best.solution,
            attempts=self.config.max_attempts,
            history=history,
        )

    @staticmethod
    def _select_best_attempt(history: list[AttemptRecord]) -> AttemptRecord:
        """Return the highest-scoring attempt, breaking ties by recency."""
        return max(history, key=lambda r: (r.score, r.attempt))

    def _generate_candidates(
        self,
        problem: Problem,
        attempt: int,
        feedback: RevisionFeedback | None,
        previous_code: str | None,
        n: int,
        previous_plan: str | None = None,
        fresh_start_hint: bool = False,
    ) -> list[Solution]:
        """Generate n candidate solutions."""
        candidates = []
        for i in range(n):
            if n > 1:
                self._log(f"Generating candidate {i+1}/{n}...")
            else:
                self._log("Generating solution...")
            solution = self.generator.generate(
                problem, attempt, feedback, previous_code,
                previous_plan=previous_plan,
                fresh_start_hint=fresh_start_hint,
            )
            self._log(f"Generated {len(solution.code)} chars of code.")
            if solution.plan:
                self._log(f"Plan: {solution.plan[:200]}")
            candidates.append(solution)
        return candidates

    @staticmethod
    def _detect_stagnation(history: list[AttemptRecord], current_score: float) -> bool:
        """Return True if score hasn't improved for 2 consecutive attempts."""
        if len(history) < 2:
            return False
        recent_scores = [r.score for r in history[-2:]]
        return all(current_score <= s for s in recent_scores)

    @staticmethod
    def _detect_stuck_loop(
        history: list[AttemptRecord], new_feedback: RevisionFeedback,
    ) -> bool:
        """Return True if the last 2 revisions have near-identical root causes."""
        recent_with_feedback = [
            r for r in history if r.feedback is not None
        ]
        if not recent_with_feedback:
            return False
        prev_feedback = recent_with_feedback[-1].feedback
        assert prev_feedback is not None
        return _word_overlap(prev_feedback.root_cause, new_feedback.root_cause) > 0.8

    def _log(self, message: str) -> None:
        print(message, file=sys.stderr)


def _word_overlap(a: str, b: str) -> float:
    """Return the fraction of overlapping words between two strings (Jaccard)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
