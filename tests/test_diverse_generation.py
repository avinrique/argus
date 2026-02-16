"""Tests for AlphaCode-style diverse initial generation."""

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
from alethia.orchestrator import Orchestrator
from alethia.prompts import DEFAULT_STRATEGIES, GenerationStrategy, generator_user_prompt


def _make_config(**overrides) -> Config:
    defaults = {
        "openai_api_key": "test-key",
        "max_attempts": 3,
        "enable_backtracking": True,
        "candidates_per_attempt": 1,
        "diverse_generation": True,
    }
    defaults.update(overrides)
    return Config(**defaults)


PROBLEM = Problem(
    title="Test",
    description="Test problem",
    test_cases=[TestCase(input="1", expected_output="2")],
)


class TestGenerationStrategy:
    def test_default_strategies_has_four(self):
        assert len(DEFAULT_STRATEGIES) == 4

    def test_strategy_is_frozen(self):
        s = DEFAULT_STRATEGIES[0]
        assert isinstance(s, GenerationStrategy)
        try:
            s.name = "changed"  # type: ignore[misc]
            assert False, "Should not be able to mutate frozen dataclass"
        except AttributeError:
            pass

    def test_strategy_names(self):
        names = [s.name for s in DEFAULT_STRATEGIES]
        assert "optimal" in names
        assert "brute_force_first" in names
        assert "creative" in names
        assert "dp_or_greedy" in names


class TestStrategyHintInPrompt:
    def test_hint_appended_to_prompt(self):
        prompt = generator_user_prompt(PROBLEM, attempt=1, strategy_hint="Use dynamic programming.")
        assert "## Strategy Hint" in prompt
        assert "Use dynamic programming." in prompt

    def test_no_hint_when_none(self):
        prompt = generator_user_prompt(PROBLEM, attempt=1, strategy_hint=None)
        assert "## Strategy Hint" not in prompt

    def test_no_hint_when_empty(self):
        prompt = generator_user_prompt(PROBLEM, attempt=1, strategy_hint="")
        assert "## Strategy Hint" not in prompt


class TestSolutionStrategy:
    def test_strategy_field_default_empty(self):
        s = Solution(code="pass", attempt=1)
        assert s.strategy == ""

    def test_strategy_field_can_be_set(self):
        s = Solution(code="pass", attempt=1, strategy="optimal")
        assert s.strategy == "optimal"


class TestConfigDiverseGeneration:
    def test_default_is_true(self):
        config = Config(openai_api_key="test")
        assert config.diverse_generation is True

    def test_from_env_disabled(self):
        import os
        env = {"OPENAI_API_KEY": "test", "ALETHIA_DIVERSE_GENERATION": "false"}
        with patch.dict(os.environ, env, clear=False):
            config = Config.from_env()
            assert config.diverse_generation is False

    def test_override(self):
        config = _make_config(diverse_generation=False)
        assert config.diverse_generation is False


class TestDiverseOrchestrator:
    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_diverse_generates_four_candidates_on_attempt_1(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=1, diverse_generation=True)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        # 4 diverse candidates, all rejected
        gen.generate.side_effect = [
            Solution(code="optimal_code", attempt=1),
            Solution(code="brute_code", attempt=1),
            Solution(code="creative_code", attempt=1),
            Solution(code="dp_code", attempt=1),
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.5),
            VerificationResult(verdict=Verdict.REJECTED, score=0.8),
            VerificationResult(verdict=Verdict.REJECTED, score=0.3),
            VerificationResult(verdict=Verdict.REJECTED, score=0.6),
        ]
        rev.revise.return_value = RevisionFeedback(root_cause="bug", suggested_fix="fix")

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert not result.solved
        # Generator should have been called 4 times (one per strategy)
        assert gen.generate.call_count == 4
        # The best candidate (brute_code, score 0.8) should be kept
        assert result.history[0].solution.code == "brute_code"
        assert result.history[0].score == 0.8

        # Each call should have received strategy_hint and temperature_override
        for i, call in enumerate(gen.generate.call_args_list):
            assert call.kwargs.get("strategy_hint") is not None or call[1].get("strategy_hint") is not None
            assert call.kwargs.get("temperature_override") is not None or call[1].get("temperature_override") is not None

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_diverse_disabled_falls_back_to_candidates_per_attempt(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=1, diverse_generation=False, candidates_per_attempt=2)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        gen.generate.side_effect = [
            Solution(code="c1", attempt=1),
            Solution(code="c2", attempt=1),
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.5),
            VerificationResult(verdict=Verdict.REJECTED, score=0.7),
        ]
        rev.revise.return_value = RevisionFeedback(root_cause="bug", suggested_fix="fix")

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert not result.solved
        # Generator should have been called 2 times (candidates_per_attempt), NOT 4
        assert gen.generate.call_count == 2
        # Calls should NOT have strategy_hint
        for call in gen.generate.call_args_list:
            assert call.kwargs.get("strategy_hint") is None
            assert call.kwargs.get("temperature_override") is None

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_diverse_accepts_early_if_candidate_passes(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=2, diverse_generation=True)

        gen = MockGen.return_value
        ver = MockVer.return_value

        # Second diverse candidate is accepted
        gen.generate.side_effect = [
            Solution(code="optimal_code", attempt=1),
            Solution(code="brute_code_good", attempt=1),
            Solution(code="creative_code", attempt=1),
            Solution(code="dp_code", attempt=1),
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.5),
            VerificationResult(verdict=Verdict.ACCEPTED, score=1.0),
        ]

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert result.solved
        assert result.solution.code == "brute_code_good"
        assert result.attempts == 1

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_attempt_2_uses_normal_flow_even_with_diverse_enabled(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=2, diverse_generation=True, candidates_per_attempt=1)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        # Attempt 1: 4 diverse candidates, all rejected
        # Attempt 2: 1 normal candidate, rejected
        gen.generate.side_effect = [
            Solution(code="opt", attempt=1),
            Solution(code="brute", attempt=1),
            Solution(code="creative", attempt=1),
            Solution(code="dp", attempt=1),
            Solution(code="revised", attempt=2),
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.3),
            VerificationResult(verdict=Verdict.REJECTED, score=0.5),
            VerificationResult(verdict=Verdict.REJECTED, score=0.2),
            VerificationResult(verdict=Verdict.REJECTED, score=0.4),
            VerificationResult(verdict=Verdict.REJECTED, score=0.6),
        ]
        rev.revise.side_effect = [
            RevisionFeedback(root_cause="off by one", suggested_fix="fix loop"),
            RevisionFeedback(root_cause="wrong algorithm", suggested_fix="try dp"),
        ]

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        assert not result.solved
        # Attempt 1: 4 diverse calls + Attempt 2: 1 normal call = 5 total
        assert gen.generate.call_count == 5
        # The 5th call (attempt 2) should NOT have strategy_hint
        last_call = gen.generate.call_args_list[4]
        assert last_call.kwargs.get("strategy_hint") is None
        assert last_call.kwargs.get("temperature_override") is None

    @patch("alethia.orchestrator.create_executor")
    @patch("alethia.orchestrator.ReviserAgent")
    @patch("alethia.orchestrator.VerifierAgent")
    @patch("alethia.orchestrator.GeneratorAgent")
    def test_diverse_sets_strategy_on_solutions(
        self, MockGen, MockVer, MockRev, mock_create_exec
    ):
        mock_create_exec.return_value = MagicMock()
        config = _make_config(max_attempts=1, diverse_generation=True)

        gen = MockGen.return_value
        ver = MockVer.return_value
        rev = MockRev.return_value

        gen.generate.side_effect = [
            Solution(code="c1", attempt=1),
            Solution(code="c2", attempt=1),
            Solution(code="c3", attempt=1),
            Solution(code="c4", attempt=1),
        ]
        ver.verify.side_effect = [
            VerificationResult(verdict=Verdict.REJECTED, score=0.1),
            VerificationResult(verdict=Verdict.REJECTED, score=0.9),
            VerificationResult(verdict=Verdict.REJECTED, score=0.1),
            VerificationResult(verdict=Verdict.REJECTED, score=0.1),
        ]
        rev.revise.return_value = RevisionFeedback(root_cause="bug", suggested_fix="fix")

        orch = Orchestrator(config)
        result = orch.solve(PROBLEM)

        # The best solution should have the strategy name set
        assert result.history[0].solution.strategy == "brute_force_first"
