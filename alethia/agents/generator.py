"""Generator agent: produces code solutions from problem descriptions."""

from __future__ import annotations

import re

from alethia.agents.base import BaseAgent, extract_code_block
from alethia.models import Problem, RevisionFeedback, Solution
from alethia.prompts import GENERATOR_SYSTEM, generator_user_prompt


class GeneratorAgent(BaseAgent):
    def generate(
        self,
        problem: Problem,
        attempt: int = 1,
        feedback: RevisionFeedback | None = None,
        previous_code: str | None = None,
        previous_plan: str | None = None,
        fresh_start_hint: bool = False,
        strategy_hint: str | None = None,
        temperature_override: float | None = None,
    ) -> Solution:
        user_prompt = generator_user_prompt(
            problem, attempt, feedback, previous_code,
            previous_plan=previous_plan,
            fresh_start_hint=fresh_start_hint,
            strategy_hint=strategy_hint,
        )
        temperature = temperature_override if temperature_override is not None else self.config.generator_temperature
        raw = self._call_llm(
            system=GENERATOR_SYSTEM,
            user=user_prompt,
            model=self.config.generator_model,
            temperature=temperature,
            images=problem.images or None,
        )
        code = extract_code_block(raw)
        plan = self._extract_plan(raw)
        return Solution(code=code, attempt=attempt, raw_response=raw, plan=plan)

    @staticmethod
    def _extract_plan(text: str) -> str:
        """Extract content from <plan>...</plan> block."""
        match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
        return match.group(1).strip() if match else ""
