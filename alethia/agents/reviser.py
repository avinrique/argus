"""Reviser agent: analyzes failures and produces structured feedback."""

from __future__ import annotations

import re

from alethia.agents.base import BaseAgent
from alethia.models import Problem, RevisionFeedback, Solution, VerificationResult
from alethia.prompts import REVISER_SYSTEM, reviser_user_prompt


class ReviserAgent(BaseAgent):
    def revise(
        self,
        problem: Problem,
        solution: Solution,
        verification: VerificationResult,
        request_diversity: bool = False,
    ) -> RevisionFeedback:
        user_prompt = reviser_user_prompt(
            problem, solution, verification, request_diversity=request_diversity,
        )
        raw = self._call_llm(
            system=REVISER_SYSTEM,
            user=user_prompt,
            model=self.config.reviser_model,
            temperature=self.config.reviser_temperature,
            images=problem.images or None,
        )
        return self._parse_feedback(raw)

    def _parse_feedback(self, text: str) -> RevisionFeedback:
        root_cause = self._extract_field(text, "ROOT_CAUSE")
        suggested_fix = self._extract_field(text, "SUGGESTED_FIX")
        focus_areas_raw = self._extract_field(text, "FOCUS_AREAS")
        focus_areas = [a.strip() for a in focus_areas_raw.split(",") if a.strip()] if focus_areas_raw else []
        should_change_raw = self._extract_field(text, "SHOULD_CHANGE_APPROACH")
        should_change_approach = should_change_raw.strip().upper().startswith("YES")
        alternative_approach = self._extract_field(text, "ALTERNATIVE_APPROACH")
        return RevisionFeedback(
            root_cause=root_cause or "Unable to determine root cause.",
            suggested_fix=suggested_fix or "Please review and fix the failing cases.",
            focus_areas=focus_areas,
            raw_response=text,
            should_change_approach=should_change_approach,
            alternative_approach=alternative_approach,
        )

    def _extract_field(self, text: str, field: str) -> str:
        pattern = rf"{field}:\s*(.*?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
