"""Verifier agent: runs tests and performs LLM logical review."""

from __future__ import annotations

import json
import re
import sys

from alethia.agents.base import BaseAgent
from alethia.executor import LocalExecutor, _CLASS_SETUP_MARKER
from alethia.executor_base import CodeExecutor
from alethia.models import (
    Problem,
    Solution,
    TestCase,
    TestCaseResult,
    Verdict,
    VerificationResult,
)
from alethia.prompts import (
    EDGE_CASE_GENERATOR_SYSTEM,
    VERIFIER_SYSTEM,
    edge_case_generator_user_prompt,
    verifier_user_prompt,
)


class VerifierAgent(BaseAgent):
    def __init__(self, config, executor: CodeExecutor | None = None) -> None:
        super().__init__(config)
        self._executor: CodeExecutor = executor or LocalExecutor()
        self._edge_cases_generated: bool = False

    def verify(self, problem: Problem, solution: Solution) -> VerificationResult:
        # Phase 0: Generate edge case tests (once per problem)
        if not self._edge_cases_generated:
            self._generate_edge_cases(problem)
            self._edge_cases_generated = True

        # Phase 1: Execute test cases
        test_results = self._run_tests(problem, solution)
        all_tests_passed = all(tr.passed for tr in test_results)
        passed_count = sum(1 for tr in test_results if tr.passed)
        total_tests = len(test_results)
        score = passed_count / total_tests if total_tests > 0 else 0.0

        # Build test summary for LLM
        test_summary = self._format_test_summary(test_results)

        # Phase 2: LLM logical review (independent of generator's reasoning)
        user_prompt = verifier_user_prompt(problem, solution, test_summary)
        llm_response = self._call_llm(
            system=VERIFIER_SYSTEM,
            user=user_prompt,
            model=self.config.verifier_model,
            temperature=self.config.verifier_temperature,
            images=problem.images or None,
        )

        llm_accepts = "VERDICT: ACCEPT" in llm_response

        # Combined verdict: REJECTED if any test fails OR LLM rejects
        if all_tests_passed and llm_accepts:
            verdict = Verdict.ACCEPTED
            summary = "All tests passed and code review accepted."
        elif not all_tests_passed:
            failed = total_tests - passed_count
            summary = f"{failed}/{total_tests} test(s) failed."
        else:
            summary = "Tests passed but code review identified issues."

        if not all_tests_passed or not llm_accepts:
            verdict = Verdict.REJECTED

        return VerificationResult(
            verdict=verdict,
            test_results=test_results,
            llm_reasoning=llm_response,
            summary=summary,
            score=score,
        )

    def _run_tests(self, problem: Problem, solution: Solution) -> list[TestCaseResult]:
        is_class_based = (
            problem.function_signature
            and problem.function_signature.strip().startswith("class ")
        )
        # For class-based problems, accumulate setup lines across test cases
        # so that state-dependent tests (e.g. trie.search('app') after trie.insert('apple'))
        # get the prior setup prepended automatically.
        accumulated_setup: list[str] = []

        results = []
        for tc in problem.test_cases:
            effective_tc = tc
            if is_class_based:
                effective_tc = self._augment_class_test_case(
                    tc, accumulated_setup, problem.function_signature or ""
                )

            code, stdin_input = self._executor.build_test_harness(
                solution.code, effective_tc, problem.function_signature
            )
            execution = self._executor.execute_code(
                code,
                stdin_input=stdin_input,
                timeout=self.config.execution_timeout,
                max_memory_mb=self.config.max_memory_mb,
            )
            actual = execution.stdout.strip()
            passed = execution.exit_code == 0 and self._executor.outputs_match(
                tc.expected_output, actual
            )
            results.append(
                TestCaseResult(
                    test_case=tc,
                    execution=execution,
                    passed=passed,
                    actual_output=actual,
                )
            )

            if is_class_based:
                self._update_accumulated_setup(
                    tc, accumulated_setup, problem.function_signature or ""
                )

        return results

    @staticmethod
    def _extract_class_name(function_signature: str) -> str:
        """Extract the class name from a signature like 'class Trie:'."""
        match = re.match(r"class\s+(\w+)", function_signature.strip())
        return match.group(1) if match else ""

    def _augment_class_test_case(
        self,
        tc: TestCase,
        accumulated_setup: list[str],
        function_signature: str,
    ) -> TestCase:
        """Prepend accumulated setup lines if the test case isn't self-contained."""
        if not accumulated_setup:
            return tc
        class_name = self._extract_class_name(function_signature)
        lines = [l for l in tc.input.strip().splitlines() if l.strip()]
        # Self-contained if any line constructs a new instance of the class
        is_self_contained = any(f"{class_name}(" in l for l in lines) if class_name else False
        if is_self_contained:
            return tc
        # Prepend accumulated setup with a marker so the harness knows not to
        # capture return values for the setup portion.
        augmented_input = (
            "\n".join(accumulated_setup)
            + "\n" + _CLASS_SETUP_MARKER + "\n"
            + "\n".join(lines)
        )
        return TestCase(
            input=augmented_input,
            expected_output=tc.expected_output,
            description=tc.description,
            is_generated=tc.is_generated,
        )

    def _update_accumulated_setup(
        self,
        tc: TestCase,
        accumulated_setup: list[str],
        function_signature: str,
    ) -> None:
        """Update accumulated setup lines after running a test case."""
        class_name = self._extract_class_name(function_signature)
        lines = [l for l in tc.input.strip().splitlines() if l.strip()]
        is_self_contained = any(f"{class_name}(" in l for l in lines) if class_name else False
        if is_self_contained:
            # Reset: new instance created, old state is irrelevant
            accumulated_setup.clear()
        # Add all lines as setup — every line is a state-changing operation
        # that subsequent test cases may depend on.
        accumulated_setup.extend(lines)

    def _format_test_summary(self, results: list[TestCaseResult]) -> str:
        lines = []
        for i, tr in enumerate(results, 1):
            status = "PASS" if tr.passed else "FAIL"
            lines.append(f"Test {i}: {status}")
            lines.append(f"  Expected: {tr.test_case.expected_output.strip()}")
            lines.append(f"  Actual:   {tr.actual_output}")
            if tr.execution.stderr:
                lines.append(f"  Stderr:   {tr.execution.stderr[:500]}")
            if tr.execution.timed_out:
                lines.append("  (Timed out)")
        return "\n".join(lines)

    def _generate_edge_cases(self, problem: Problem) -> None:
        """Generate edge case test cases using an LLM call. Appends to problem.test_cases."""
        try:
            user_prompt = edge_case_generator_user_prompt(problem)
            raw = self._call_llm(
                system=EDGE_CASE_GENERATOR_SYSTEM,
                user=user_prompt,
                model=self.config.verifier_model,
                temperature=0.3,
                images=problem.images or None,
            )
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[: cleaned.rfind("```")]
            data = json.loads(cleaned)
            cases = data.get("test_cases", [])
            added = 0
            for tc_data in cases:
                if "input" in tc_data and "expected_output" in tc_data:
                    problem.test_cases.append(
                        TestCase(
                            input=str(tc_data["input"]),
                            expected_output=str(tc_data["expected_output"]),
                            description=tc_data.get("description", "Generated edge case"),
                            is_generated=True,
                        )
                    )
                    added += 1
            print(f"Generated {added} edge case test(s).", file=sys.stderr)
        except Exception as exc:
            print(f"Edge case generation failed (continuing with original tests): {exc}", file=sys.stderr)
