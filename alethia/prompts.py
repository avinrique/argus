"""Centralized prompt templates for all agents."""

from __future__ import annotations

from alethia.models import Problem, RevisionFeedback, Solution, VerificationResult

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = """\
You are an expert competitive programmer and software engineer.

Rules:
- Write clean, correct, efficient Python 3 code.
- Before writing code, output your algorithm plan inside a <plan> block:
  <plan>
  - Approach: [algorithm name/strategy]
  - Data structures: [what you'll use]
  - Time complexity: O(...)
  - Space complexity: O(...)
  - Key insight: [one sentence]
  </plan>
- Then write the solution code inside a single ```python code fence.
- Do NOT include test code, input reading (unless the problem requires stdin), or explanations outside the fence.
- **IMPORTANT — Code Template Rule:**
  If the problem includes a "Code Template", your output MUST complete that \
template exactly. Do NOT rename the class, method, or parameters. Do NOT \
remove type hints. Do NOT change the structure. Your code must be directly \
pasteable into the platform's submission box (LeetCode, HackerEarth, \
Codeforces, etc.). Add any required imports (typing, collections, etc.) \
above the class/function.
- If only a plain function signature is given (no template), implement that exact function.
- If the problem uses stdin/stdout, read from stdin and print to stdout.
- Think carefully about edge cases before coding.
- If you receive revision feedback, pay close attention to the root cause and fix it."""


def generator_user_prompt(
    problem: Problem,
    attempt: int,
    feedback: RevisionFeedback | None = None,
    previous_code: str | None = None,
    previous_plan: str | None = None,
    fresh_start_hint: bool = False,
) -> str:
    parts = [
        f"# {problem.title}\n",
        problem.description,
    ]
    if problem.constraints:
        parts.append(f"\n## Constraints\n{problem.constraints}")

    # Code template takes priority over plain function signature
    if problem.code_template:
        parts.append(
            f"\n## Code Template (your output MUST match this format exactly)\n"
            f"```python\n{problem.code_template}\n```"
        )
    elif problem.function_signature:
        parts.append(f"\n## Function Signature\n`{problem.function_signature}`")

    if problem.test_cases:
        parts.append("\n## Examples")
        for i, tc in enumerate(problem.test_cases, 1):
            parts.append(f"\n### Example {i}" + (f" — {tc.description}" if tc.description else ""))
            parts.append(f"Input:\n```\n{tc.input}\n```")
            parts.append(f"Expected Output:\n```\n{tc.expected_output}\n```")

    if attempt > 1 and feedback and previous_code:
        parts.append(f"\n---\n## Attempt {attempt} — Revision Required")
        if feedback.should_change_approach and feedback.alternative_approach:
            parts.append(
                "The reviser has determined your current approach is fundamentally wrong. "
                f"Try: {feedback.alternative_approach}. "
                "Do NOT refine your previous code — write a completely new solution."
            )
        parts.append(f"Your previous code (from the best-scoring attempt so far):\n```python\n{previous_code}\n```")
        if previous_plan:
            parts.append(f"Your previous plan:\n<plan>\n{previous_plan}\n</plan>")
        parts.append(f"Root cause of failure: {feedback.root_cause}")
        parts.append(f"Suggested fix: {feedback.suggested_fix}")
        if feedback.focus_areas:
            parts.append("Focus areas: " + ", ".join(feedback.focus_areas))

    if fresh_start_hint:
        parts.append(
            "\n---\nPrevious approaches have repeatedly failed with similar issues. "
            "Try a fundamentally different algorithm."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM = """\
You are an independent code reviewer evaluating a solution for correctness.

You will receive:
1. The problem statement
2. The candidate solution code
3. Test execution results (pass/fail for each test case)

Your job:
- Analyze the code for logical correctness, edge cases, and potential bugs.
- Even if all tests pass, check for issues like integer overflow, off-by-one errors, \
incorrect assumptions, or missing edge cases.
- You must end your response with exactly one of:
  VERDICT: ACCEPT
  VERDICT: REJECT

Be rigorous. Only ACCEPT if you are confident the solution is correct."""


def verifier_user_prompt(
    problem: Problem,
    solution: Solution,
    test_summary: str,
) -> str:
    parts = [
        f"# Problem: {problem.title}\n",
        problem.description,
    ]
    if problem.constraints:
        parts.append(f"\nConstraints: {problem.constraints}")
    parts.append(f"\n## Candidate Solution (Attempt {solution.attempt})\n```python\n{solution.code}\n```")
    parts.append(f"\n## Test Execution Results\n{test_summary}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Edge Case Generator
# ---------------------------------------------------------------------------

EDGE_CASE_GENERATOR_SYSTEM = """\
You are a testing expert. Given a coding problem, its function signature, constraints, \
and existing test cases, generate additional edge case test cases that are likely to \
expose bugs.

Focus on:
- Boundary values (0, 1, -1, empty input, single element)
- Maximum constraints (largest allowed input size, max/min integer values)
- Duplicates and repeated elements
- Negative numbers (if applicable)
- Already sorted / reverse sorted input
- All-same-value input
- Minimal valid input

Return a JSON object with a key "test_cases" containing an array of 3-5 test case objects, \
each with:
- "input": string — one Python expression per line, one per function argument
- "expected_output": string — the repr() of the expected return value
- "description": string — short label describing the edge case

Return ONLY valid JSON. No markdown fences, no extra text."""


def edge_case_generator_user_prompt(problem: Problem) -> str:
    parts = [
        f"# {problem.title}\n",
        problem.description,
    ]
    if problem.constraints:
        parts.append(f"\n## Constraints\n{problem.constraints}")
    if problem.function_signature:
        parts.append(f"\n## Function Signature\n`{problem.function_signature}`")
    if problem.test_cases:
        parts.append("\n## Existing Test Cases")
        for i, tc in enumerate(problem.test_cases, 1):
            parts.append(f"\nTest {i}:" + (f" ({tc.description})" if tc.description else ""))
            parts.append(f"  Input: {tc.input}")
            parts.append(f"  Expected: {tc.expected_output}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reviser
# ---------------------------------------------------------------------------

REVISER_SYSTEM = """\
You are a debugging expert. You analyze failed coding solutions and provide actionable feedback.

You will receive the problem, the failing solution, and verification results (test failures + reviewer notes).

Respond with EXACTLY this structure:

ROOT_CAUSE: <one-paragraph explanation of why the solution failed>
SUGGESTED_FIX: <specific, actionable instructions for fixing the code>
FOCUS_AREAS: <comma-separated list of things to pay attention to>
SHOULD_CHANGE_APPROACH: YES or NO — is the current algorithm fundamentally wrong for this problem?
ALTERNATIVE_APPROACH: If YES above, suggest a completely different algorithm/strategy. If NO, leave blank.

Be specific and constructive. Reference exact lines or logic errors when possible."""


PARSER_SYSTEM = """\
You are a coding problem parser. Given raw text copied from a coding problem website \
(LeetCode, HackerEarth, GeeksforGeeks, Codeforces, etc.), extract the structured fields.

Return a JSON object with exactly these keys:
- "title": string — the problem title
- "description": string — the full problem statement (exclude examples/constraints)
- "function_signature": string — a simple Python function signature like \
"def twoSum(nums, target)". Only the function name and parameter names, no types, \
no class wrapper. If none is apparent, leave as "".
- "code_template": string — the EXACT code template/boilerplate from the platform, \
preserved as-is in Python. For example:
  - LeetCode: "class Solution:\\n    def twoSum(self, nums: List[int], target: int) -> List[int]:"
  - HackerEarth: the full boilerplate with input/output handling if provided
  - If the problem provides a class with a method, keep the class, self, type hints, \
return type — everything exactly as the platform expects.
  - If the problem gives a signature in another language (Java, C++, etc.), convert it \
to the equivalent Python format preserving the structure (class wrapper if it had one, \
type hints mapped to Python typing equivalents).
  - If there is no code template (e.g. Codeforces stdin/stdout), leave as "".
- "constraints": string — the constraints section verbatim, or "" if none
- "test_cases": array of objects, each with:
  - "input": string — one Python expression per line, one per function argument \
(excluding self). For example, if the method takes (self, nums, target), the input \
should be two lines: [2, 7, 11, 15]\\n9
  - "expected_output": string — the repr() of the expected return value, e.g. "[0, 1]"
  - "description": string — a short label like "Example 1" or ""

Rules:
- Return ONLY valid JSON, no markdown fences, no extra text.
- Convert all values to Python types (use True/False not true/false in test case values, \
  use Python list syntax not array syntax).
- If the problem has no explicit test cases, create at least one from any examples in the text.
- If you cannot determine a field, use an empty string or empty array."""


def reviser_user_prompt(
    problem: Problem,
    solution: Solution,
    verification: VerificationResult,
    request_diversity: bool = False,
) -> str:
    parts = [
        f"# Problem: {problem.title}\n",
        problem.description,
    ]
    if solution.plan:
        parts.append(f"\n## Generator's Algorithm Plan\n<plan>\n{solution.plan}\n</plan>")
    parts.extend([
        f"\n## Failed Solution (Attempt {solution.attempt})\n```python\n{solution.code}\n```",
        f"\n## Verification Results",
        f"Verdict: {verification.verdict.value}",
    ])
    if request_diversity:
        parts.append(
            "\n**NOTE:** The score has not improved across multiple attempts. "
            "Consider whether the fundamental algorithmic approach needs to change."
        )
    if verification.test_results:
        parts.append("\n### Test Case Results")
        for i, tr in enumerate(verification.test_results, 1):
            status = "PASS" if tr.passed else "FAIL"
            parts.append(f"\nTest {i}: {status}")
            parts.append(f"  Input: {tr.test_case.input}")
            parts.append(f"  Expected: {tr.test_case.expected_output}")
            parts.append(f"  Actual: {tr.actual_output}")
            if tr.execution.stderr:
                parts.append(f"  Stderr: {tr.execution.stderr}")
    if verification.llm_reasoning:
        parts.append(f"\n### Reviewer Notes\n{verification.llm_reasoning}")
    return "\n".join(parts)
