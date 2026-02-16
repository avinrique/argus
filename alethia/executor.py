"""Subprocess-based code executor with resource limits."""

from __future__ import annotations

import ast
import re
import subprocess
import sys

from alethia.models import ExecutionResult, TestCase


class LocalExecutor:
    """Executes code locally via subprocess with resource limits."""

    def execute_code(
        self,
        code: str,
        stdin_input: str = "",
        timeout: int = 10,
        max_memory_mb: int = 256,
    ) -> ExecutionResult:
        return execute_code(code, stdin_input, timeout, max_memory_mb)

    def build_test_harness(
        self,
        solution_code: str,
        test_case: TestCase,
        function_signature: str | None,
    ) -> tuple[str, str]:
        return build_test_harness(solution_code, test_case, function_signature)

    def outputs_match(self, expected: str, actual: str) -> bool:
        return outputs_match(expected, actual)


# ---------------------------------------------------------------------------
# Module-level functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def execute_code(
    code: str,
    stdin_input: str = "",
    timeout: int = 10,
    max_memory_mb: int = 256,
) -> ExecutionResult:
    """Run Python code in a subprocess with resource limits."""
    # Build a wrapper that sets resource limits (Unix only) then execs the code
    wrapper = _build_wrapper(code, max_memory_mb)
    try:
        result = subprocess.run(
            [sys.executable, "-c", wrapper],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr="Execution timed out",
            exit_code=-1,
            timed_out=True,
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=str(e),
            exit_code=-1,
        )


def _build_wrapper(code: str, max_memory_mb: int) -> str:
    """Wrap user code with resource limits."""
    limit_bytes = max_memory_mb * 1024 * 1024
    # Use resource module to set memory limits on Unix
    return (
        "import resource as _resource\n"
        "try:\n"
        f"    _resource.setrlimit(_resource.RLIMIT_AS, ({limit_bytes}, {limit_bytes}))\n"
        "except (ValueError, _resource.error):\n"
        "    pass\n"
        "del _resource\n"
        f"{code}\n"
    )


def _extract_class_method(code: str) -> tuple[str, str] | None:
    """Detect class+method pattern in solution code dynamically.

    Scans the code for `class ClassName:` followed by `def method(self, ...)`.
    Returns (class_name, method_name) or None.
    """
    match = re.search(
        r"class\s+(\w+)\s*.*?:\s*\n"       # class Foo:
        r"(?:\s*(?:#.*|\"\"\".*?\"\"\"|\'\'\'.*?\'\'\')?\n)*"  # skip comments/docstrings
        r"\s+def\s+(\w+)\s*\(\s*self\b",   # def bar(self
        code,
        re.DOTALL,
    )
    if match:
        return match.group(1), match.group(2)
    return None


def build_test_harness(
    solution_code: str,
    test_case: TestCase,
    function_signature: str | None,
) -> tuple[str, str]:
    """Build executable code that runs a test case against the solution.

    Returns (full_code, stdin_input).

    Detection order:
    1. If the solution code contains a class with a self-method → instantiate & call
    2. If function_signature starts with "class " → old-style class harness
    3. If function_signature is a plain "def func(...)" → function harness
    4. Otherwise → stdin/stdout mode
    """
    # Auto-detect: scan the actual solution code for class+method pattern
    class_method = _extract_class_method(solution_code)
    if class_method:
        class_name, method_name = class_method
        return _build_method_harness(solution_code, test_case, class_name, method_name), ""

    if function_signature:
        sig = function_signature.strip()
        if sig.startswith("class "):
            return _build_class_harness(solution_code, test_case), ""
        return _build_function_harness(solution_code, test_case, function_signature), ""

    # stdin/stdout mode: just run the code with test_case.input as stdin
    return solution_code, test_case.input


def _build_method_harness(
    solution_code: str,
    test_case: TestCase,
    class_name: str,
    method_name: str,
) -> str:
    """Wrap a class-method solution (LeetCode, HackerEarth, etc.).

    Instantiates the class, calls the method with parsed args, prints result.
    Works for any class name and method name dynamically.
    """
    return (
        f"{solution_code}\n\n"
        f"_input_lines = {test_case.input!r}.strip().split(\"\\n\")\n"
        f"_args = [eval(line) for line in _input_lines if line.strip()]\n"
        f"_obj = {class_name}()\n"
        f"_result = _obj.{method_name}(*_args)\n"
        f"print(repr(_result))\n"
    )


def _build_function_harness(
    solution_code: str,
    test_case: TestCase,
    function_signature: str,
) -> str:
    """Wrap a function-based solution with a test harness."""
    # Extract function name from signature like "def two_sum(nums, target)"
    func_name = function_signature.split("(")[0].replace("def ", "").strip()

    # Parse the test input as Python literals
    # test_case.input is expected to be one Python expression per line for each arg
    harness = (
        f"{solution_code}\n\n"
        f"_input_lines = {test_case.input!r}.strip().split(\"\\n\")\n"
        f"_args = [eval(line) for line in _input_lines if line.strip()]\n"
        f"_result = {func_name}(*_args)\n"
        f"print(repr(_result))\n"
    )
    return harness


# Marker used by the verifier to separate prepended setup lines (whose return
# values should NOT be captured) from the main test-case lines.
_CLASS_SETUP_MARKER = "#__SETUP_END__"


def _build_class_harness(
    solution_code: str,
    test_case: TestCase,
) -> str:
    """Wrap a class-based solution with a test harness.

    The test case input is raw Python statements (e.g.
    ``obj = Trie()\\nobj.insert('apple')\\nobj.search('apple')``).
    Every statement's return value is captured into a list:
    assignments contribute ``None``; expressions contribute their value.
    If only one statement is present the bare value is printed instead of a
    one-element list so that state-dependent single-line test cases still match.
    """
    raw_input = test_case.input.strip()

    # The verifier may prepend accumulated setup lines separated by a marker.
    if _CLASS_SETUP_MARKER in raw_input:
        setup_part, main_part = raw_input.split(_CLASS_SETUP_MARKER, 1)
        setup_lines = [l for l in setup_part.strip().splitlines() if l.strip()]
        main_lines = [l for l in main_part.strip().splitlines() if l.strip()]
    else:
        setup_lines = []
        main_lines = [l for l in raw_input.splitlines() if l.strip()]

    if not main_lines:
        return f"{solution_code}\nprint(repr(None))\n"

    parts = [solution_code, ""]

    # Execute setup lines without capturing their results.
    parts.extend(setup_lines)

    # Capture results from the main test-case lines.
    parts.append("_results = []")
    for line in main_lines:
        if _is_assignment(line):
            parts.append(line)
            parts.append("_results.append(None)")
        else:
            parts.append(f"_results.append({line})")

    # Single result → print bare value; multiple → print the list.
    parts.append("if len(_results) == 1:")
    parts.append("    print(repr(_results[0]))")
    parts.append("else:")
    parts.append("    print(repr(_results))")
    return "\n".join(parts) + "\n"


def _is_assignment(line: str) -> bool:
    """Return True if *line* is a Python assignment statement."""
    try:
        tree = ast.parse(line.strip())
        return isinstance(tree.body[0], (ast.Assign, ast.AugAssign, ast.AnnAssign))
    except Exception:
        return False


def outputs_match(expected: str, actual: str) -> bool:
    """Compare expected and actual outputs flexibly."""
    expected = expected.strip()
    actual = actual.strip()

    # Exact match
    if expected == actual:
        return True

    # Normalize whitespace
    if " ".join(expected.split()) == " ".join(actual.split()):
        return True

    # Try parsing as Python literals for structural comparison
    try:
        expected_val = ast.literal_eval(expected)
        actual_val = ast.literal_eval(actual)
        if expected_val == actual_val:
            return True
        # Handle sets represented as lists (order-independent)
        if isinstance(expected_val, list) and isinstance(actual_val, list):
            if sorted(map(repr, expected_val)) == sorted(map(repr, actual_val)):
                return True
    except (ValueError, SyntaxError):
        pass

    # Normalize JSON-style null ↔ Python None, then retry literal comparison.
    # This handles expected outputs like "[null, null, True, False]" from
    # LeetCode-style class problems.
    norm_expected = expected.replace("null", "None")
    norm_actual = actual.replace("null", "None")
    if norm_expected == norm_actual:
        return True
    try:
        expected_val = ast.literal_eval(norm_expected)
        actual_val = ast.literal_eval(norm_actual)
        if expected_val == actual_val:
            return True
        if isinstance(expected_val, list) and isinstance(actual_val, list):
            if sorted(map(repr, expected_val)) == sorted(map(repr, actual_val)):
                return True
    except (ValueError, SyntaxError):
        pass

    return False
