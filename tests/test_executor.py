"""Tests for the code executor."""

from alethia.executor import execute_code, outputs_match, build_test_harness
from alethia.models import TestCase


def test_simple_execution():
    result = execute_code("print('hello')")
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert not result.timed_out


def test_timeout():
    result = execute_code("import time; time.sleep(100)", timeout=1)
    assert result.timed_out
    assert result.exit_code == -1


def test_crash():
    result = execute_code("raise ValueError('boom')")
    assert result.exit_code != 0
    assert "ValueError" in result.stderr


def test_stdin():
    result = execute_code("x = input(); print(f'got {x}')", stdin_input="hello\n")
    assert result.exit_code == 0
    assert result.stdout.strip() == "got hello"


def test_outputs_match_exact():
    assert outputs_match("42", "42")


def test_outputs_match_whitespace():
    assert outputs_match("  42  ", "42")
    assert outputs_match("hello\n", "hello")


def test_outputs_match_python_literals():
    assert outputs_match("[1, 2]", "[1,2]")
    assert outputs_match("(1, 2, 3)", "(1,2,3)")


def test_outputs_match_list_order():
    assert outputs_match("[0, 1]", "[1, 0]")


def test_outputs_no_match():
    assert not outputs_match("42", "43")
    assert not outputs_match("[1, 2]", "[1, 3]")


def test_function_harness():
    code = "def add(a, b):\n    return a + b"
    tc = TestCase(input="3\n4", expected_output="7")
    full_code, stdin_input = build_test_harness(code, tc, "def add(a, b)")
    assert "add" in full_code
    assert stdin_input == ""
    result = execute_code(full_code)
    assert result.exit_code == 0
    assert result.stdout.strip() == "7"


def test_stdin_mode_harness():
    code = "n = int(input()); print(n * 2)"
    tc = TestCase(input="5", expected_output="10")
    full_code, stdin_input = build_test_harness(code, tc, None)
    assert full_code == code
    assert stdin_input == "5"
    result = execute_code(full_code, stdin_input=stdin_input)
    assert result.exit_code == 0
    assert result.stdout.strip() == "10"
