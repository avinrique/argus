"""Tests for data models and code block extraction."""

from alethia.agents.base import extract_code_block
from alethia.models import Problem, TestCase, Verdict


def test_extract_python_fence():
    text = 'Here is the code:\n```python\ndef foo():\n    return 42\n```\nDone.'
    assert extract_code_block(text) == "def foo():\n    return 42"


def test_extract_generic_fence():
    text = 'Code:\n```\nprint("hi")\n```'
    assert extract_code_block(text) == 'print("hi")'


def test_extract_no_fence():
    text = "def foo():\n    return 42"
    assert extract_code_block(text) == text


def test_verdict_enum():
    assert Verdict.ACCEPTED.value == "ACCEPTED"
    assert Verdict.REJECTED.value == "REJECTED"


def test_problem_creation():
    p = Problem(
        title="Test",
        description="A test problem",
        test_cases=[TestCase(input="1", expected_output="2")],
    )
    assert p.title == "Test"
    assert len(p.test_cases) == 1
    assert p.function_signature is None
