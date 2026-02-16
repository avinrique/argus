"""CLI interface for Alethia."""

from __future__ import annotations

import argparse
import json
import sys

from alethia.config import Config
from alethia.models import Problem, TestCase
from alethia.orchestrator import Orchestrator


def load_problem(path: str) -> Problem:
    """Load a problem from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    test_cases = [
        TestCase(
            input=tc["input"],
            expected_output=tc["expected_output"],
            description=tc.get("description", ""),
        )
        for tc in data.get("test_cases", [])
    ]
    return Problem(
        title=data["title"],
        description=data["description"],
        test_cases=test_cases,
        function_signature=data.get("function_signature"),
        constraints=data.get("constraints", ""),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="alethia",
        description="Alethia: Coding problem solver agent",
    )
    subparsers = parser.add_subparsers(dest="command")

    solve_parser = subparsers.add_parser("solve", help="Solve a coding problem")
    solve_parser.add_argument("problem", help="Path to problem JSON file")
    solve_parser.add_argument("--max-attempts", type=int, default=None)
    solve_parser.add_argument("--model", type=str, default=None)
    solve_parser.add_argument("-o", "--output", type=str, default=None, help="Write solution to file")
    solve_parser.add_argument(
        "--executor", choices=["local", "judge0"], default=None, help="Execution backend"
    )
    solve_parser.add_argument("--judge0-url", type=str, default=None, help="Judge0 API base URL")
    solve_parser.add_argument("--judge0-api-key", type=str, default=None, help="Judge0 API key")
    solve_parser.add_argument(
        "--no-backtrack", action="store_true", default=False, help="Disable backtracking (linear mode)"
    )
    solve_parser.add_argument(
        "--candidates", type=int, default=None, help="Candidates per attempt"
    )

    args = parser.parse_args(argv)

    if args.command != "solve":
        parser.print_help()
        sys.exit(1)

    # Build config from env + CLI overrides
    overrides = {}
    if args.max_attempts is not None:
        overrides["max_attempts"] = args.max_attempts
    if args.model is not None:
        overrides["model"] = args.model
    if args.executor is not None:
        overrides["executor_type"] = args.executor
    if args.judge0_url is not None:
        overrides["judge0_url"] = args.judge0_url
    if args.judge0_api_key is not None:
        overrides["judge0_api_key"] = args.judge0_api_key
    if args.no_backtrack:
        overrides["enable_backtracking"] = False
    if args.candidates is not None:
        overrides["candidates_per_attempt"] = args.candidates

    try:
        config = Config.from_env(**overrides)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    problem = load_problem(args.problem)
    orchestrator = Orchestrator(config)
    result = orchestrator.solve(problem)

    if result.solved and result.solution:
        print(result.solution.code)
        if args.output:
            with open(args.output, "w") as f:
                f.write(result.solution.code + "\n")
            print(f"\nSolution written to {args.output}", file=sys.stderr)
    else:
        print(f"Failed to solve '{problem.title}' after {result.attempts} attempts.", file=sys.stderr)
        sys.exit(1)
