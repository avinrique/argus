# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Argus is a coding problem solver agent by Avinrique (EarthlingAIDTech), inspired by the Aletheia research paper (arXiv:2602.10177v2). The original Aletheia is a math research agent that iteratively generates, verifies, and revises solutions in natural language. This project adapts that architecture to solve coding problems.

## Architecture (Aletheia-Inspired Three-Agent Loop)

The core design uses three subagents that interact in a loop until a solution passes verification or a retry limit is reached:

1. **Generator** - Takes a coding problem and produces a candidate solution (code). On revision cycles, it also receives feedback from the Verifier.
2. **Verifier** - Evaluates the generated solution for correctness. This includes running tests, checking edge cases, and analyzing logical correctness. A key insight from the paper: decoupling verification from generation significantly improves solution quality because the generator may "bluff" through errors in its own reasoning trace.
3. **Reviser** - When the Verifier rejects a solution, the Reviser analyzes the failure, identifies the root cause, and provides structured feedback to the Generator for the next attempt.

The loop runs: Generator -> Verifier -> (if rejected) Reviser -> Generator -> ... until accepted or max attempts reached.

### Key Design Principles from the Paper

- **Separate verification from generation**: The Verifier must operate independently from the Generator's chain-of-thought to avoid being misled by the Generator's reasoning trace.
- **Tool use is critical**: The original paper found that tool use (search, code execution) dramatically reduced hallucinations. For coding problems, this means actually executing code, running test suites, and using linters.
- **Admit failure gracefully**: The agent should be able to say "I cannot solve this" rather than returning incorrect solutions. The paper showed this improved overall reliability.
- **Inference-time scaling**: More compute at inference time yields better results, with diminishing returns at extreme scales.

## Reference

The design is documented in `2602.10177v2.pdf` (Towards Autonomous Mathematics Research, Google DeepMind, Feb 2026). See especially:
- Section 2: Agent architecture (Figure 1), scaling laws
- Section 2.2: Agentic harness design rationale
- Section 2.3: Importance of tool use
