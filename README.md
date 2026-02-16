# Argus

AI-powered coding problem solver that iteratively generates, verifies, and revises solutions until they pass. Built by [Avinrique](https://github.com/avinrique) (EarthlingAIDTech).

Inspired by the [Aletheia research paper](https://arxiv.org/abs/2602.10177v2) (Google DeepMind, 2026) which demonstrated that decoupling verification from generation dramatically improves solution quality.

## How It Works

Argus uses a **three-agent loop** that mirrors how expert programmers debug:

```
Generator  -->  Verifier  -->  (if rejected)  Reviser  -->  Generator  --> ...
```

1. **Generator** — Produces a candidate solution with an algorithm plan
2. **Verifier** — Runs test cases + independent LLM review (separate from generator's reasoning to avoid being misled)
3. **Reviser** — Analyzes failures, identifies root cause, provides structured feedback for the next attempt

The loop continues until a solution is accepted or the retry limit is reached. Key features:
- **Backtracking** — Tracks all attempts with scores, can branch from the best historical attempt instead of just the latest
- **Stagnation detection** — Detects when scores plateau and forces approach diversity
- **Stuck loop detection** — If the reviser keeps giving the same feedback, triggers a fresh start
- **Edge case generation** — Verifier auto-generates additional test cases beyond the provided ones
- **Multi-candidate** — Can generate multiple candidates per attempt and pick the best

## Features

- **Web UI** — Flask app with real-time SSE streaming of the solve process
- **Platform-agnostic code templates** — Paste LeetCode/HackerEarth/etc. boilerplate, solver outputs code that matches exactly
- **Image support** — Upload problem diagrams/images, all agents see them via vision API
- **Solution history** — Every solve attempt saved and browsable
- **Leaderboard** — Track solving stats across users
- **Problem management** — Tags, difficulty ratings, public/private visibility, export/import as JSON
- **Problem sets** — Group problems into playlists
- **Dashboard** — Charts for solve history, score distribution, success rate
- **Code editor** — CodeMirror editor with syntax highlighting for viewing/editing solutions
- **Solution explanations** — One-click LLM explanation of any solution with complexity analysis
- **Dark/light mode** — Toggle with localStorage persistence
- **Multi-provider LLM** — OpenAI or Ollama (local models), configurable per user

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key (or Ollama running locally)

### Install

```bash
git clone https://github.com/avinrique/argus.git
cd argus
pip install -e .
```

### Run the Web App

```bash
export OPENAI_API_KEY=sk-...
python -m alethia.web.app
```

Open http://localhost:5001, sign up, configure your LLM provider in Settings, and start solving.

### Run from CLI

```bash
export OPENAI_API_KEY=sk-...
alethia solve problems/two_sum.json
```

### Use with Ollama (free, local)

```bash
# Start Ollama with a model
ollama run llama3

# Set environment
export ALETHIA_LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434/v1
alethia solve problems/two_sum.json
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required for OpenAI) | OpenAI API key |
| `ALETHIA_LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `ALETHIA_MAX_ATTEMPTS` | `5` | Max solve attempts |
| `ALETHIA_EXECUTION_TIMEOUT` | `10` | Code execution timeout (seconds) |
| `ALETHIA_MAX_MEMORY_MB` | `256` | Memory limit for code execution |
| `ALETHIA_ENABLE_BACKTRACKING` | `true` | Enable backtracking to best historical attempt |
| `ALETHIA_CANDIDATES_PER_ATTEMPT` | `1` | Number of candidate solutions per attempt |
| `ALETHIA_EXECUTOR` | `local` | `local` or `judge0` |
| `FLASK_SECRET_KEY` | `dev-secret-key...` | Flask session secret (change in production) |

## Problem Format

Problems are JSON files:

```json
{
  "title": "Two Sum",
  "description": "Given an array of integers and a target, return indices of two numbers that add up to target.",
  "function_signature": "def two_sum(nums, target)",
  "code_template": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:",
  "constraints": "2 <= nums.length <= 10^4",
  "test_cases": [
    {
      "input": "[2, 7, 11, 15]\n9",
      "expected_output": "[0, 1]",
      "description": "Basic case"
    }
  ]
}
```

## Project Structure

```
alethia/
  agents/
    base.py          # Shared LLM calling logic with vision support
    generator.py     # Produces code solutions
    verifier.py      # Runs tests + LLM review + edge case generation
    reviser.py       # Analyzes failures, produces structured feedback
  web/
    app.py           # Flask app with 27 routes + SSE streaming
    db.py            # SQLite schema and migrations
    auth.py          # Session-based auth helpers
    templates/       # Jinja2 templates (Tailwind CSS)
  orchestrator.py    # Main solve loop with backtracking
  executor.py        # Local code execution with sandboxing
  config.py          # Environment-based configuration
  prompts.py         # All LLM prompt templates
  models.py          # Data models (Problem, Solution, TestCase, etc.)
tests/               # pytest test suite
problems/            # Example problem JSON files
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Research Background

The architecture is based on [*Towards Autonomous Mathematics Research*](https://arxiv.org/abs/2602.10177v2) (Aletheia, Google DeepMind, Feb 2026). Key insights adapted for coding:

- **Separate verification from generation** — The verifier operates independently from the generator's chain-of-thought to avoid being misled by reasoning traces
- **Tool use reduces hallucinations** — Actually executing code and running tests catches errors that pure LLM review misses
- **Admit failure gracefully** — Better to say "I can't solve this" than return incorrect solutions
- **Inference-time scaling** — More attempts and candidates yield better results, with diminishing returns

## License

MIT
