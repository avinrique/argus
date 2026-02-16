"""Flask web application for Argus."""

from __future__ import annotations

import json
import os
import queue
import re
import threading

from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, session, stream_with_context, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from alethia.config import Config
from alethia.models import Problem, RevisionFeedback, Solution, TestCase, Verdict, VerificationResult
from alethia.orchestrator import AttemptRecord, Orchestrator, SolveResult
from alethia.prompts import DEFAULT_STRATEGIES, PARSER_SYSTEM
from alethia.web.auth import get_current_user_id, login_required
from alethia.web.db import close_db, get_db, init_db

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
app.teardown_appcontext(close_db)

UPLOADS_DIR = Path(__file__).resolve().parent.parent.parent / "instance" / "uploads"

with app.app_context():
    init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    """Convert a title to a URL-friendly slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _problem_from_row(row) -> Problem:
    """Convert a SQLite row to a Problem dataclass."""
    test_cases_raw = json.loads(row["test_cases_json"])
    test_cases = [
        TestCase(
            input=tc["input"],
            expected_output=tc["expected_output"],
            description=tc.get("description", ""),
        )
        for tc in test_cases_raw
    ]
    images = json.loads(row["images_json"]) if "images_json" in row.keys() and row["images_json"] else []
    return Problem(
        title=row["title"],
        description=row["description"],
        test_cases=test_cases,
        function_signature=row["function_signature"] or None,
        constraints=row["constraints"],
        code_template=row["code_template"] if "code_template" in row.keys() else "",
        images=images,
    )


def _config_for_current_user(**overrides) -> Config:
    """Build a Config using the current user's LLM settings as defaults."""
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (get_current_user_id(),)).fetchone()
    if user is None:
        raise ValueError("Not logged in")

    user_overrides: dict = {}
    if user["llm_provider"]:
        user_overrides["llm_provider"] = user["llm_provider"]
    if user["openai_api_key"]:
        user_overrides["openai_api_key"] = user["openai_api_key"]
    if user["ollama_base_url"]:
        user_overrides["ollama_base_url"] = user["ollama_base_url"]
    if user["preferred_model"]:
        user_overrides["model"] = user["preferred_model"]

    # Caller overrides take priority
    user_overrides.update(overrides)
    return Config.from_env(**user_overrides)


def _collect_test_cases() -> list[dict]:
    """Collect test cases from form data."""
    test_cases = []
    i = 0
    while True:
        tc_input = request.form.get(f"tc_input_{i}")
        tc_output = request.form.get(f"tc_output_{i}")
        if tc_input is None and tc_output is None:
            break
        if tc_input and tc_output:
            test_cases.append(
                {
                    "input": tc_input.strip(),
                    "expected_output": tc_output.strip(),
                    "description": request.form.get(f"tc_desc_{i}", "").strip(),
                }
            )
        i += 1
    return test_cases


def _user_has_llm_config(user_row) -> bool:
    """Check whether a user has configured enough to run the solver."""
    if user_row is None:
        return False
    if user_row["llm_provider"] == "ollama":
        return bool(user_row["ollama_base_url"])
    # openai
    return bool(user_row["openai_api_key"])


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _save_uploaded_images(slug: str) -> list[str]:
    """Save uploaded image files and return list of absolute paths."""
    files = request.files.getlist("images")
    if not files:
        return []
    upload_dir = UPLOADS_DIR / slug
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        # Sanitize filename
        safe_name = re.sub(r"[^\w.\-]", "_", f.filename)
        dest = upload_dir / safe_name
        f.save(str(dest))
        saved.append(str(dest))
    return saved


# ---------------------------------------------------------------------------
# Context processor — makes `current_user` available in all templates
# ---------------------------------------------------------------------------

@app.context_processor
def inject_user():
    user_id = get_current_user_id()
    if user_id is None:
        return {"current_user": None}
    db = get_db()
    user = db.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
    return {"current_user": user}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    password_confirm = request.form.get("password_confirm", "")

    if not email or not password:
        return render_template("signup.html", error="Email and password are required.", email=email)
    if len(password) < 6:
        return render_template("signup.html", error="Password must be at least 6 characters.", email=email)
    if password != password_confirm:
        return render_template("signup.html", error="Passwords do not match.", email=email)

    db = get_db()
    if db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
        return render_template("signup.html", error="An account with that email already exists.", email=email)

    password_hash = generate_password_hash(password)
    cursor = db.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
    db.commit()
    session["user_id"] = cursor.lastrowid
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid email or password.", email=email)

    session["user_id"] = user["id"]
    next_url = request.args.get("next") or url_for("index")
    return redirect(next_url)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/uploads/<path:filepath>")
def serve_upload(filepath: str):
    """Serve uploaded image files."""
    return send_from_directory(str(UPLOADS_DIR), filepath)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (get_current_user_id(),)).fetchone()

    if request.method == "GET":
        return render_template("settings.html", user=user)

    llm_provider = request.form.get("llm_provider", "openai")
    openai_api_key = request.form.get("openai_api_key", "").strip()
    ollama_base_url = request.form.get("ollama_base_url", "").strip()
    preferred_model = request.form.get("preferred_model", "").strip()

    db.execute(
        """UPDATE users
           SET llm_provider = ?, openai_api_key = ?, ollama_base_url = ?, preferred_model = ?
           WHERE id = ?""",
        (llm_provider, openai_api_key, ollama_base_url, preferred_model, get_current_user_id()),
    )
    db.commit()

    user = db.execute("SELECT * FROM users WHERE id = ?", (get_current_user_id(),)).fetchone()
    return render_template("settings.html", user=user, success=True)


# ---------------------------------------------------------------------------
# StreamingOrchestrator (unchanged)
# ---------------------------------------------------------------------------

class StreamingOrchestrator(Orchestrator):
    """Orchestrator that emits rich structured events for the web UI."""

    def __init__(self, config, log_queue):
        super().__init__(config)
        self._queue = log_queue

    def _emit(self, event_type: str, **kwargs):
        """Emit a structured event to the frontend."""
        self._queue.put({"type": event_type, **kwargs})

    def _log(self, message):
        super()._log(message)
        self._emit("log", message=message)

    def solve(self, problem: Problem) -> SolveResult:
        # Emit config overview
        self._emit("phase", agent="config", title="Configuration")
        self._emit("detail", label="Problem", value=problem.title)
        self._emit("detail", label="Test cases", value=str(len(problem.test_cases)))
        if problem.function_signature:
            self._emit("detail", label="Signature", value=problem.function_signature)
        self._emit("detail", label="Model", value=self.config.generator_model)
        self._emit("detail", label="Max attempts", value=str(self.config.max_attempts))
        self._emit("detail", label="Backtracking",
                   value="enabled" if self.config.enable_backtracking else "disabled")
        self._emit("detail", label="Candidates/attempt",
                   value=str(self.config.candidates_per_attempt))
        self._emit("detail", label="Diverse generation",
                   value="enabled" if self.config.diverse_generation else "disabled")

        history: list[AttemptRecord] = []
        force_fresh_start = False

        for attempt in range(1, self.config.max_attempts + 1):
            self._emit("attempt", number=attempt, total=self.config.max_attempts)

            # Select revision base
            feedback = None
            previous_code = None
            previous_plan = None
            parent_attempt = None
            fresh_start_hint = False

            if force_fresh_start or not history:
                fresh_start_hint = force_fresh_start
                force_fresh_start = False
                self._emit("phase", agent="generator", title="Generator")
                if fresh_start_hint:
                    self._emit("detail", label="Mode", value="Fresh start — stuck loop detected")
                else:
                    self._emit("detail", label="Mode", value="First attempt — generating from scratch")
                self._emit("detail", label="Input",
                           value=f"Problem description ({len(problem.description)} chars) + {len(problem.test_cases)} test examples")
            elif self.config.enable_backtracking:
                base = self._select_best_attempt(history)
                feedback = base.feedback
                previous_code = base.solution.code
                previous_plan = base.solution.plan
                parent_attempt = base.attempt
                self._emit("backtrack",
                           from_attempt=base.attempt,
                           score=round(base.score, 2),
                           total_history=len(history))
                self._emit("phase", agent="generator", title="Generator")
                self._emit("detail", label="Mode",
                           value=f"Revising attempt {base.attempt} (score: {base.score:.0%})")
                self._emit("detail", label="Previous code", value=f"{len(previous_code)} chars")
                if feedback:
                    self._emit("detail", label="Root cause fed in",
                               value=feedback.root_cause[:150])
                    self._emit("detail", label="Fix instructions",
                               value=feedback.suggested_fix[:150])
            else:
                base = history[-1]
                feedback = base.feedback
                previous_code = base.solution.code
                previous_plan = base.solution.plan
                parent_attempt = base.attempt
                self._emit("phase", agent="generator", title="Generator")
                self._emit("detail", label="Mode",
                           value=f"Linear revision of attempt {base.attempt}")
                self._emit("detail", label="Previous code", value=f"{len(previous_code)} chars")
                if feedback:
                    self._emit("detail", label="Root cause fed in",
                               value=feedback.root_cause[:150])

            # Generate candidates
            candidates: list[Solution] = []
            use_diverse = self.config.diverse_generation and not history and not feedback
            if use_diverse:
                for strategy in DEFAULT_STRATEGIES:
                    self._emit("log", message=f"  Generating with strategy '{strategy.name}' (temp={strategy.temperature})...")
                    solution = self.generator.generate(
                        problem, attempt,
                        strategy_hint=strategy.hint,
                        temperature_override=strategy.temperature,
                    )
                    solution.strategy = strategy.name
                    self._emit("code_preview", code=solution.code[:300],
                               total_chars=len(solution.code),
                               strategy=strategy.name)
                    candidates.append(solution)
            else:
                for i in range(self.config.candidates_per_attempt):
                    if self.config.candidates_per_attempt > 1:
                        self._emit("log", message=f"  Generating candidate {i+1}/{self.config.candidates_per_attempt}...")
                    else:
                        self._emit("log", message="  Calling LLM to generate code...")
                    solution = self.generator.generate(
                        problem, attempt, feedback, previous_code,
                        previous_plan=previous_plan,
                        fresh_start_hint=fresh_start_hint,
                    )
                    self._emit("code_preview", code=solution.code[:300],
                               total_chars=len(solution.code))
                    candidates.append(solution)

            # Verify each candidate
            best_solution: Solution | None = None
            best_verification: VerificationResult | None = None
            best_score = -1.0

            for i, candidate in enumerate(candidates):
                self._emit("phase", agent="verifier", title="Verifier")
                if len(candidates) > 1:
                    self._emit("detail", label="Candidate", value=f"{i+1}/{len(candidates)}")
                self._emit("log", message="  Running test cases...")

                verification = self.verifier.verify(problem, candidate)

                # Emit individual test results
                for j, tr in enumerate(verification.test_results):
                    desc = tr.test_case.description or f"Test {j+1}"
                    if tr.passed:
                        self._emit("test_result", name=desc, passed=True)
                    else:
                        self._emit("test_result", name=desc, passed=False,
                                   expected=tr.test_case.expected_output,
                                   actual=tr.actual_output,
                                   stderr=tr.execution.stderr[:200] if tr.execution.stderr else "")

                passed = sum(1 for tr in verification.test_results if tr.passed)
                total = len(verification.test_results)
                self._emit("detail", label="Tests passed", value=f"{passed}/{total}")
                self._emit("log", message="  Running LLM review (independent from generator)...")
                self._emit("verdict",
                           value=verification.verdict.value,
                           score=round(verification.score, 2))

                if verification.verdict == Verdict.ACCEPTED:
                    self._emit("accepted", attempt=attempt)
                    record = AttemptRecord(
                        attempt=attempt, solution=candidate,
                        verification=verification, score=verification.score,
                        parent_attempt=parent_attempt,
                    )
                    history.append(record)
                    return SolveResult(solved=True, solution=candidate,
                                      attempts=attempt, history=history)

                if verification.score > best_score:
                    best_score = verification.score
                    best_solution = candidate
                    best_verification = verification

            assert best_solution is not None
            assert best_verification is not None

            # Stagnation detection
            request_diversity = self._detect_stagnation(history, best_score)
            if request_diversity:
                self._emit("log", message="  Score stagnation detected — requesting approach diversity.")

            # Revise
            self._emit("phase", agent="reviser", title="Reviser")
            self._emit("detail", label="Input",
                       value=f"Failed solution ({len(best_solution.code)} chars) + verification results")
            self._emit("log", message="  Analyzing failure and generating feedback...")

            rev_feedback = self.reviser.revise(
                problem, best_solution, best_verification,
                request_diversity=request_diversity,
            )

            self._emit("feedback",
                       root_cause=rev_feedback.root_cause,
                       suggested_fix=rev_feedback.suggested_fix,
                       focus_areas=rev_feedback.focus_areas)

            # Stuck loop detection
            if self._detect_stuck_loop(history, rev_feedback):
                self._emit("log", message="  Stuck loop detected — forcing fresh start on next attempt.")
                force_fresh_start = True

            record = AttemptRecord(
                attempt=attempt, solution=best_solution,
                verification=best_verification, feedback=rev_feedback,
                score=best_verification.score, parent_attempt=parent_attempt,
            )
            history.append(record)

        # Failed
        best = self._select_best_attempt(history)
        self._emit("failed", best_score=round(best.score, 2),
                   total_attempts=self.config.max_attempts)
        return SolveResult(solved=False, solution=best.solution,
                           attempts=self.config.max_attempts, history=history)


def _build_result_data(solve_result):
    """Convert a SolveResult into a JSON-serializable dict."""
    result = {
        "solved": solve_result.solved,
        "attempts": solve_result.attempts,
        "code": solve_result.solution.code if solve_result.solution else "",
        "score": solve_result.history[-1].score if solve_result.history else 0,
        "history": [],
    }
    for record in solve_result.history:
        attempt_data = {
            "attempt": record.attempt,
            "score": record.score,
            "verdict": record.verification.verdict.value,
            "code": record.solution.code,
            "strategy": record.solution.strategy,
            "summary": record.verification.summary,
            "test_results": [],
        }
        for tr in record.verification.test_results:
            attempt_data["test_results"].append(
                {
                    "description": tr.test_case.description,
                    "passed": tr.passed,
                    "expected": tr.test_case.expected_output,
                    "actual": tr.actual_output,
                    "stderr": tr.execution.stderr,
                }
            )
        if record.feedback:
            attempt_data["feedback"] = {
                "root_cause": record.feedback.root_cause,
                "suggested_fix": record.feedback.suggested_fix,
                "focus_areas": record.feedback.focus_areas,
            }
        result["history"].append(attempt_data)
    return result


# ---------------------------------------------------------------------------
# Problem routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    db = get_db()
    user_id = get_current_user_id()
    tag_filter = request.args.get("tag", "").strip().lower()
    difficulty_filter = request.args.get("difficulty", "").strip().lower()

    # Show all public problems + current user's private ones
    if user_id is not None:
        rows = db.execute(
            "SELECT id, slug, title, function_signature, test_cases_json, tags, difficulty, is_public, user_id "
            "FROM problems WHERE is_public = 1 OR user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, slug, title, function_signature, test_cases_json, tags, difficulty, is_public, user_id "
            "FROM problems WHERE is_public = 1 ORDER BY created_at DESC"
        ).fetchall()

    # Collect all tags for the filter bar
    all_tags: set[str] = set()
    problems = []
    for r in rows:
        tc = json.loads(r["test_cases_json"])
        row_tags = [t.strip() for t in r["tags"].split(",") if t.strip()] if r["tags"] else []
        for t in row_tags:
            all_tags.add(t)
        # Apply tag filter
        if tag_filter and tag_filter not in row_tags:
            continue
        # Apply difficulty filter
        if difficulty_filter and r["difficulty"] != difficulty_filter:
            continue
        # Count solutions for this problem
        sol_count = db.execute(
            "SELECT COUNT(*) as cnt FROM solutions WHERE problem_id = ?", (r["id"],)
        ).fetchone()["cnt"]
        problems.append(
            {
                "slug": r["slug"],
                "title": r["title"],
                "function_signature": r["function_signature"],
                "test_case_count": len(tc),
                "tags": row_tags,
                "difficulty": r["difficulty"],
                "is_public": r["is_public"],
                "is_owner": user_id is not None and user_id == r["user_id"],
                "solution_count": sol_count,
            }
        )
    return render_template(
        "index.html", problems=problems, all_tags=sorted(all_tags),
        tag_filter=tag_filter, difficulty_filter=difficulty_filter,
    )


@app.route("/create", methods=["GET"])
@login_required
def create_form():
    return render_template("create.html")


@app.route("/create/parse", methods=["POST"])
@login_required
def create_parse():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("raw_text") or "").strip()
    if not raw_text:
        return jsonify({"error": "raw_text is required"}), 400

    try:
        config = _config_for_current_user()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    client = config.create_openai_client()
    try:
        response = client.chat.completions.create(
            model=config.generator_model,
            messages=[
                {"role": "system", "content": PARSER_SYSTEM},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
    except Exception as e:
        return jsonify({"error": f"LLM parsing failed: {e}"}), 500

    return jsonify(parsed)


@app.route("/create", methods=["POST"])
@login_required
def create_submit():
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    function_signature = request.form.get("function_signature", "").strip()
    code_template = request.form.get("code_template", "").strip()
    constraints = request.form.get("constraints", "").strip()
    tags = ",".join(t.strip().lower() for t in request.form.get("tags", "").split(",") if t.strip())
    difficulty = request.form.get("difficulty", "medium").strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    is_public = 1 if request.form.get("is_public") != "0" else 0

    if not title or not description:
        return render_template(
            "create.html",
            error="Title and description are required.",
            form=request.form,
        )

    test_cases = _collect_test_cases()
    if not test_cases:
        return render_template(
            "create.html",
            error="At least one test case is required.",
            form=request.form,
        )

    slug = slugify(title)
    db = get_db()

    # Ensure slug uniqueness
    if db.execute("SELECT 1 FROM problems WHERE slug = ?", (slug,)).fetchone():
        return render_template(
            "create.html",
            error="A problem with a similar title already exists.",
            form=request.form,
        )

    # Handle image uploads
    image_paths = _save_uploaded_images(slug)

    db.execute(
        """INSERT INTO problems (slug, user_id, title, description, function_signature, code_template, constraints, test_cases_json, tags, difficulty, is_public, images_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (slug, get_current_user_id(), title, description, function_signature, code_template, constraints, json.dumps(test_cases), tags, difficulty, is_public, json.dumps(image_paths)),
    )
    db.commit()
    return redirect(url_for("problem_detail", slug=slug))


@app.route("/problem/<slug>")
def problem_detail(slug: str):
    db = get_db()
    row = db.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return "Problem not found", 404

    user_id = get_current_user_id()
    is_owner = user_id is not None and user_id == row["user_id"]

    # Private problem: only owner can see
    if not row["is_public"] and not is_owner:
        return "Problem not found", 404

    # Build a dict that templates can use
    test_cases = json.loads(row["test_cases_json"])
    tags = [t.strip() for t in row["tags"].split(",") if t.strip()] if row["tags"] else []
    code_template = row["code_template"] if "code_template" in row.keys() else ""
    images_raw = json.loads(row["images_json"]) if "images_json" in row.keys() and row["images_json"] else []
    # Convert absolute paths to web-servable URLs
    image_urls = []
    for img_path in images_raw:
        p = Path(img_path)
        if p.is_file():
            # /uploads/<slug>/filename.png
            rel = p.relative_to(UPLOADS_DIR)
            image_urls.append(f"/uploads/{rel}")
    problem = {
        "title": row["title"],
        "description": row["description"],
        "function_signature": row["function_signature"],
        "code_template": code_template,
        "constraints": row["constraints"],
        "test_cases": test_cases,
        "tags": tags,
        "difficulty": row["difficulty"],
        "is_public": row["is_public"],
        "images": image_urls,
    }

    # Count solutions
    solution_count = db.execute(
        "SELECT COUNT(*) as cnt FROM solutions WHERE problem_id = ?", (row["id"],)
    ).fetchone()["cnt"]

    # Determine LLM config availability
    has_llm_config = False
    if user_id is not None:
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        has_llm_config = _user_has_llm_config(user)

    return render_template(
        "problem.html", problem=problem, slug=slug,
        is_owner=is_owner, has_llm_config=has_llm_config,
        solution_count=solution_count, problem_id=row["id"],
    )


@app.route("/problem/<slug>/edit", methods=["GET", "POST"])
@login_required
def edit_problem(slug: str):
    db = get_db()
    row = db.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return "Problem not found", 404
    if row["user_id"] != get_current_user_id():
        return "Forbidden", 403

    existing_images = json.loads(row["images_json"]) if "images_json" in row.keys() and row["images_json"] else []
    # Build web URLs for existing images
    existing_image_urls = []
    for img_path in existing_images:
        p = Path(img_path)
        if p.is_file():
            rel = p.relative_to(UPLOADS_DIR)
            existing_image_urls.append(f"/uploads/{rel}")

    if request.method == "GET":
        problem = {
            "title": row["title"],
            "description": row["description"],
            "function_signature": row["function_signature"],
            "code_template": row["code_template"] if "code_template" in row.keys() else "",
            "constraints": row["constraints"],
            "test_cases": json.loads(row["test_cases_json"]),
            "tags": row["tags"],
            "difficulty": row["difficulty"],
            "is_public": row["is_public"],
            "images": existing_image_urls,
        }
        return render_template("create.html", edit_mode=True, problem=problem, slug=slug)

    # POST — save edits
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    function_signature = request.form.get("function_signature", "").strip()
    code_template = request.form.get("code_template", "").strip()
    constraints = request.form.get("constraints", "").strip()
    tags = ",".join(t.strip().lower() for t in request.form.get("tags", "").split(",") if t.strip())
    difficulty = request.form.get("difficulty", "medium").strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    is_public = 1 if request.form.get("is_public") != "0" else 0

    if not title or not description:
        problem = {
            "title": title, "description": description,
            "function_signature": function_signature, "code_template": code_template,
            "constraints": constraints,
            "test_cases": json.loads(row["test_cases_json"]),
            "tags": tags, "difficulty": difficulty, "is_public": is_public,
        }
        return render_template(
            "create.html", edit_mode=True, problem=problem, slug=slug,
            error="Title and description are required.", form=request.form,
        )

    test_cases = _collect_test_cases()
    if not test_cases:
        problem = {
            "title": title, "description": description,
            "function_signature": function_signature, "code_template": code_template,
            "constraints": constraints,
            "test_cases": json.loads(row["test_cases_json"]),
            "tags": tags, "difficulty": difficulty, "is_public": is_public,
        }
        return render_template(
            "create.html", edit_mode=True, problem=problem, slug=slug,
            error="At least one test case is required.", form=request.form,
        )

    # Handle image uploads — merge new uploads with existing images
    new_image_paths = _save_uploaded_images(slug)
    # Keep existing images unless user uploaded replacements
    all_images = existing_images + new_image_paths if new_image_paths else existing_images

    db.execute(
        """UPDATE problems
           SET title = ?, description = ?, function_signature = ?, code_template = ?,
               constraints = ?, test_cases_json = ?, tags = ?, difficulty = ?, is_public = ?,
               images_json = ?
           WHERE slug = ?""",
        (title, description, function_signature, code_template, constraints, json.dumps(test_cases), tags, difficulty, is_public, json.dumps(all_images), slug),
    )
    db.commit()
    return redirect(url_for("problem_detail", slug=slug))


@app.route("/problem/<slug>/solve", methods=["POST"])
@login_required
def solve_problem(slug: str):
    db = get_db()
    row = db.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return json.dumps({"type": "error", "message": "Problem not found"}), 404

    # Read config overrides from form
    overrides: dict = {}
    model = request.form.get("model", "").strip()
    if model:
        overrides["model"] = model
    max_attempts = request.form.get("max_attempts", "").strip()
    if max_attempts:
        overrides["max_attempts"] = int(max_attempts)
    backtracking = request.form.get("backtracking")
    if backtracking == "off":
        overrides["enable_backtracking"] = False
    candidates = request.form.get("candidates", "").strip()
    if candidates:
        overrides["candidates_per_attempt"] = int(candidates)
    diverse = request.form.get("diverse_generation")
    if diverse == "off":
        overrides["diverse_generation"] = False

    try:
        config = _config_for_current_user(**overrides)
    except ValueError as e:
        return json.dumps({"type": "error", "message": str(e)}), 400

    problem = _problem_from_row(row)
    log_queue: queue.Queue = queue.Queue()

    problem_id = row["id"]
    solving_user_id = get_current_user_id()

    def run_solver():
        try:
            orchestrator = StreamingOrchestrator(config, log_queue)
            solve_result = orchestrator.solve(problem)
            result_data = _build_result_data(solve_result)
            # Save solution to database
            try:
                with app.app_context():
                    sdb = get_db()
                    sdb.execute(
                        """INSERT INTO solutions (problem_id, user_id, solved, score, attempts, code, history_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            problem_id,
                            solving_user_id,
                            1 if solve_result.solved else 0,
                            result_data["score"],
                            solve_result.attempts,
                            result_data["code"],
                            json.dumps(result_data["history"]),
                        ),
                    )
                    sdb.commit()
            except Exception:
                pass  # Don't fail the solve if saving fails
            log_queue.put({"type": "done", "result": result_data})
        except Exception as e:
            log_queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"})

    thread = threading.Thread(target=run_solver, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                msg = log_queue.get(timeout=300)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Solver timed out'})}\n\n"
                break
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/problem/<slug>/delete", methods=["POST"])
@login_required
def delete_problem(slug: str):
    db = get_db()
    row = db.execute("SELECT user_id FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is not None and row["user_id"] == get_current_user_id():
        db.execute("DELETE FROM problems WHERE slug = ?", (slug,))
        db.commit()
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Solution History
# ---------------------------------------------------------------------------

@app.route("/problem/<slug>/history")
def problem_history(slug: str):
    db = get_db()
    row = db.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return "Problem not found", 404

    user_id = get_current_user_id()
    is_owner = user_id is not None and user_id == row["user_id"]
    if not row["is_public"] and not is_owner:
        return "Problem not found", 404

    solutions = db.execute(
        "SELECT * FROM solutions WHERE problem_id = ? ORDER BY created_at DESC",
        (row["id"],),
    ).fetchall()

    return render_template(
        "history.html",
        problem_title=row["title"],
        slug=slug,
        solutions=solutions,
        is_owner=is_owner,
    )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@app.route("/leaderboard")
def leaderboard():
    db = get_db()
    stats = db.execute("""
        SELECT
            u.id,
            u.email,
            COUNT(s.id) as total_solves,
            SUM(CASE WHEN s.solved = 1 THEN 1 ELSE 0 END) as successful_solves,
            ROUND(AVG(s.attempts), 1) as avg_attempts,
            ROUND(AVG(s.score) * 100, 1) as avg_score,
            (SELECT COUNT(*) FROM problems WHERE user_id = u.id) as problems_created
        FROM users u
        LEFT JOIN solutions s ON s.user_id = u.id
        GROUP BY u.id
        HAVING total_solves > 0
        ORDER BY successful_solves DESC, avg_score DESC
    """).fetchall()
    return render_template("leaderboard.html", stats=stats)


# ---------------------------------------------------------------------------
# Export / Import Problems
# ---------------------------------------------------------------------------

@app.route("/problem/<slug>/export")
def export_problem(slug: str):
    db = get_db()
    row = db.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return "Problem not found", 404
    user_id = get_current_user_id()
    is_owner = user_id is not None and user_id == row["user_id"]
    if not row["is_public"] and not is_owner:
        return "Problem not found", 404

    data = {
        "title": row["title"],
        "description": row["description"],
        "function_signature": row["function_signature"],
        "code_template": row["code_template"] if "code_template" in row.keys() else "",
        "constraints": row["constraints"],
        "test_cases": json.loads(row["test_cases_json"]),
        "tags": row["tags"],
        "difficulty": row["difficulty"],
    }
    return Response(
        json.dumps(data, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{slug}.json"'},
    )


@app.route("/import", methods=["GET", "POST"])
@login_required
def import_problem():
    if request.method == "GET":
        return render_template("import.html")

    file = request.files.get("file")
    if not file or not file.filename:
        return render_template("import.html", error="Please select a JSON file.")

    try:
        data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return render_template("import.html", error="Invalid JSON file.")

    title = data.get("title", "").strip()
    description = data.get("description", "").strip()
    if not title or not description:
        return render_template("import.html", error="JSON must contain 'title' and 'description'.")

    slug = slugify(title)
    db = get_db()

    # Make slug unique if needed
    base_slug = slug
    counter = 1
    while db.execute("SELECT 1 FROM problems WHERE slug = ?", (slug,)).fetchone():
        slug = f"{base_slug}-{counter}"
        counter += 1

    tags = ",".join(t.strip().lower() for t in data.get("tags", "").split(",") if t.strip()) if data.get("tags") else ""
    difficulty = data.get("difficulty", "medium")
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    db.execute(
        """INSERT INTO problems (slug, user_id, title, description, function_signature, code_template, constraints, test_cases_json, tags, difficulty, is_public)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (
            slug, get_current_user_id(), title, description,
            data.get("function_signature", ""), data.get("code_template", ""),
            data.get("constraints", ""),
            json.dumps(data.get("test_cases", [])), tags, difficulty,
        ),
    )
    db.commit()
    return redirect(url_for("problem_detail", slug=slug))


# ---------------------------------------------------------------------------
# Solution Explanation
# ---------------------------------------------------------------------------

@app.route("/problem/<slug>/explain", methods=["POST"])
@login_required
def explain_solution(slug: str):
    db = get_db()
    row = db.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return jsonify({"error": "Problem not found"}), 404

    data = request.get_json(silent=True) or {}
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        config = _config_for_current_user()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    client = config.create_openai_client()
    try:
        response = client.chat.completions.create(
            model=config.generator_model,
            messages=[
                {"role": "system", "content": (
                    "You are a coding tutor. Explain the given solution clearly and concisely. "
                    "Include: 1) The approach/algorithm used, 2) Time complexity, 3) Space complexity, "
                    "4) Key insights. Use markdown formatting."
                )},
                {"role": "user", "content": (
                    f"Problem: {row['title']}\n\n"
                    f"Description: {row['description']}\n\n"
                    f"Solution code:\n```python\n{code}\n```\n\n"
                    "Explain this solution."
                )},
            ],
            temperature=0.3,
        )
        explanation = response.choices[0].message.content or ""
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 500

    return jsonify({"explanation": explanation})


# ---------------------------------------------------------------------------
# Custom Test Case Runner
# ---------------------------------------------------------------------------

@app.route("/problem/<slug>/run", methods=["POST"])
@login_required
def run_test_case(slug: str):
    db = get_db()
    row = db.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return jsonify({"error": "Problem not found"}), 404

    data = request.get_json(silent=True) or {}
    code = data.get("code", "").strip()
    tc_input = data.get("input", "").strip()
    tc_expected = data.get("expected_output", "").strip()

    if not code:
        return jsonify({"error": "No code provided"}), 400

    problem = _problem_from_row(row)
    test_case = TestCase(input=tc_input, expected_output=tc_expected, description="Custom test")

    try:
        config = _config_for_current_user()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    from alethia.executor_factory import create_executor
    executor = create_executor(config)
    solution = Solution(code=code, attempt=0)

    try:
        exec_result = executor.execute(solution.code, test_case.input, problem.function_signature or "")
        actual = exec_result.stdout.strip()
        passed = actual == tc_expected
        return jsonify({
            "passed": passed,
            "actual": actual,
            "expected": tc_expected,
            "stderr": exec_result.stderr,
            "timed_out": exec_result.timed_out,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Problem Sets / Playlists
# ---------------------------------------------------------------------------

@app.route("/sets")
@login_required
def list_sets():
    db = get_db()
    user_id = get_current_user_id()
    sets = db.execute(
        """SELECT ps.*, COUNT(psi.problem_id) as problem_count
           FROM problem_sets ps
           LEFT JOIN problem_set_items psi ON psi.set_id = ps.id
           WHERE ps.user_id = ?
           GROUP BY ps.id
           ORDER BY ps.created_at DESC""",
        (user_id,),
    ).fetchall()
    return render_template("sets.html", sets=sets)


@app.route("/sets/create", methods=["POST"])
@login_required
def create_set():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    if not name:
        return redirect(url_for("list_sets"))
    db = get_db()
    cursor = db.execute(
        "INSERT INTO problem_sets (user_id, name, description) VALUES (?, ?, ?)",
        (get_current_user_id(), name, description),
    )
    db.commit()
    return redirect(url_for("set_detail", set_id=cursor.lastrowid))


@app.route("/sets/<int:set_id>")
@login_required
def set_detail(set_id: int):
    db = get_db()
    pset = db.execute("SELECT * FROM problem_sets WHERE id = ? AND user_id = ?",
                       (set_id, get_current_user_id())).fetchone()
    if pset is None:
        return "Set not found", 404

    items = db.execute(
        """SELECT p.slug, p.title, p.function_signature, p.difficulty, p.tags, psi.position
           FROM problem_set_items psi
           JOIN problems p ON p.id = psi.problem_id
           WHERE psi.set_id = ?
           ORDER BY psi.position""",
        (set_id,),
    ).fetchall()

    # Get all user's problems for the add dropdown (exclude already-added ones)
    added_slugs = {item["slug"] for item in items}
    all_problems = db.execute(
        "SELECT id, slug, title FROM problems WHERE is_public = 1 OR user_id = ? ORDER BY title",
        (get_current_user_id(),),
    ).fetchall()
    available_problems = [p for p in all_problems if p["slug"] not in added_slugs]

    return render_template("set_detail.html", pset=pset, items=items, available_problems=available_problems)


@app.route("/sets/<int:set_id>/add", methods=["POST"])
@login_required
def add_to_set(set_id: int):
    db = get_db()
    pset = db.execute("SELECT * FROM problem_sets WHERE id = ? AND user_id = ?",
                       (set_id, get_current_user_id())).fetchone()
    if pset is None:
        return "Set not found", 404

    problem_id = request.form.get("problem_id", type=int)
    if problem_id:
        max_pos = db.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM problem_set_items WHERE set_id = ?",
            (set_id,),
        ).fetchone()["next_pos"]
        try:
            db.execute(
                "INSERT INTO problem_set_items (set_id, problem_id, position) VALUES (?, ?, ?)",
                (set_id, problem_id, max_pos),
            )
            db.commit()
        except Exception:
            pass  # Already exists
    return redirect(url_for("set_detail", set_id=set_id))


@app.route("/sets/<int:set_id>/remove/<int:problem_id>", methods=["POST"])
@login_required
def remove_from_set(set_id: int, problem_id: int):
    db = get_db()
    pset = db.execute("SELECT * FROM problem_sets WHERE id = ? AND user_id = ?",
                       (set_id, get_current_user_id())).fetchone()
    if pset is None:
        return "Set not found", 404
    db.execute("DELETE FROM problem_set_items WHERE set_id = ? AND problem_id = ?", (set_id, problem_id))
    db.commit()
    return redirect(url_for("set_detail", set_id=set_id))


@app.route("/sets/<int:set_id>/delete", methods=["POST"])
@login_required
def delete_set(set_id: int):
    db = get_db()
    db.execute("DELETE FROM problem_sets WHERE id = ? AND user_id = ?", (set_id, get_current_user_id()))
    db.commit()
    return redirect(url_for("list_sets"))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    user_id = get_current_user_id()

    # User stats
    total_solves = db.execute("SELECT COUNT(*) as cnt FROM solutions WHERE user_id = ?", (user_id,)).fetchone()["cnt"]
    successful_solves = db.execute("SELECT COUNT(*) as cnt FROM solutions WHERE user_id = ? AND solved = 1", (user_id,)).fetchone()["cnt"]
    problems_created = db.execute("SELECT COUNT(*) as cnt FROM problems WHERE user_id = ?", (user_id,)).fetchone()["cnt"]

    # Recent solve data for charts (last 30 solves)
    recent = db.execute(
        "SELECT solved, score, attempts, created_at FROM solutions WHERE user_id = ? ORDER BY created_at DESC LIMIT 30",
        (user_id,),
    ).fetchall()
    chart_data = [{"solved": r["solved"], "score": r["score"], "attempts": r["attempts"], "date": r["created_at"]} for r in reversed(recent)]

    # Score distribution
    score_dist = db.execute(
        """SELECT
            CASE
                WHEN score < 0.2 THEN '0-20%'
                WHEN score < 0.4 THEN '20-40%'
                WHEN score < 0.6 THEN '40-60%'
                WHEN score < 0.8 THEN '60-80%'
                ELSE '80-100%'
            END as bucket,
            COUNT(*) as cnt
           FROM solutions WHERE user_id = ?
           GROUP BY bucket ORDER BY bucket""",
        (user_id,),
    ).fetchall()
    score_dist_data = [{"bucket": r["bucket"], "count": r["cnt"]} for r in score_dist]

    return render_template(
        "dashboard.html",
        total_solves=total_solves,
        successful_solves=successful_solves,
        problems_created=problems_created,
        chart_data=json.dumps(chart_data),
        score_dist_data=json.dumps(score_dist_data),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
