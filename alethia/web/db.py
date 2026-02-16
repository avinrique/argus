"""SQLite database helpers for Argus web app."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import g, current_app

DB_PATH = Path(__file__).resolve().parent.parent.parent / "instance" / "argus.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    llm_provider TEXT NOT NULL DEFAULT 'openai',
    openai_api_key TEXT NOT NULL DEFAULT '',
    ollama_base_url TEXT NOT NULL DEFAULT 'http://localhost:11434/v1',
    preferred_model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS problems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    function_signature TEXT NOT NULL DEFAULT '',
    code_template TEXT NOT NULL DEFAULT '',
    images_json TEXT NOT NULL DEFAULT '[]',
    constraints TEXT NOT NULL DEFAULT '',
    test_cases_json TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '',
    difficulty TEXT NOT NULL DEFAULT 'medium',
    is_public INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_id INTEGER NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id),
    solved INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    code TEXT NOT NULL DEFAULT '',
    history_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS problem_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS problem_set_items (
    set_id INTEGER NOT NULL REFERENCES problem_sets(id) ON DELETE CASCADE,
    problem_id INTEGER NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (set_id, problem_id)
);
"""

# Migrations for existing databases that don't have the new columns/tables
MIGRATIONS = [
    "ALTER TABLE problems ADD COLUMN tags TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE problems ADD COLUMN difficulty TEXT NOT NULL DEFAULT 'medium'",
    "ALTER TABLE problems ADD COLUMN is_public INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE problems ADD COLUMN code_template TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE problems ADD COLUMN images_json TEXT NOT NULL DEFAULT '[]'",
]


def get_db() -> sqlite3.Connection:
    """Return a per-request database connection stored on Flask *g*."""
    if "db" not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(exc=None):
    """Close the database connection at the end of a request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables if they don't exist, then run migrations for new columns."""
    db = get_db()
    db.executescript(SCHEMA)
    db.commit()
    # Run column-add migrations (safe to fail if column already exists)
    for sql in MIGRATIONS:
        try:
            db.execute(sql)
            db.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
