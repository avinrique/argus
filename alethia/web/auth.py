"""Simple session-based auth helpers for Argus web app."""

from __future__ import annotations

from functools import wraps

from flask import redirect, request, session, url_for


def get_current_user_id() -> int | None:
    """Return the logged-in user's ID, or None."""
    return session.get("user_id")


def login_required(f):
    """Decorator that redirects to /login if the user is not authenticated."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if get_current_user_id() is None:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)

    return decorated
