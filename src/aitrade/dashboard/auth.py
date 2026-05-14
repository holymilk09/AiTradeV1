"""HTTP Basic auth for the dashboard.

Single shared password; constant-time compared. Any caller without a
valid password gets a 401 with ``WWW-Authenticate: Basic`` so the
browser shows the standard login dialog.

The dashboard is **personal use only** — single-user, single-password.
For multi-tenant production this would need real auth, but it'd also
need a lot more than auth (rate limits, audit log, etc.). We keep the
scope honest.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from aitrade.config import Settings, get_settings

_security = HTTPBasic()


def require_auth(
    credentials: HTTPBasicCredentials = Depends(_security),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency that 401s unless the request supplies the configured password.

    Returns the username on success (so route handlers can log it).
    The username is unchecked — only the password matters.
    """
    expected = settings.aitrade_dashboard_password.get_secret_value()
    if not expected:
        # Misconfiguration on our side, not the client's. 503 is the right code.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard password not configured (set AITRADE_DASHBOARD_PASSWORD).",
        )
    # Constant-time compare to avoid timing leaks.
    if not secrets.compare_digest(credentials.password.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
