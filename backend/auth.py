"""
Minimal API-key auth for the IRIS backend.

Design choice: a single shared API key via header, not full OAuth/JWT.
This is intentionally simple — enough to stop the API being wide open
to the internet, without the complexity of a user/session system that
this project doesn't need yet. See ARCHITECTURE.md for the reasoning
and what a real multi-tenant deployment would need instead.

Set IRIS_API_KEY in the environment (or backend/.env) to enable auth.
If IRIS_API_KEY is unset, auth is disabled — useful for local dev,
but server.py prints a loud warning so it's never silently insecure.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status

API_KEY = os.getenv("IRIS_API_KEY", "").strip()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency — raise 401 unless the caller sent a matching key.

    No-ops (auth disabled) if IRIS_API_KEY isn't set in the environment.
    """
    if not API_KEY:
        return  # auth disabled — local/dev mode
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )
