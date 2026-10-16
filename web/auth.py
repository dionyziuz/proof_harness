"""Authentication and Invitation Verification layer (Supabase + Invitations)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from web.config import WebConfig
from web.database import PortalDatabase

security = HTTPBearer(auto_error=False)


def _sign_session(data: Dict[str, Any], secret: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(data).encode("utf-8")).decode("utf-8").rstrip("=")
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_session(token: str, secret: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload, sig = parts
        expected_sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        # Add padding back
        padded = payload + "=" * ((4 - len(payload) % 4) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


class AuthManager:
    """Manages Supabase authentication, invitation codes, and session tokens."""

    def __init__(self, config: WebConfig, db: PortalDatabase):
        self.config = config
        self.db = db

    async def verify_supabase_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Validates a Supabase JWT token via Supabase GoTrue endpoint."""
        if not self.config.supabase_url or not self.config.supabase_anon_key:
            return None

        url = f"{self.config.supabase_url.rstrip('/')}/auth/v1/user"
        headers = {
            "apikey": self.config.supabase_anon_key,
            "Authorization": f"Bearer {access_token}",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code == 200:
                    return res.json()
        except Exception:
            pass
        return None

    def authenticate_invitation(
        self,
        email: str,
        invite_code: str,
        password: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Verifies invitation code and issues an authenticated session."""
        email = email.strip().lower()
        invite_code = invite_code.strip().upper()

        if not email or "@" not in email:
            return False, "A valid email address is required.", {}

        # 1. Administrator check (configured via ADMIN_EMAILS in .env)
        admin_emails = [e.strip().lower() for e in self.config.admin_emails]
        is_admin = email in admin_emails

        # 2. Check invitation validity
        valid, role_or_msg = self.db.verify_invitation(invite_code, email)
        if (
            not valid
            and not is_admin
            and invite_code != self.config.default_invite_code.upper()
            and invite_code != self.config.admin_invitation_code.upper()
        ):
            return False, role_or_msg, {}

        role = "admin" if is_admin else ("admin" if role_or_msg == "admin" else "mathematician")

        # 3. Issue signed portal session token (14 days validity)
        exp = int(time.time()) + (14 * 86400)
        session_data = {
            "email": email,
            "role": role,
            "exp": exp,
            "created_at": time.time(),
        }
        token = _sign_session(session_data, self.config.session_secret)
        return True, "Authentication successful", {
            "token": token,
            "email": email,
            "role": role,
            "expires_at": exp,
        }

    def get_current_user_from_request(self, request: Request) -> Optional[Dict[str, Any]]:
        # 1. Check Authorization Bearer header
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        # 2. Check Cookie
        if not token:
            token = request.cookies.get("prime_session", "")

        if not token:
            return None

        return _verify_session(token, self.config.session_secret)


# Global dependency helpers
_auth_manager: Optional[AuthManager] = None


def set_auth_manager(manager: AuthManager) -> None:
    global _auth_manager
    _auth_manager = manager


def get_current_user(request: Request) -> Dict[str, Any]:
    if not _auth_manager:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth manager uninitialized")
    user = _auth_manager.get_current_user_from_request(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid session or invitation key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to access this resource.",
        )
    return user
