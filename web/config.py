"""Web Portal Configuration for Prime Math."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment configuration (.env)
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class WebConfig:
    port: int = 8790
    host: str = "0.0.0.0"
    base_url: str = "http://127.0.0.1:8790"
    
    # Storage paths
    data_dir: Path = PROJECT_ROOT / "data" / "web"
    db_path: Path = PROJECT_ROOT / "data" / "web" / "portal.db"
    logs_dir: Path = PROJECT_ROOT / "data" / "web" / "logs"
    proofs_dir: Path = PROJECT_ROOT / "data" / "web" / "proofs"
    
    # Auth & Supabase
    auth_provider: str = "supabase"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    admin_invitation_code: str = "PRIME_MATH_ADMIN_2026"
    default_invite_code: str = "PRIME_MATH_MATHEMATICIAN_2026"
    admin_emails: list[str] = field(default_factory=list)
    session_secret: str = "prime-math-secret-session-key-2026"
    
    # Hardware & Queue worker limits (APU protection)
    max_concurrent_workers: int = 1
    solver_timeout_seconds: float = 3600.0
    solver_max_iterations: int = 8
    enable_thinking: bool = True
    
    @classmethod
    def from_env(cls) -> WebConfig:
        port = int(os.getenv("WEB_PORTAL_PORT", "8790"))
        host = os.getenv("WEB_PORTAL_HOST", "0.0.0.0")
        base_url = os.getenv("WEB_PORTAL_BASE_URL", f"http://127.0.0.1:{port}")
        
        data_dir = Path(os.getenv("WEB_DATA_DIR", str(PROJECT_ROOT / "data" / "web")))
        db_path = data_dir / "portal.db"
        logs_dir = data_dir / "logs"
        proofs_dir = data_dir / "proofs"
        
        admin_emails_raw = os.getenv("ADMIN_EMAILS", "")
        admin_emails = [
            e.strip().lower() for e in admin_emails_raw.split(",") if e.strip()
        ]
        
        return cls(
            port=port,
            host=host,
            base_url=base_url,
            data_dir=data_dir,
            db_path=db_path,
            logs_dir=logs_dir,
            proofs_dir=proofs_dir,
            auth_provider=os.getenv("AUTH_PROVIDER", "supabase"),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            admin_invitation_code=os.getenv("ADMIN_INVITATION_CODE", "PRIME_MATH_ADMIN_2026"),
            default_invite_code=os.getenv("PRIME_MATH_INVITE_CODE", "PRIME_MATH_MATHEMATICIAN_2026"),
            admin_emails=admin_emails,
            session_secret=os.getenv("SESSION_SECRET", "prime-math-secret-session-key-2026"),
            max_concurrent_workers=int(os.getenv("WEB_QUEUE_MAX_WORKERS", "1")),
            solver_timeout_seconds=float(os.getenv("TCS_SOLVER_TIMEOUT_SECONDS", "3600.0")),
            solver_max_iterations=int(os.getenv("TCS_SOLVER_MAX_ITERATIONS", "8")),
            enable_thinking=os.getenv("TCS_SOLVER_ENABLE_THINKING", "true").lower() == "true",
        )
