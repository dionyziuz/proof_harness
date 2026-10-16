"""SQLite Database layer for Prime Math Web Portal."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from web.config import WebConfig


class PortalDatabase:
    """Thread-safe SQLite database manager for web queue, settings, and invitations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                problem_id TEXT NOT NULL,
                title TEXT NOT NULL,
                field TEXT NOT NULL,
                user_email TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                priority INTEGER DEFAULT 0,
                problem_json TEXT NOT NULL,
                task_prompt TEXT NOT NULL,
                iterations INTEGER DEFAULT 0,
                max_iterations INTEGER DEFAULT 8,
                duration_s REAL DEFAULT 0.0,
                validity REAL,
                candidate_proof TEXT,
                log_path TEXT,
                error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status_prio ON jobs (status, priority DESC, created_at ASC);
            CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs (user_email);

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS invitations (
                code TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                created_by TEXT DEFAULT 'admin',
                used_by TEXT,
                used_at TEXT,
                is_active INTEGER DEFAULT 1,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                role TEXT DEFAULT 'mathematician',
                invitation_code TEXT,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            );
            """)

            # Default settings
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM settings WHERE key = 'operator_lock'")
            if not cur.fetchone():
                conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('operator_lock', 'false')")
                conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('operator_lock_reason', 'Server available for mathematician submissions')")
                conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('max_workers', '1')")

            # Seed initial master invitations
            default_invites = [
                ("PRIME_MATH_MATHEMATICIAN_2026", "Master invitation key for academic mathematicians", "system"),
                ("PRIME_MATH_ADMIN_2026", "Master administrator invitation key", "system"),
            ]
            now = datetime.now(timezone.utc).isoformat()
            for code, notes, creator in default_invites:
                cur.execute("SELECT 1 FROM invitations WHERE code = ?", (code,))
                if not cur.fetchone():
                    conn.execute(
                        "INSERT INTO invitations (code, created_at, created_by, is_active, notes) VALUES (?, ?, ?, 1, ?)",
                        (code, now, creator, notes),
                    )

            # Cleanup any legacy keys
            conn.execute("DELETE FROM invitations WHERE code = 'FINANCE_ADMIN_2026'")

    def sync_admin_users(self, admin_emails: List[str]) -> None:
        """Syncs authorized administrator emails configured from .env."""
        if not admin_emails:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            for adm_email in admin_emails:
                adm_email = adm_email.strip().lower()
                if adm_email:
                    conn.execute(
                        "INSERT INTO users (email, role, created_at) VALUES (?, 'admin', ?) "
                        "ON CONFLICT(email) DO UPDATE SET role = 'admin'",
                        (adm_email, now),
                    )

    # ── Settings ─────────────────────────────────────────────────────────────

    def get_setting(self, key: str, default: str = "") -> str:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            return str(row["value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))

    def is_operator_locked(self) -> Tuple[bool, str]:
        locked_val = self.get_setting("operator_lock", "false").lower() == "true"
        reason = self.get_setting("operator_lock_reason", "Operator priority evaluation active")
        return locked_val, reason

    def set_operator_lock(self, locked: bool, reason: Optional[str] = None) -> None:
        self.set_setting("operator_lock", "true" if locked else "false")
        if reason:
            self.set_setting("operator_lock_reason", reason)

    # ── Job Management ───────────────────────────────────────────────────────

    def create_job(self, job_dict: Dict[str, Any]) -> str:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, problem_id, title, field, user_email, status, created_at,
                    started_at, completed_at, priority, problem_json, task_prompt,
                    iterations, max_iterations, duration_s, validity, candidate_proof,
                    log_path, error
                ) VALUES (
                    :id, :problem_id, :title, :field, :user_email, :status, :created_at,
                    :started_at, :completed_at, :priority, :problem_json, :task_prompt,
                    :iterations, :max_iterations, :duration_s, :validity, :candidate_proof,
                    :log_path, :error
                )
                """,
                {
                    "id": job_dict["id"],
                    "problem_id": job_dict["problem_id"],
                    "title": job_dict["title"],
                    "field": job_dict.get("field", "Theoretical Mathematics"),
                    "user_email": job_dict["user_email"],
                    "status": job_dict["status"],
                    "created_at": job_dict["created_at"],
                    "started_at": job_dict.get("started_at"),
                    "completed_at": job_dict.get("completed_at"),
                    "priority": job_dict.get("priority", 0),
                    "problem_json": json.dumps(job_dict["problem_json"]),
                    "task_prompt": job_dict["task_prompt"],
                    "iterations": job_dict.get("iterations", 0),
                    "max_iterations": job_dict.get("max_iterations", 8),
                    "duration_s": job_dict.get("duration_s", 0.0),
                    "validity": job_dict.get("validity"),
                    "candidate_proof": job_dict.get("candidate_proof"),
                    "log_path": job_dict.get("log_path"),
                    "error": job_dict.get("error"),
                },
            )
        return job_dict["id"]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["problem_json"] = json.loads(d["problem_json"])
            return d

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        fields = []
        vals = []
        for k, v in updates.items():
            fields.append(f"{k} = ?")
            if k == "problem_json" and isinstance(v, (dict, list)):
                vals.append(json.dumps(v))
            else:
                vals.append(v)
        vals.append(job_id)

        with self._lock, self._get_conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", vals)

    def reset_orphaned_jobs(self) -> int:
        """Resets any jobs left in 'running' state after server crashes or restarts back to 'queued'."""
        with self._lock, self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'"
            )
            return cur.rowcount

    def list_jobs(self, user_email: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            if user_email:
                cur = conn.execute(
                    "SELECT id, problem_id, title, field, user_email, status, created_at, started_at, completed_at, "
                    "iterations, max_iterations, duration_s, validity, (candidate_proof IS NOT NULL) AS has_proof "
                    "FROM jobs WHERE user_email = ? ORDER BY created_at DESC LIMIT ?",
                    (user_email, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT id, problem_id, title, field, user_email, status, created_at, started_at, completed_at, "
                    "iterations, max_iterations, duration_s, validity, (candidate_proof IS NOT NULL) AS has_proof "
                    "FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cur.fetchall()]

    def get_queue_position(self, job_id: str) -> Optional[int]:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("SELECT status, created_at, priority FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if not row or row["status"] not in ("queued", "held_operator_priority"):
                return None
            cur_pos = conn.execute(
                "SELECT COUNT(*) as pos FROM jobs WHERE status IN ('queued', 'held_operator_priority') "
                "AND (priority > ? OR (priority = ? AND created_at < ?))",
                (row["priority"], row["priority"], row["created_at"]),
            )
            pos_row = cur_pos.fetchone()
            return int(pos_row["pos"]) + 1 if pos_row else 1

    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Pulls the next executable queued job (skips held jobs if locked)."""
        locked, _ = self.is_operator_locked()
        with self._lock, self._get_conn() as conn:
            if locked:
                # If operator locked, only priority >= 10 (internal operator jobs) can be dequeued
                cur = conn.execute(
                    "SELECT * FROM jobs WHERE status = 'queued' AND priority >= 10 ORDER BY priority DESC, created_at ASC LIMIT 1"
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM jobs WHERE status IN ('queued', 'held_operator_priority') ORDER BY priority DESC, created_at ASC LIMIT 1"
                )
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["problem_json"] = json.loads(d["problem_json"])
            return d

    def get_queue_metrics(self) -> Dict[str, Any]:
        with self._lock, self._get_conn() as conn:
            counts = {
                "queued": 0,
                "held": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
            }
            cur = conn.execute("SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status")
            for row in cur.fetchall():
                st = row["status"]
                if st == "held_operator_priority":
                    counts["held"] += row["cnt"]
                elif st in counts:
                    counts[st] += row["cnt"]

            locked, reason = self.is_operator_locked()
            max_workers = int(self.get_setting("max_workers", "1"))
            return {
                "operator_lock": locked,
                "operator_lock_reason": reason,
                "max_workers": max_workers,
                "active_workers": counts["running"],
                "queued_count": counts["queued"],
                "held_count": counts["held"],
                "running_count": counts["running"],
                "completed_count": counts["completed"],
            }

    # ── Invitations & Users ──────────────────────────────────────────────────

    def verify_invitation(self, code: str, email: str) -> Tuple[bool, str]:
        code = code.strip().upper()
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM invitations WHERE code = ? AND is_active = 1", (code,))
            row = cur.fetchone()
            if not row:
                return False, "Invalid or expired invitation key."
            
            # Record usage
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE invitations SET used_by = ?, used_at = ? WHERE code = ?",
                (email.lower(), now, code),
            )
            role = "admin" if "ADMIN" in code else "mathematician"
            conn.execute(
                "INSERT OR REPLACE INTO users (email, role, invitation_code, created_at, last_login_at) "
                "VALUES (?, ?, ?, COALESCE((SELECT created_at FROM users WHERE email = ?), ?), ?)",
                (email.lower(), role, code, email.lower(), now, now),
            )
            return True, role

    def create_invitation(self, code: str, notes: str = "", created_by: str = "admin") -> str:
        code = code.strip().upper()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO invitations (code, created_at, created_by, is_active, notes) VALUES (?, ?, ?, 1, ?)",
                (code, now, created_by, notes),
            )
        return code

    def list_invitations(self) -> List[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM invitations ORDER BY created_at DESC LIMIT 100")
            return [dict(row) for row in cur.fetchall()]

    def get_user(self, email: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
            row = cur.fetchone()
            return dict(row) if row else None
