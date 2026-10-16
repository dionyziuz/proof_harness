"""FastAPI Web Server for Prime Math Research Portal."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from web.auth import AuthManager, get_current_user, require_admin, set_auth_manager
from web.config import WebConfig
from web.database import PortalDatabase
from web.examples import EXAMPLE_PROBLEMS
from web.models import (
    AuthVerifyRequest,
    InvitationCode,
    InvitationCreateRequest,
    JobRecord,
    JobSummary,
    OperatorLockToggle,
    ProblemSubmission,
    QueueSettings,
)
from web.queue_manager import ProverQueueWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prime_math.web.server")

# Initialize app and configuration
config = WebConfig.from_env()
db = PortalDatabase(config.db_path)
auth_manager = AuthManager(config, db)
set_auth_manager(auth_manager)
worker = ProverQueueWorker(config, db)

app = FastAPI(
    title="Prime Math Research Portal",
    description="Mathematician problem submission, queue management, and autonomous theorem proving.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    db.sync_admin_users(config.admin_emails)
    worker.start()
    logger.info(f"Prime Math Portal started on {config.host}:{config.port} (base: {config.base_url})")


@app.on_event("shutdown")
def shutdown_event():
    worker.stop()
    logger.info("Prime Math Portal shutdown cleanly.")


# ── Public & System Endpoints ───────────────────────────────────────────────

@app.get("/api/health")
def health() -> Dict[str, Any]:
    locked, reason = db.is_operator_locked()
    metrics = db.get_queue_metrics()
    return {
        "status": "healthy",
        "service": "Prime Math Research Portal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operator_lock": locked,
        "operator_lock_reason": reason,
        "active_workers": metrics["active_workers"],
        "max_workers": metrics["max_workers"],
        "hardware": "AMD Ryzen AI Max+ 395 (64GB UMA)",
    }


@app.get("/api/examples")
def list_examples() -> List[Dict[str, Any]]:
    """Returns curated example open math problems in valid JSON schema."""
    return EXAMPLE_PROBLEMS


# ── Authentication & Invitations ───────────────────────────────────────────

@app.post("/api/auth/login")
def login(req: AuthVerifyRequest, response: Response) -> Dict[str, Any]:
    """Authenticates via Invitation Key (and optional Supabase JWT)."""
    success, msg, data = auth_manager.authenticate_invitation(
        email=req.email,
        invite_code=req.invite_code,
        password=req.password,
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)

    # Set secure session cookie
    response.set_cookie(
        key="prime_session",
        value=data["token"],
        max_age=14 * 86400,
        httponly=True,
        samesite="lax",
    )
    return {
        "success": True,
        "token": data["token"],
        "email": data["email"],
        "role": data["role"],
        "expires_at": data["expires_at"],
    }


@app.get("/api/auth/me")
def get_me(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "email": user["email"],
        "role": user.get("role", "mathematician"),
    }


@app.post("/api/auth/logout")
def logout(response: Response) -> Dict[str, bool]:
    response.delete_cookie("prime_session")
    return {"success": True}


# ── Problem Submission & Queue API ──────────────────────────────────────────

@app.post("/api/problems/submit", response_model=Dict[str, Any])
def submit_problem(
    sub: ProblemSubmission,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Validates problem JSON and enqueues job with operator precedence handling."""
    now = datetime.now(timezone.utc).isoformat()
    job_id = f"pm_{int(time.time())}_{sub.problem_id[:16]}"
    
    # Check operator priority lock
    locked, reason = db.is_operator_locked()
    initial_status = "held_operator_priority" if locked else "queued"
    
    # Operators (admins) get priority 10, normal users priority 0
    priority = 10 if user.get("role") == "admin" else 0
    
    task_prompt = sub.to_formatted_prompt()
    
    job_record = {
        "id": job_id,
        "problem_id": sub.problem_id,
        "title": sub.title,
        "field": sub.field,
        "user_email": user["email"],
        "status": initial_status,
        "created_at": now,
        "priority": priority,
        "problem_json": sub.model_dump(),
        "task_prompt": task_prompt,
        "iterations": 0,
        "max_iterations": sub.max_iterations or config.solver_max_iterations,
        "duration_s": 0.0,
    }
    
    db.create_job(job_record)
    queue_pos = db.get_queue_position(job_id)
    
    logger.info(f"New job {job_id} submitted by {user['email']}. Status: {initial_status}, Pos: {queue_pos}")
    return {
        "job_id": job_id,
        "problem_id": sub.problem_id,
        "title": sub.title,
        "status": initial_status,
        "queue_position": queue_pos,
        "operator_locked": locked,
        "operator_lock_reason": reason if locked else None,
        "created_at": now,
    }


@app.get("/api/jobs")
def list_jobs(user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """Lists jobs. Admins see all jobs; mathematicians see their submissions."""
    is_admin = user.get("role") == "admin"
    jobs = db.list_jobs(user_email=None if is_admin else user["email"])
    
    # Enrich with queue positions for active jobs
    for j in jobs:
        if j["status"] in ("queued", "held_operator_priority"):
            j["queue_position"] = db.get_queue_position(j["id"])
    return jobs


@app.get("/api/jobs/{job_id}")
def get_job_detail(job_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    # Ensure permission
    if user.get("role") != "admin" and job["user_email"] != user["email"]:
        raise HTTPException(status_code=403, detail="Access denied to this job record.")
        
    job["queue_position"] = db.get_queue_position(job_id)
    return job


@app.get("/api/jobs/{job_id}/proof", response_class=PlainTextResponse)
def get_job_proof(job_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> str:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if user.get("role") != "admin" and job["user_email"] != user["email"]:
        raise HTTPException(status_code=403, detail="Access denied.")
        
    proof = job.get("candidate_proof") or ""
    if not proof:
        proof_path = config.proofs_dir / f"{job_id}.tex"
        if proof_path.is_file():
            proof = proof_path.read_text(encoding="utf-8")
    if not proof:
        raise HTTPException(status_code=404, detail="No formal proof artifact emitted yet.")
    return proof


@app.get("/api/jobs/{job_id}/logs/stream")
async def stream_job_logs(job_id: str, request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """Server-Sent Events (SSE) live log streaming endpoint."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if user.get("role") != "admin" and job["user_email"] != user["email"]:
        raise HTTPException(status_code=403, detail="Access denied.")

    log_path = Path(job.get("log_path") or (config.logs_dir / f"{job_id}.log"))

    async def event_generator() -> AsyncGenerator[str, None]:
        last_pos = 0
        while not await request.is_disconnected():
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
                    fp.seek(last_pos)
                    chunk = fp.read()
                    last_pos = fp.tell()
                    if chunk:
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Check if job completed
            current_job = db.get_job(job_id)
            if current_job and current_job["status"] in ("completed", "failed", "cancelled"):
                yield f"data: {json.dumps({'done': True, 'status': current_job['status']})}\n\n"
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if user.get("role") != "admin" and job["user_email"] != user["email"]:
        raise HTTPException(status_code=403, detail="Access denied.")
    
    if job["status"] in ("queued", "held_operator_priority"):
        db.update_job(job_id, {"status": "cancelled", "completed_at": datetime.now(timezone.utc).isoformat()})
        return {"success": True, "status": "cancelled"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status '{job['status']}'.")


# ── Admin Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/admin/settings", response_model=QueueSettings)
def get_admin_settings(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    return db.get_queue_metrics()


@app.post("/api/admin/lock")
def toggle_operator_lock(
    req: OperatorLockToggle,
    user: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    """Toggles operator lock status so user submissions are held during local operator benchmarks."""
    db.set_operator_lock(req.locked, req.reason)
    logger.info(f"Operator lock set to {req.locked} by {user['email']}. Reason: {req.reason}")
    return {
        "operator_lock": req.locked,
        "operator_lock_reason": req.reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/admin/invitations")
def create_invitation_code(
    req: InvitationCreateRequest,
    user: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    code = f"MATH_{uuid.uuid4().hex[:8].upper()}"
    db.create_invitation(code=code, notes=req.notes or "", created_by=user["email"])
    return {"code": code, "notes": req.notes, "created_at": datetime.now(timezone.utc).isoformat()}


@app.get("/api/admin/invitations")
def list_invitation_codes(user: Dict[str, Any] = Depends(require_admin)) -> List[Dict[str, Any]]:
    return db.list_invitations()


# ── Frontend Static Assets & SPA ───────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Prime Math Research Portal</h1><p>Portal is initializing.</p>")
