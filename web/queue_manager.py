"""Persistent Queue Manager and Worker Service for Prime Math Prover Jobs."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from web.config import WebConfig
from web.database import PortalDatabase

logger = logging.getLogger("prime_math.web.queue")


class ProverQueueWorker:
    """Manages asynchronous proving execution from the database queue."""

    def __init__(self, config: WebConfig, db: PortalDatabase):
        self.config = config
        self.db = db
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._active_jobs: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

        # Ensure directories exist
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.proofs_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Starts the queue monitoring background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        orphans = self.db.reset_orphaned_jobs()
        if orphans:
            logger.info(f"[Queue Worker] Reset {orphans} orphaned 'running' job(s) back to queued.")
        self._worker_thread = threading.Thread(target=self._run_loop, name="PrimeMathQueueWorker", daemon=True)
        self._worker_thread.start()
        logger.info("[Queue Worker] Background queue worker started.")

    def stop(self) -> None:
        """Signals the worker to stop gracefully."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            logger.info("[Queue Worker] Background queue worker stopped.")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                locked, reason = self.db.is_operator_locked()

                # Clean up finished active jobs from tracking dict
                with self._lock:
                    finished = [jid for jid, th in self._active_jobs.items() if not th.is_alive()]
                    for jid in finished:
                        del self._active_jobs[jid]

                    active_count = len(self._active_jobs)
                    max_workers = int(self.db.get_setting("max_workers", str(self.config.max_concurrent_workers)))

                if locked:
                    # If operator locked, held jobs remain in queue
                    time.sleep(3.0)
                    continue

                if active_count >= max_workers:
                    time.sleep(2.0)
                    continue

                # Dequeue next job
                job = self.db.get_next_job()
                if not job:
                    time.sleep(2.0)
                    continue

                job_id = job["id"]
                now_str = datetime.now(timezone.utc).isoformat()
                self.db.update_job(job_id, {"status": "running", "started_at": now_str})

                job_thread = threading.Thread(
                    target=self._execute_job,
                    args=(job,),
                    name=f"ProverJob-{job_id}",
                    daemon=True,
                )
                with self._lock:
                    self._active_jobs[job_id] = job_thread
                job_thread.start()

            except Exception as e:
                logger.error(f"[Queue Worker Loop Error]: {e}", exc_info=True)
                time.sleep(5.0)

    def _execute_job(self, job: Dict[str, Any]) -> None:
        job_id = job["id"]
        task_id = job["problem_id"]
        task_prompt = job["task_prompt"]
        max_iters = job.get("max_iterations") or self.config.solver_max_iterations
        log_path = self.config.logs_dir / f"{job_id}.log"
        self.db.update_job(job_id, {"log_path": str(log_path)})

        start_time = time.perf_counter()
        logger.info(f"[Job {job_id}] Starting solve for '{job['title']}' (PID: {task_id})...")

        try:
            # Import heavy prover stack lazily
            from core_engine.llm_client import TunnelThinkingClient
            from core_engine.memory import CoALAMemoryManager
            from core_engine.prover import MathProver
            from core_engine.prover_kernel import TCSCompliantKernel

            client = TunnelThinkingClient()
            kernel = TCSCompliantKernel()
            memory = CoALAMemoryManager()

            solver = MathProver(
                client=client,
                kernel=kernel,
                memory=memory,
                default_max_iterations=max_iters,
                log_path=log_path,
                enable_thinking=self.config.enable_thinking,
                enable_literature_search=bool(job.get("problem_json", {}).get("literature_search", False)),
            )

            # Step listener updates DB progress in real time
            def on_step(event: str, data: Any) -> None:
                if event == "turn_start":
                    self.db.update_job(job_id, {"iterations": int(data)})
                elif event == "audit_result":
                    val = data.get("validity")
                    if val is not None:
                        self.db.update_job(job_id, {"validity": float(val)})

            solve_res = solver.solve(
                task_id=task_id,
                task_prompt=task_prompt,
                max_iterations=max_iters,
                max_time_seconds=self.config.solver_timeout_seconds,
                on_step=on_step,
            )

            duration = time.perf_counter() - start_time
            now_str = datetime.now(timezone.utc).isoformat()
            proof = solve_res.candidate_proof or solve_res.final_response or ""

            # Save proof artifact to file
            if proof:
                proof_file = self.config.proofs_dir / f"{job_id}.tex"
                proof_file.write_text(proof, encoding="utf-8")

            status = "completed" if solve_res.verified or len(proof.strip()) > 100 else "failed"
            self.db.update_job(
                job_id,
                {
                    "status": status,
                    "completed_at": now_str,
                    "duration_s": round(duration, 2),
                    "iterations": solve_res.iterations,
                    "candidate_proof": proof,
                    "error": solve_res.error,
                },
            )
            logger.info(f"[Job {job_id}] Finished with status={status} in {duration:.1f}s")

        except Exception as exc:
            duration = time.perf_counter() - start_time
            now_str = datetime.now(timezone.utc).isoformat()
            logger.error(f"[Job {job_id}] Execution error: {exc}", exc_info=True)
            self.db.update_job(
                job_id,
                {
                    "status": "failed",
                    "completed_at": now_str,
                    "duration_s": round(duration, 2),
                    "error": str(exc),
                },
            )
