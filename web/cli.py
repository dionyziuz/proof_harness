"""Command-line interface for managing the Prime Math Research Web Portal."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.config import WebConfig
from web.database import PortalDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description="Prime Math Research Web Portal CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Server runner
    run_parser = subparsers.add_parser("run", help="Launch the Prime Math web portal server")
    run_parser.add_argument("--port", type=int, default=8790, help="Port to listen on (default: 8790)")
    run_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    run_parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload for development")

    serve_parser = subparsers.add_parser("serve", help="Launch the Prime Math web portal server (alias for run)")
    serve_parser.add_argument("--port", type=int, default=8790, help="Port to listen on (default: 8790)")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    serve_parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload for development")

    # Lock operator priority
    lock_parser = subparsers.add_parser("lock", help="Lock the web queue to grant 100% priority to local operator jobs")
    lock_parser.add_argument("--reason", type=str, default="Local operator benchmark evaluation active on APU", help="Reason for lock")

    # Unlock operator priority
    unlock_parser = subparsers.add_parser("unlock", help="Unlock the web queue to resume processing user submissions")

    # Queue status
    subparsers.add_parser("status", help="Inspect live queue metrics and operator lock state")

    # Invitation key generator
    invite_parser = subparsers.add_parser("invite", help="Generate a new invitation key for a mathematician")
    invite_parser.add_argument("--notes", type=str, default="Invited mathematician", help="Optional notes on recipient")

    args = parser.parse_args()
    config = WebConfig.from_env()
    db = PortalDatabase(config.db_path)
    db.sync_admin_users(config.admin_emails)

    if args.command in ("run", "serve"):
        import uvicorn
        print(f"Starting Prime Math Research Portal on http://{args.host}:{args.port}")
        uvicorn.run("web.server:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "lock":
        db.set_operator_lock(True, args.reason)
        print(f"✓ Operator Priority Lock ACTIVE: '{args.reason}'")
        print("Web submissions will be held in queue; operator tasks have 100% compute precedence.")

    elif args.command == "unlock":
        db.set_operator_lock(False, "Server available for mathematician submissions")
        print("✓ Operator Priority Lock RELEASED.")
        print("Web queue worker will now resume processing queued mathematical problems.")

    elif args.command == "status":
        metrics = db.get_queue_metrics()
        locked, reason = db.is_operator_locked()
        print("================ Prime Math Web Portal Status ================")
        print(f"Operator Lock:     {'LOCKED (HELD)' if locked else 'UNLOCKED (ACTIVE)'}")
        print(f"Lock Reason:       {reason}")
        print(f"Workers:           {metrics['active_workers']} active / {metrics['max_workers']} max concurrency")
        print(f"Queued Jobs:       {metrics['queued_count']}")
        print(f"Held Jobs:         {metrics['held_count']}")
        print(f"Completed Jobs:    {metrics['completed_count']}")
        print("==============================================================")

    elif args.command == "invite":
        code = f"MATH_{uuid.uuid4().hex[:8].upper()}"
        db.create_invitation(code=code, notes=args.notes, created_by="cli-admin")
        print("================ New Mathematician Invitation ================")
        print(f"Invitation Key:    {code}")
        print(f"Recipient Notes:   {args.notes}")
        print("==============================================================")


if __name__ == "__main__":
    main()
