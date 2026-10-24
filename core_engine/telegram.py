"""Telegram Notifications Module for Prime Math Benchmarks.

Loads Telegram bot credentials from .env or environment variables and dispatches
structured HTML progress notifications.
"""

from __future__ import annotations

import html
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core_engine.config import resolve_dotenv_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


def load_telegram_credentials(dotenv_path: Optional[Path] = None) -> Tuple[str, str]:
    """Load TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment or active .env."""
    target_env = resolve_dotenv_path(dotenv_path)
    if target_env.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=target_env, override=bool(dotenv_path is not None))
        except ImportError:
            pass

    env_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    env_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if env_token and env_chat_id:
        return env_token, env_chat_id

    raise ValueError(
        "Telegram credentials not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env."
    )


def send_telegram_message(bot_token: str, chat_id: str, html_text: str, retries: int = 3) -> bool:
    """Send an HTML-formatted message to Telegram with retry logic."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            if attempt == retries:
                print(f"[telegram] Failed to send message after {retries} attempts: {e}", file=sys.stderr)
                return False
            time.sleep(2 * attempt)
    return False


def format_task_report(entry: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> str:
    """Format a single task result report in HTML for Telegram."""
    task_idx = entry.get("task_index", "?")
    task_id = entry.get("task_id", "unknown")
    dur_s = entry.get("duration_s", 0.0)
    iters = entry.get("iterations", "?")
    p_chars = entry.get("candidate_chars", 0)

    # Detect benchmark domain (Physics vs TCS-Bench)
    str_tid = str(task_id).lower()
    is_physics = (
        "gr-qc" in str_tid
        or "2510.24491" in str_tid
        or "physics" in str_tid
        or (metrics and metrics.get("target_total_problems") == 2)
    )
    bench_name = "Physics Benchmark" if is_physics else "TCS-Bench"
    bench_icon = "⚛️" if is_physics else "🎯"

    # TCS Judge evaluation
    val = entry.get("judge_b_validity")
    if val is None:
        val = entry.get("tcs_validity", 0.0)
    if val is None:
        val = 0.0
    pass_tcs = bool(entry.get("pass_b", False) or (val >= 0.50))
    verdict_badge = "PASS ✅" if pass_tcs else "FAIL ❌"

    if pass_tcs:
        overall_badge = "🏆 <b>PASS ✅ (TCS Judge Valid ≥0.50)</b>"
    elif val is not None and val >= 0.40:
        overall_badge = "⚠️ <b>PARTIAL (Directionally Sound Reduction ≥0.40)</b>"
    else:
        overall_badge = "❌ <b>FAIL ❌ (Rejected by TCS Judge)</b>"

    target_disp = metrics.get("target_total_problems", 300) if metrics else 300
    lines = [
        f"{bench_icon} <b>{bench_name} Problem [{task_idx}/{target_disp}] Completed</b>",
        f"<b>ID:</b> <code>{html.escape(str(task_id))}</code>",
        f"⏱ <b>Solve Time:</b> <code>{dur_s:.1f}s</code> | <b>Turns:</b> <code>{iters}</code> | <b>Proof Size:</b> <code>{p_chars} chars</code>",
        "",
        "⚖️ <b>TCS Judge Decision:</b>",
        f"  • <b>TCS Judge Score:</b> <code>{val:.2f} / 1.00</code> (Pass Threshold ≥ 0.50) → <b>{verdict_badge}</b>",
        f"  • <b>Verdict:</b> {overall_badge}",
    ]

    # Include referee critique / issues if available
    issues = entry.get("judge_issues")
    if not issues:
        try:
            tid = str(task_id)
            for jdir in [Path("data/eval_results/tcs_pipeline_300_v3/judgments"), Path("judgments")]:
                jf = jdir / f"judge_{tid}.json"
                if jf.exists():
                    jd = json.loads(jf.read_text(encoding="utf-8"))
                    b_obj = jd.get("b", {})
                    issues = b_obj.get("issues")
                    if not issues and isinstance(b_obj.get("raw"), str):
                        issues = json.loads(b_obj["raw"]).get("issues")
                    if issues:
                        break
        except Exception:
            pass

    if issues and str(issues).lower() != "none":
        clean_issues = html.escape(str(issues).strip())
        if len(clean_issues) > 300:
            clean_issues = clean_issues[:297] + "..."
        lines.append(f"  • <b>Referee Note:</b> <i>{clean_issues}</i>")

    if metrics:
        checked = metrics.get("total_checked", 0)
        target = metrics.get("target_total_problems", target_disp)
        pct = metrics.get("progress_percent", 0.0)

        tcs_data = metrics.get("judge_b") or metrics.get("tcs_judge") or {}
        proven_data = metrics.get("proven") or tcs_data
        dir_data = metrics.get("directionally_sound", {})

        tcs_passes = tcs_data.get("passes", proven_data.get("passes", 0))
        tcs_pct = tcs_data.get("pass_percentage", proven_data.get("pass_percentage", 0.0))

        dir_passes = dir_data.get("passes", 0)
        dir_pct = dir_data.get("pass_percentage", 0.0)

        score_lines = [
            "",
            f"📊 <b>{bench_name} Scoreboard ({checked}/{target} - {pct}%):</b>",
            f"  • <b>TCS Judge Pass Rate (≥0.50):</b> {tcs_passes}/{checked} ({tcs_pct}%)",
            f"  • <b>Directionally Sound (≥0.40):</b> {dir_passes}/{checked} ({dir_pct}%)",
        ]
        lines.extend(score_lines)

    return "\n".join(lines)


def format_final_summary(metrics: Dict[str, Any]) -> str:
    """Format final run completion summary in HTML for Telegram."""
    target = metrics.get("target_total_problems", 300)
    checked = metrics.get("total_checked", 0)

    bench_name = "Physics Benchmark" if target <= 5 else "TCS-Bench"

    tcs_data = metrics.get("judge_b") or metrics.get("tcs_judge") or {}
    proven_data = metrics.get("proven") or tcs_data
    dir_data = metrics.get("directionally_sound", {})

    tcs_passes = tcs_data.get("passes", proven_data.get("passes", 0))
    tcs_pct = tcs_data.get("pass_percentage", proven_data.get("pass_percentage", 0.0))

    dir_passes = dir_data.get("passes", 0)
    dir_pct = dir_data.get("pass_percentage", 0.0)

    lines = [
        f"🏁 <b>{bench_name} Complete!</b>",
        f"All <b>{checked}/{target}</b> problems evaluated under TCS Judge validation.",
        "",
        "📈 <b>Final Benchmark Metrics:</b>",
        f"  • <b>TCS Judge Pass Rate (≥0.50):</b> <code>{tcs_passes}/{checked}</code> (<b>{tcs_pct}%</b>)",
        f"  • <b>Directionally Sound (≥0.40):</b> <code>{dir_passes}/{checked}</code> (<b>{dir_pct}%</b>)",
        "",
        f"<i>Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
    ]
    return "\n".join(lines)
