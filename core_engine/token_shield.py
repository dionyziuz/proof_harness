"""Token Shielding and Traceback Sanitization for Prime Math.

Protects LLM context windows from explosive outputs (e.g. infinite loops,
massive combinatorial lists, dense matrices) and cleans tracebacks to isolate
the root-cause mathematical or coding error.
"""

from __future__ import annotations

import re
import traceback
from typing import Any, List


def shield_output(
    output: str,
    max_chars: int = 3000,
    max_lines: int = 60,
    head_lines: int = 25,
    tail_lines: int = 25,
) -> str:
    """Shields stdout/stderr from overwhelming LLM context window.

    If output exceeds limits, retains the head and tail while inserting
    a concise summary of truncated lines.
    """
    if not output:
        return ""

    lines = output.splitlines()
    total_lines = len(lines)

    if total_lines > max_lines:
        omitted = total_lines - (head_lines + tail_lines)
        if omitted > 0:
            truncated_parts = (
                lines[:head_lines]
                + [f"\n... [{omitted} lines truncated by TokenShield to preserve context] ...\n"]
                + lines[-tail_lines:]
            )
            output = "\n".join(truncated_parts)

    if len(output) > max_chars:
        half = max_chars // 2
        output = (
            output[:half]
            + f"\n... [{len(output) - max_chars} characters omitted] ...\n"
            + output[-half:]
        )

    return output


def sanitize_traceback(exc: BaseException) -> str:
    """Extracts a clean, concise traceback emphasizing user code and error cause.

    Strips internal kernel execution wrappers so the LLM gets actionable
    mathematical and logic error feedback rather than internal stack noise.
    """
    raw_tb = traceback.format_exception(type(exc), exc, exc.__traceback__)

    filtered_frames = []
    for frame in raw_tb:
        if "core_engine/kernel.py" in frame or "prime_math/kernel.py" in frame or "run_code_with_timeout" in frame:
            continue
        filtered_frames.append(frame)

    cleaned = "".join(filtered_frames).strip()
    if not cleaned:
        cleaned = f"{type(exc).__name__}: {str(exc)}"

    return shield_output(cleaned, max_chars=1500, max_lines=30)


def summarize_value(val: Any, max_len: int = 120) -> str:
    """Returns a compact preview of a Python object."""
    try:
        if hasattr(val, "as_expr"):
            s = str(val.as_expr())
        elif hasattr(val, "shape"):
            s = f"array(shape={val.shape}, dtype={val.dtype})"
        elif isinstance(val, (list, tuple, set)):
            s = f"{type(val).__name__}(len={len(val)}, preview={repr(val)[:60]})"
        elif isinstance(val, dict):
            s = f"dict(len={len(val)}, keys={list(val.keys())[:5]})"
        else:
            s = repr(val)
    except Exception:
        s = f"<{type(val).__name__} object>"

    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def extract_high_level_steps(reasoning: str, max_points: int = 3) -> List[str]:
    """Distills raw chain-of-thought tokens into concise, high-level milestone steps."""
    if not reasoning:
        return []

    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", reasoning.strip())
    steps: List[str] = []

    for s in raw_sentences:
        s = s.strip()
        # Filter out trivial conversational filler
        if len(s) < 15 or len(s) > 200:
            continue
        lower = s.lower()
        if any(lower.startswith(prefix) for prefix in ("thinking:", "need to maybe", "could also", "so i should")):
            continue

        clean = re.sub(r"^(step\s*\d+:?|first,?|next,?|then,?|we\s+can\s+)\s*", "", s, flags=re.IGNORECASE).strip()
        if clean:
            clean = clean[0].upper() + clean[1:]
            if clean not in steps:
                steps.append(clean)
        if len(steps) >= max_points:
            break

    if not steps and raw_sentences:
        first = raw_sentences[0].strip()
        steps = [first[:120] + "..." if len(first) > 120 else first]

    return steps
