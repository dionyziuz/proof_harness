"""Lightweight prompt template loader for Prime Math.

Loads external Markdown prompt templates from prompts/ directory with LRU caching.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=64)
def load_prover_prompt(name: str, domain: str = "general") -> str:
    """Loads an external prompt template by name and domain (defaulting to 'general' with fallback to 'tcs')."""
    clean_name = name.removesuffix(".md")
    candidates = [
        PROMPTS_DIR / domain / f"{clean_name}.md",
        PROMPTS_DIR / "general" / f"{clean_name}.md",
        PROMPTS_DIR / "tcs" / f"{clean_name}.md",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    content = path.read_text(encoding="utf-8")
    if "__SHARED_MATH_ENV__" in content:
        from .prompts import SHARED_MATH_ENVIRONMENT
        content = content.replace("__SHARED_MATH_ENV__", SHARED_MATH_ENVIRONMENT)
    return content


@lru_cache(maxsize=32)
def load_tcs_prompt(name: str) -> str:
    """Loads an external TCS prompt template by name (e.g. 'system', 'context_architect')."""
    return load_prover_prompt(name, domain="tcs")
