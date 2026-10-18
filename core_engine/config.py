"""Configuration loader for Prime Math.

Credentials (API keys) are NEVER hardcoded: they must be supplied via the
project-root ``.env`` file (gitignored; see ``.env.example``) or the process
environment. Endpoints/models may additionally be discovered from
~/.config/models.json. Environment variables take highest priority.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. .env or API keys) is missing."""


@dataclass
class FastComposerConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = ""  # from PRIME_MATH_FAST_API_KEY (.env) or models.json vllm2.apiKey
    model: str = "default"
    timeout_seconds: float = 25.0
    max_tokens: int = 4096
    temperature: float = 0.1
    context_window: int = 65536


@dataclass
class PrimeMathConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""  # from PRIME_MATH_API_KEY / TUNNEL_API_KEY (.env)
    model: str = "default"
    timeout_seconds: float = 240.0
    max_tokens: int = 65536
    temperature: float = 0.7
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    default_execution_timeout: float = 60.0
    max_context_window: int = 131072
    jina_api_key: str = ""  # from JINA_API_KEY (.env)
    thinking_token_budget: Optional[int] = 16384
    # vLLM-specific request params (repetition_penalty, chat_template_kwargs,
    # thinking_token_budget). Strict OpenAI-compatible endpoints (e.g. Meta's
    # Model API) reject them with 400 unknown-parameter errors; set
    # PRIME_MATH_VLLM_PARAMS=false for those.
    vllm_params: bool = True
    # Meta-native reasoning effort (minimal/low/medium/high/xhigh, plus max on
    # muse-spark-1.3). Sent as-is when set; omitted means the model picks its
    # own depth. From PRIME_MATH_REASONING_EFFORT.
    reasoning_effort: Optional[str] = None
    fast_composer: FastComposerConfig = field(default_factory=FastComposerConfig)


def validate_config(cfg: PrimeMathConfig) -> None:
    """Fail loudly if any required credential is missing."""
    missing: List[str] = []
    if not cfg.api_key:
        missing.append("PRIME_MATH_API_KEY (or TUNNEL_API_KEY)")
    if not cfg.jina_api_key:
        missing.append("JINA_API_KEY")
    if cfg.fast_composer.enabled and not cfg.fast_composer.api_key:
        missing.append("PRIME_MATH_FAST_API_KEY (or set PRIME_MATH_FAST_ENABLED=false)")
    if missing:
        raise ConfigError(
            "Missing required credentials: " + ", ".join(missing)
            + f". Set them in {DEFAULT_DOTENV_PATH} (template: .env.example)."
        )


def load_config(
    config_path: Optional[Path] = None,
    dotenv_path: Optional[Path] = None,
) -> PrimeMathConfig:
    """Loads Prime Math configuration from .env, ~/.config/models.json, and environment variables.

    Raises:
        ConfigError: if the .env file is missing or required credentials are absent.
    """
    # 0. Load the project .env (required). override=False keeps explicit process env vars authoritative.
    env_file = dotenv_path or DEFAULT_DOTENV_PATH
    if not env_file.is_file():
        raise ConfigError(
            f"Required .env file not found at {env_file}. "
            "Copy .env.example to .env and fill in your credentials."
        )
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise ConfigError("python-dotenv is required to load .env (pip install python-dotenv)") from exc
    load_dotenv(dotenv_path=env_file, override=False)

    cfg = PrimeMathConfig()

    # 1. Attempt loading from models.json
    p = config_path or Path.home() / ".config" / "models.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            providers = data.get("providers", {})
            
            # Fast Composer (Local vLLM / Gemma 4)
            vllm2_info = providers.get("vllm2", {})
            if vllm2_info:
                if "baseUrl" in vllm2_info:
                    cfg.fast_composer.base_url = vllm2_info["baseUrl"]
                if "apiKey" in vllm2_info:
                    cfg.fast_composer.api_key = vllm2_info["apiKey"]
                models = vllm2_info.get("models", [])
                if models and isinstance(models, list):
                    cfg.fast_composer.model = models[0].get("id", "default")

            # Primary Orchestrator (OpenAI-compatible vLLM / thinking endpoint)
            tunnel_info = providers.get("tunnel", {})
            if tunnel_info:
                if "baseUrl" in tunnel_info:
                    cfg.base_url = tunnel_info["baseUrl"]
                if "apiKey" in tunnel_info:
                    cfg.api_key = tunnel_info["apiKey"]
                models = tunnel_info.get("models", [])
                if models and isinstance(models, list):
                    cfg.model = models[0].get("id", "default")
                    if "maxTokens" in models[0]:
                        cfg.max_tokens = int(models[0]["maxTokens"])
                    if "contextWindow" in models[0]:
                        cfg.max_context_window = int(models[0]["contextWindow"])
        except Exception:
            # Fallback to defaults gracefully
            pass

    # 2. Environment variable overrides take highest priority
    if "PRIME_MATH_BASE_URL" in os.environ:
        cfg.base_url = os.environ["PRIME_MATH_BASE_URL"]
    elif "TUNNEL_BASE_URL" in os.environ:
        cfg.base_url = os.environ["TUNNEL_BASE_URL"]

    if "PRIME_MATH_API_KEY" in os.environ:
        cfg.api_key = os.environ["PRIME_MATH_API_KEY"]
    elif "TUNNEL_API_KEY" in os.environ:
        cfg.api_key = os.environ["TUNNEL_API_KEY"]

    if "PRIME_MATH_MODEL" in os.environ:
        cfg.model = os.environ["PRIME_MATH_MODEL"]
    elif "TUNNEL_MODEL" in os.environ:
        cfg.model = os.environ["TUNNEL_MODEL"]
    if "PRIME_MATH_TIMEOUT" in os.environ:
        try:
            cfg.timeout_seconds = float(os.environ["PRIME_MATH_TIMEOUT"])
        except ValueError:
            pass
    if "PRIME_MATH_MAX_TOKENS" in os.environ:
        try:
            cfg.max_tokens = int(os.environ["PRIME_MATH_MAX_TOKENS"])
        except ValueError:
            pass

    # Fast Composer overrides
    if "PRIME_MATH_FAST_ENABLED" in os.environ:
        val = os.environ["PRIME_MATH_FAST_ENABLED"].lower()
        cfg.fast_composer.enabled = val not in ("0", "false", "no", "off")
    if "PRIME_MATH_FAST_BASE_URL" in os.environ:
        cfg.fast_composer.base_url = os.environ["PRIME_MATH_FAST_BASE_URL"]
    if "PRIME_MATH_FAST_API_KEY" in os.environ:
        cfg.fast_composer.api_key = os.environ["PRIME_MATH_FAST_API_KEY"]

    # vLLM-specific request params on/off (for strict OpenAI-compatible endpoints)
    if "PRIME_MATH_VLLM_PARAMS" in os.environ:
        val = os.environ["PRIME_MATH_VLLM_PARAMS"].lower()
        cfg.vllm_params = val not in ("0", "false", "no", "off")

    # Meta-native reasoning effort (passed through verbatim when set)
    if "PRIME_MATH_REASONING_EFFORT" in os.environ:
        cfg.reasoning_effort = os.environ["PRIME_MATH_REASONING_EFFORT"].strip() or None
    if "JINA_API_KEY" in os.environ:
        cfg.jina_api_key = os.environ["JINA_API_KEY"]

    validate_config(cfg)
    return cfg
