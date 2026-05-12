"""
config.py
---------
Loads all environment variables once at import time.
If OPENAI_API_KEY is missing the application refuses to start with a clear error.

Variables
---------
Required:
    OPENAI_API_KEY        Your OpenAI secret key  (sk-...)

Optional:
    OPENAI_MODEL          GPT model               (default: gpt-4o)
    OPENAI_MAX_TOKENS     Max tokens per call     (default: 4096)
    OPENAI_TEMPERATURE    Sampling temperature    (default: 0.2)
    ORGANISATION          Default org in reports  (default: Patterson Cheney Automotive Group)
    ALLOWED_ORIGINS       Comma-separated CORS    (default: * — lock down in production)
    LOG_LEVEL             Uvicorn log level        (default: info)
"""

import os
from dotenv import load_dotenv

# Loads .env if present — harmless if the file doesn't exist (production
# environments set vars directly via Docker / k8s secrets / hosting platform).
load_dotenv()


def _require(name: str) -> str:
    """Read a required env var or abort with a clear message."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"\n\n  [VMA] Required environment variable not set: {name}\n"
            f"  Copy .env.example → .env and fill in the value, then restart.\n"
        )
    return value


def _optional(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


# ── Validated settings ─────────────────────────────────────────────────────────

OPENAI_API_KEY:     str   = _require("OPENAI_API_KEY")
OPENAI_MODEL:       str   = _optional("OPENAI_MODEL",        "gpt-4o")
OPENAI_MAX_TOKENS:  int   = int(_optional("OPENAI_MAX_TOKENS",  "4096"))
OPENAI_TEMPERATURE: float = float(_optional("OPENAI_TEMPERATURE", "0.2"))

ORGANISATION: str = _optional(
    "ORGANISATION", "Patterson Cheney Automotive Group"
)

_raw_origins: str = _optional("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",")]

LOG_LEVEL: str = _optional("LOG_LEVEL", "info")