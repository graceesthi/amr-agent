"""Central configuration. Every tunable lives here and nowhere else.

Read once at import time from the environment (populated by .env). Modules
import ``settings`` rather than calling os.getenv directly, so that a single
grep shows every knob the system has.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Repo root = parent of src/. Everything resolves relative to this so the agent
# behaves identically whether launched from the repo root or from src/.
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- LLM (OpenAI) ------------------------------------------------------
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")  # blank = default
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    sc_temperature: float = _float("SELF_CONSISTENCY_TEMPERATURE", 0.7)
    sc_k: int = _int("SELF_CONSISTENCY_K", 3)

    # --- Retrieval ---------------------------------------------------------
    # The cross-encoder reranker is the one local model in the stack, because
    # OpenAI has no reranking endpoint (see embeddings.py / retrieval.py).
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rrf_k: int = _int("RRF_K", 60)
    candidates_per_arm: int = _int("CANDIDATES_PER_ARM", 20)
    rerank_top_n: int = _int("RERANK_TOP_N", 5)

    # Parent-child chunking. Children are what we *search*; parents are what we
    # *read*. Child ~ one dense claim, parent ~ enough surrounding text to keep
    # the claim interpretable.
    child_chars: int = _int("CHILD_CHARS", 450)
    child_overlap: int = _int("CHILD_OVERLAP", 80)
    parent_chars: int = _int("PARENT_CHARS", 1800)

    # --- Budget ------------------------------------------------------------
    token_budget_per_run: int = _int("TOKEN_BUDGET_PER_RUN", 60_000)

    # --- Observability -----------------------------------------------------
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    agent_version: str = os.getenv("AGENT_VERSION", "1.0.0")

    # --- Data --------------------------------------------------------------
    corpus_dir: Path = field(
        default_factory=lambda: ROOT / os.getenv("CORPUS_DIR", "data/seed")
    )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # Convenience aliases so downstream code reads `chat_model` rather than
    # reaching for the OpenAI-specific field names.
    @property
    def chat_model(self) -> str:
        return self.openai_model

    @property
    def dense_embedding_model(self) -> str:
        return self.openai_embedding_model


settings = Settings()

# ---------------------------------------------------------------------------
# Cost model — REAL dollars. Every chat and embedding call is billed by OpenAI.
#
# Rates are USD per 1M tokens, verified July 2026, and MUST be re-checked
# against OpenAI's current pricing page before the figure is quoted in the
# report — pricing has moved several times.
#   gpt-4o-mini:            $0.15 in / $0.60 out per 1M
#   gpt-4o:                 $2.50 in / $10.00 out per 1M (independent RAGAS judge)
#   text-embedding-3-small: $0.02 per 1M (input only)
# ---------------------------------------------------------------------------
CHAT_PRICING_USD_PER_MTOK = {
    # model: (input, output)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}
EMBEDDING_PRICING_USD_PER_MTOK = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

# Fallback rate used for any model not in the table above.
_DEFAULT_CHAT_RATE = (0.15, 0.60)


def chat_cost(prompt_tokens: int, completion_tokens: int, model: str = "") -> float:
    """USD billed for one chat call."""
    model = model or settings.chat_model
    rate_in, rate_out = CHAT_PRICING_USD_PER_MTOK.get(model, _DEFAULT_CHAT_RATE)
    return prompt_tokens / 1_000_000 * rate_in + completion_tokens / 1_000_000 * rate_out


def embedding_cost(tokens: int, model: str = "") -> float:
    """USD billed for embedding ``tokens`` tokens (embeddings are input-only)."""
    model = model or settings.dense_embedding_model
    rate = EMBEDDING_PRICING_USD_PER_MTOK.get(model, 0.02)
    return tokens / 1_000_000 * rate
