"""Chat interface over the OpenAI API.

Responsibilities:
  * the one place that talks to the chat model,
  * token accounting (from the API's usage object),
  * every call opens a Langfuse generation span,
  * every call debits the run's TokenBudget.

Everything else in the codebase calls ``chat()`` and never touches the OpenAI
SDK directly.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import observability as obs
from config import chat_cost, settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when OpenAI is unreachable. The message tells the user how to fix."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return chat_cost(self.prompt_tokens, self.completion_tokens, self.model)


# ---------------------------------------------------------------------------
# Client (lazy)
# ---------------------------------------------------------------------------

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client

    if not settings.openai_api_key:
        raise LLMUnavailable(
            "OPENAI_API_KEY is not set. Add it to .env (get a key at "
            "platform.openai.com)."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMUnavailable(
            "The 'openai' package is not installed. Run: "
            "pip install -r requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    _client = OpenAI(**kwargs)
    return _client


def health_check() -> None:
    """Fail fast with an actionable message instead of a stack trace mid-run."""
    if not settings.openai_api_key:
        raise LLMUnavailable(
            "OPENAI_API_KEY is not set. Add it to .env (get a key at "
            "platform.openai.com)."
        )
    try:
        _get_client().models.retrieve(settings.openai_model)
    except LLMUnavailable:
        raise
    except Exception as exc:
        raise LLMUnavailable(
            f"OpenAI is not reachable with the configured key/model "
            f"({settings.openai_model}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def chat(
    system: str,
    user: str,
    *,
    span_name: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    budget: Optional["object"] = None,
    **span_metadata: Any,
) -> LLMResponse:
    """Single-turn chat completion, traced and budgeted.

    Args:
        system: system prompt.
        user: user prompt.
        span_name: name of the Langfuse generation span (e.g. "synthesis.sample_1").
        temperature: 0.0 for deterministic steps, >0 for Self-Consistency samples.
        budget: a guardrails.TokenBudget; debited after the call. Optional so
            that unit tests can call chat() without constructing one.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    with obs.generation(
        span_name, model=settings.openai_model, input=messages,
        temperature=temperature, **span_metadata,
    ) as gen:
        started = time.perf_counter()
        try:
            raw = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMUnavailable(
                f"OpenAI call '{span_name}' failed: {exc}"
            ) from exc
        latency = time.perf_counter() - started

        text = (raw.choices[0].message.content or "") if raw.choices else ""
        usage = raw.usage
        ptok = int(getattr(usage, "prompt_tokens", 0) or 0)
        ctok = int(getattr(usage, "completion_tokens", 0) or 0)

        response = LLMResponse(
            text=text, prompt_tokens=ptok, completion_tokens=ctok,
            latency_s=latency, model=settings.openai_model,
        )

        gen.update(
            output=text,
            usage_details={
                "input": ptok, "output": ctok, "total": response.total_tokens
            },
            metadata={
                "latency_s": round(latency, 3),
                "cost_usd": round(response.cost_usd, 6),
            },
        )

    if budget is not None:
        budget.charge(response.total_tokens, label=span_name)

    return response


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, default: Any = None) -> Any:
    """Best-effort JSON parse of an LLM response. Never raises."""
    candidates = []
    candidates.extend(_FENCE.findall(text))
    candidates.append(text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue

    log.debug("Could not parse JSON from LLM output: %r", text[:200])
    return default
