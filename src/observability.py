"""Langfuse instrumentation with a no-op fallback.

Design note
-----------
Langfuse changed its Python API between v2 (``client.trace()`` /
``trace.span()``) and v3 (OpenTelemetry-style ``start_as_current_span``). Rather
than pin one and break on the other, this module exposes ONE internal interface
(``trace()`` / ``span()`` / ``generation()`` context managers) and adapts to
whichever client is installed. If Langfuse is missing or unconfigured, every
context manager becomes a no-op and the agent still runs — observability is
never allowed to be the reason a run fails.

Span hierarchy produced by one agent run:

    amr-agent-run                       (trace, tagged with agent_version)
    ├── guardrail.l1_input_filter       (span)
    ├── plan                            (generation — LLM)
    ├── tool.search_amr_literature      (span)
    ├── tool.get_resistance_profile     (span)
    ├── synthesis.sample_1..k           (generation — LLM, one per SC sample)
    ├── synthesis.vote                  (span)
    └── critic.review                   (generation — LLM)
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from typing import Any, Iterator, Optional

from config import settings

log = logging.getLogger(__name__)

_client: Any = None
_api: str = "noop"  # one of: "v3", "v2", "noop"


def _init() -> None:
    global _client, _api
    if _api != "noop" or not settings.langfuse_enabled:
        return
    try:
        from langfuse import Langfuse  # type: ignore

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        # v3 exposes start_as_current_span; v2 exposes trace.
        _api = "v3" if hasattr(_client, "start_as_current_span") else "v2"
        log.info("Langfuse initialised (%s API)", _api)
    except Exception as exc:  # pragma: no cover - depends on install
        log.warning("Langfuse disabled (%s); continuing without tracing", exc)
        _client, _api = None, "noop"


_init()


class _NoopSpan:
    """Duck-types the bits of a Langfuse span we actually call."""

    def update(self, **_: Any) -> None:
        return None

    def end(self, **_: Any) -> None:
        return None

    def score(self, **_: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

# The active trace object, so nested spans attach to the right parent when we
# are on the v2 API (which has no implicit context propagation).
_current_trace: Any = None


@contextlib.contextmanager
def trace(name: str, *, user_id: str = "local", **metadata: Any) -> Iterator[Any]:
    """Root span for one agent run. Tags the run with the agent version."""
    global _current_trace
    meta = {"agent_version": settings.agent_version, **metadata}

    if _api == "noop":
        yield _NoopSpan()
        return

    if _api == "v3":
        with _client.start_as_current_span(name=name) as sp:
            sp.update_trace(user_id=user_id, metadata=meta, tags=["amr-agent"])
            _current_trace = sp
            try:
                yield sp
            finally:
                _current_trace = None
        _flush()
        return

    # v2
    tr = _client.trace(
        name=name,
        id=str(uuid.uuid4()),
        user_id=user_id,
        metadata=meta,
        tags=["amr-agent"],
        version=settings.agent_version,
    )
    _current_trace = tr
    try:
        yield tr
    finally:
        _current_trace = None
        _flush()


@contextlib.contextmanager
def span(name: str, *, input: Any = None, **metadata: Any) -> Iterator[Any]:
    """Non-LLM unit of work: a tool call, a guardrail check, a vote."""
    if _api == "noop" or _current_trace is None:
        yield _NoopSpan()
        return

    if _api == "v3":
        with _client.start_as_current_span(name=name, input=input) as sp:
            if metadata:
                sp.update(metadata=metadata)
            yield sp
        return

    sp = _current_trace.span(name=name, input=input, metadata=metadata or None)
    try:
        yield sp
    finally:
        with contextlib.suppress(Exception):
            sp.end()


@contextlib.contextmanager
def generation(
    name: str,
    *,
    model: Optional[str] = None,
    input: Any = None,
    **metadata: Any,
) -> Iterator[Any]:
    """An LLM call. Separate from ``span`` so Langfuse can cost/token it."""
    model = model or settings.chat_model
    if _api == "noop" or _current_trace is None:
        yield _NoopSpan()
        return

    if _api == "v3":
        with _client.start_as_current_generation(
            name=name, model=model, input=input
        ) as gen:
            if metadata:
                gen.update(metadata=metadata)
            yield gen
        return

    gen = _current_trace.generation(
        name=name, model=model, input=input, metadata=metadata or None
    )
    try:
        yield gen
    finally:
        with contextlib.suppress(Exception):
            gen.end()


def _flush() -> None:
    if _client is not None:
        with contextlib.suppress(Exception):
            _client.flush()


def flush() -> None:
    """Call before process exit so buffered events are actually sent."""
    _flush()


def enabled() -> bool:
    return _api != "noop"
