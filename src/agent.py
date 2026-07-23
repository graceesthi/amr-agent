"""AMR Surveillance Agent — main loop.

    python src/agent.py                          # runs the demo question
    python src/agent.py "your question here"
    python src/agent.py --interactive
    python src/agent.py --question "..." --json  # machine-readable output

Run shape
---------

    L1 filter on the question
        └─ refuse here if blocked; nothing downstream sees the text
    plan (LLM)                        → 1-3 tool calls
    for each planned call:
        L4 gate  → MCP call → L1 filter on every returned passage
    context assembly
    synthesis (LLM × k)  → Self-Consistency vote
    critic (LLM)         → PASS / REVISE / FAIL verdict
    report

Two agent roles are involved: the ANALYST, which plans and synthesises, and the
CRITIC, which audits the analyst's output against the retrieved context before
anything is returned. The critic can downgrade confidence and can force a
FAIL, in which case the answer is returned with the verdict attached rather
than silently.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import observability as obs  # noqa: E402
import reasoning  # noqa: E402
from config import ROOT, settings  # noqa: E402
from guardrails import (  # noqa: E402
    ACTION_RISK_MATRIX,
    ActionBlocked,
    TokenBudget,
    interactive_approval,
    l1_input_filter,
    l4_action_gate,
    wrap_untrusted,
)
from llm import LLMUnavailable, chat, extract_json, health_check  # noqa: E402
from retrieval import Passage, assemble_context  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("agent")

DEMO_QUESTION = (
    "Has carbapenem resistance in Klebsiella pneumoniae worsened in Southern "
    "Europe compared with Northern Europe, and what does that imply for "
    "empirical therapy guidance?"
)


# ===========================================================================
# Result container
# ===========================================================================


@dataclass
class RunMetrics:
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)

    def record_llm(self, response) -> None:
        self.llm_calls += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.cost_usd += response.cost_usd

    def record_tool(self, name: str) -> None:
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def to_dict(self) -> dict:
        return {
            "latency_s": round(self.latency_s, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "budget": self.budget,
        }


@dataclass
class AgentResult:
    question: str
    answer: str
    confidence: str
    agreement: float
    critic_verdict: str
    critic_summary: str
    critic_issues: List[dict]
    contexts: List[str]
    blocked: bool = False
    block_reason: str = ""
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "self_consistency_agreement": round(self.agreement, 3),
            "critic": {
                "verdict": self.critic_verdict,
                "summary": self.critic_summary,
                "issues": self.critic_issues,
            },
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "n_contexts": len(self.contexts),
            "metrics": self.metrics.to_dict(),
        }


# ===========================================================================
# MCP client
# ===========================================================================


class MCPToolClient:
    """Spawns src/mcp_server.py and speaks MCP to it over stdio."""

    def __init__(self) -> None:
        self._stack: Optional[AsyncExitStack] = None
        self._session = None
        self.available: List[str] = []

    async def __aenter__(self) -> "MCPToolClient":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).resolve().parent / "mcp_server.py")],
            env=None,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

        listing = await self._session.list_tools()
        self.available = [t.name for t in listing.tools]
        log.info("MCP server ready — tools: %s", self.available)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc_info)

    async def call(self, name: str, args: dict) -> str:
        result = await self._session.call_tool(name, args)
        parts = []
        for block in result.content:
            parts.append(getattr(block, "text", str(block)))
        return "\n".join(parts)


# ===========================================================================
# Steps
# ===========================================================================


def plan_tool_calls(question: str, budget: TokenBudget, metrics: RunMetrics) -> List[dict]:
    """Ask the planner which tools to call. Falls back to a plain search."""
    fallback = [
        {"tool": "search_amr_literature", "args": {"query": question, "top_n": 5}}
    ]

    try:
        response = chat(
            reasoning.PLANNER_SYSTEM,
            f"QUESTION: {question}\n\nReturn the JSON array of tool calls.",
            span_name="plan",
            temperature=0.0,
            max_tokens=350,
            budget=budget,
        )
    except LLMUnavailable:
        raise
    metrics.record_llm(response)

    plan = extract_json(response.text, default=None)
    if not isinstance(plan, list) or not plan:
        log.warning("Planner returned unparseable output; falling back to search")
        return fallback

    valid: List[dict] = []
    for step in plan[:3]:
        if not isinstance(step, dict):
            continue
        name = step.get("tool")
        # Deny-by-default: a hallucinated tool name never reaches the gate.
        if name not in ACTION_RISK_MATRIX:
            log.warning("Planner proposed unknown tool %r — dropped", name)
            continue
        valid.append({"tool": name, "args": step.get("args") or {}})

    return valid or fallback


async def gather_evidence(
    question: str,
    client: MCPToolClient,
    budget: TokenBudget,
    metrics: RunMetrics,
    *,
    approval_fn=None,
) -> tuple[List[str], List[str]]:
    """Execute the plan. Returns (context_blocks, injection_warnings)."""
    plan = plan_tool_calls(question, budget, metrics)
    log.info("Plan: %s", [s["tool"] for s in plan])

    blocks: List[str] = []
    warnings: List[str] = []

    for step in plan:
        tool, args = step["tool"], step["args"]

        if not budget.check(2000):
            warnings.append(
                f"TokenBudget nearly exhausted — skipped remaining tool calls "
                f"after {len(blocks)} result(s)."
            )
            budget.note_skip("tool_calls")
            break

        # --- L4 -----------------------------------------------------------
        try:
            l4_action_gate(tool, args, approval_fn=approval_fn)
        except ActionBlocked as exc:
            log.warning("L4 blocked %s: %s", tool, exc)
            warnings.append(f"L4 blocked tool '{tool}': {exc}")
            continue

        # --- call ---------------------------------------------------------
        with obs.span(f"tool.{tool}", input=args) as sp:
            try:
                raw = await client.call(tool, args)
            except Exception as exc:  # noqa: BLE001
                log.warning("Tool %s failed: %s", tool, exc)
                sp.update(output={"error": str(exc)})
                warnings.append(f"Tool '{tool}' failed: {exc}")
                continue
            metrics.record_tool(tool)
            sp.update(output={"chars": len(raw)})

        # Tool output counts against the budget: it becomes prompt tokens.
        budget.charge(len(raw) // 4, label=f"tool.{tool}")

        # --- L1 on the RESULT (indirect injection defence) ------------------
        payload = extract_json(raw, default=None)

        # A tool that returned an error contributed no evidence. Letting the
        # error string through as a context block would inflate the context
        # count and give the model a "passage" to cite that says nothing.
        if isinstance(payload, dict) and "error" in payload:
            warnings.append(f"Tool '{tool}' returned an error: {payload['error']}")
            log.warning("Tool %s returned an error: %s", tool, payload["error"])
            continue

        texts = _extract_texts(payload, raw)

        for text in texts:
            check = l1_input_filter(text, origin="retrieved")
            if not check.allowed:
                warnings.append(
                    f"A passage returned by '{tool}' was dropped: "
                    f"{check.reason}"
                )
                log.warning("L1 dropped a retrieved passage: %s", check.triggered)
                continue
            if check.triggered:
                warnings.append(
                    f"A passage from '{tool}' contained suspicious content "
                    f"({', '.join(check.triggered)}); it was sanitised before use."
                )
            blocks.append(check.text)

    return blocks, warnings


def _extract_texts(payload: Any, raw: str) -> List[str]:
    """Pull the human-readable strings out of a tool's JSON envelope."""
    if not isinstance(payload, dict):
        return [raw]
    if "error" in payload:
        return [f"(tool error: {payload['error']})"]
    if "results" in payload:
        return [
            f"source: {r.get('source', '?')} — {r.get('title', '')}\n"
            f"{r.get('text', '')}"
            for r in payload.get("results", [])
        ]
    # Structured tools: pass the JSON through, it is already terse.
    return [json.dumps(payload, ensure_ascii=False, indent=2)]


def run_critic(
    question: str,
    context: str,
    answer: str,
    budget: TokenBudget,
    metrics: RunMetrics,
) -> dict:
    """Second agent role. Audits the analyst's answer; never rewrites it."""
    default = {
        "verdict": "REVISE",
        "issues": [
            {
                "check": "grounding",
                "severity": "minor",
                "detail": "Critic output could not be parsed; treat as unreviewed.",
            }
        ],
        "suggested_confidence": "MEDIUM",
        "one_line_summary": "Critic did not return parseable JSON.",
    }

    if not budget.check(1200):
        default["one_line_summary"] = "Critic skipped — TokenBudget exhausted."
        return default

    response = chat(
        reasoning.CRITIC_SYSTEM,
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT THE ANALYST WAS GIVEN:\n{context}\n\n"
        f"ANALYST'S ANSWER:\n{answer}\n\n"
        f"Return your JSON review.",
        span_name="critic.review",
        temperature=0.0,
        max_tokens=700,
        budget=budget,
    )
    metrics.record_llm(response)

    review = extract_json(response.text, default=None)
    if not isinstance(review, dict) or "verdict" not in review:
        return default

    review.setdefault("issues", [])
    review.setdefault("suggested_confidence", "MEDIUM")
    review.setdefault("one_line_summary", "")
    review["verdict"] = str(review["verdict"]).upper()
    return review


# ===========================================================================
# Orchestration
# ===========================================================================


async def run_agent(
    question: str, *, interactive: bool = False
) -> AgentResult:
    started = time.perf_counter()
    budget = TokenBudget()
    metrics = RunMetrics()
    approval_fn = interactive_approval if interactive else None

    with obs.trace(
        "amr-agent-run",
        question=question[:200],
        model=settings.chat_model,
        sc_k=settings.sc_k,
    ) as tr:
        # --- L1 on the user's question ---------------------------------
        check = l1_input_filter(question, origin="user")
        if not check.allowed:
            log.warning("L1 blocked the question: %s", check.triggered)
            metrics.latency_s = time.perf_counter() - started
            metrics.budget = budget.summary()
            return AgentResult(
                question=question,
                answer=(
                    "This request was blocked before any tool ran.\n\n"
                    f"{check.reason}\n\n"
                    "If this was a legitimate surveillance question, rephrase "
                    "it without instructions directed at the system itself."
                ),
                confidence="N/A",
                agreement=0.0,
                critic_verdict="NOT_RUN",
                critic_summary="Input rejected at L1; no answer was generated.",
                critic_issues=[],
                contexts=[],
                blocked=True,
                block_reason=check.reason,
                metrics=metrics,
            )

        safe_question = check.text

        # --- evidence ---------------------------------------------------
        async with MCPToolClient() as client:
            blocks, warnings = await gather_evidence(
                safe_question, client, budget, metrics, approval_fn=approval_fn
            )

        if not blocks:
            metrics.latency_s = time.perf_counter() - started
            metrics.budget = budget.summary()
            return AgentResult(
                question=question,
                answer=(
                    "No usable evidence was retrieved for this question, so no "
                    "answer is given.\n\n"
                    + ("\n".join(f"- {w}" for w in warnings) if warnings else
                       "- The corpus returned no matching passages.")
                ),
                confidence="LOW",
                agreement=0.0,
                critic_verdict="NOT_RUN",
                critic_summary="Nothing to review.",
                critic_issues=[],
                contexts=[],
                metrics=metrics,
            )

        passages = [
            Passage(text=b, source="tool", title="", score=0.0, matched_child="")
            for b in blocks
        ]
        context = assemble_context(passages)
        fenced_context = wrap_untrusted(context)

        # --- synthesis + Self-Consistency -------------------------------
        vote, responses = reasoning.synthesise(
            safe_question, fenced_context, budget=budget
        )
        for response in responses:
            metrics.record_llm(response)

        answer_md = vote.winner.to_markdown()

        # --- critic ------------------------------------------------------
        review = run_critic(safe_question, context, answer_md, budget, metrics)

        # Final confidence = the more conservative of Self-Consistency's
        # adjusted level and the critic's suggestion. Two independent checks,
        # and we take the lower — an agent that talks itself up when its own
        # reviewer disagrees is the failure mode worth engineering against.
        rank = reasoning._CONFIDENCE_RANK
        final_rank = min(
            rank.get(vote.adjusted_confidence, 1),
            rank.get(str(review.get("suggested_confidence", "MEDIUM")).upper(), 2),
        )
        final_confidence = reasoning._RANK_CONFIDENCE[final_rank]

        metrics.latency_s = time.perf_counter() - started
        metrics.budget = budget.summary()

        footer = [
            "",
            "---",
            f"**Self-consistency**: {vote.note} "
            f"(k={settings.sc_k}, agreement={vote.agreement:.0%})",
            f"**Critic verdict**: {review['verdict']} — "
            f"{review.get('one_line_summary', '')}",
            f"**Final confidence**: {final_confidence}",
        ]
        if review.get("issues"):
            footer.append("**Critic issues**:")
            footer += [
                f"- [{i.get('severity', '?')}] {i.get('check', '?')}: "
                f"{i.get('detail', '')}"
                for i in review["issues"]
            ]
        if warnings:
            footer.append("**Guardrail notices**:")
            footer += [f"- {w}" for w in warnings]
        if budget.was_exceeded:
            footer.append(
                f"**Budget**: ceiling of {budget.limit} tokens was exceeded at "
                f"'{budget.exceeded_at}'; the run degraded to a partial answer."
            )

        tr.update(
            output={
                "confidence": final_confidence,
                "critic_verdict": review["verdict"],
                "agreement": vote.agreement,
            },
            metadata=metrics.to_dict(),
        )

        return AgentResult(
            question=question,
            answer=answer_md + "\n" + "\n".join(footer),
            confidence=final_confidence,
            agreement=vote.agreement,
            critic_verdict=review["verdict"],
            critic_summary=review.get("one_line_summary", ""),
            critic_issues=review.get("issues", []),
            contexts=blocks,
            metrics=metrics,
        )


async def answer(question: str, *, interactive: bool = False) -> AgentResult:
    """Public entry point, also used by the RAGAS harness."""
    try:
        return await run_agent(question, interactive=interactive)
    finally:
        obs.flush()


# ===========================================================================
# CLI
# ===========================================================================


def _print(result: AgentResult) -> None:
    print("\n" + "=" * 78)
    print(f"QUESTION: {result.question}")
    print("=" * 78 + "\n")
    print(result.answer)
    print("\n" + "-" * 78)
    m = result.metrics.to_dict()
    print(
        f"model={settings.chat_model}  "
        f"latency={m['latency_s']}s  tokens={m['total_tokens']}  "
        f"llm_calls={m['llm_calls']}  tools={m['tool_calls']}  "
        f"cost=${m['cost_usd']:.6f}"
    )
    if obs.enabled():
        print(f"Langfuse trace sent to {settings.langfuse_host}")
    else:
        print("Langfuse not configured — set keys in .env to see traces.")
    print("-" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="AMR Surveillance Agent")
    parser.add_argument("question", nargs="?", help="question to answer")
    parser.add_argument("--question", dest="q_flag", help="question to answer")
    parser.add_argument(
        "--interactive", action="store_true",
        help="prompt for approval on HIGH-risk tools, then loop for questions",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    try:
        health_check()
    except LLMUnavailable as exc:
        print(f"\n[setup] {exc}\n", file=sys.stderr)
        return 2

    question = args.question or args.q_flag

    if args.interactive and not question:
        print("AMR Surveillance Agent — Ctrl-D or 'quit' to exit.\n")
        while True:
            try:
                q = input("question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in {"quit", "exit"}:
                return 0
            if not q:
                continue
            _print(asyncio.run(answer(q, interactive=True)))

    if not question:
        question = DEMO_QUESTION
        # stderr, not stdout: --json output must stay machine-parseable.
        print("[no question given — running the demo question]", file=sys.stderr)

    result = asyncio.run(answer(question, interactive=args.interactive))

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
