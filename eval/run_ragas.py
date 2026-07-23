#!/usr/bin/env python3
"""RAGAS evaluation — baseline vs final, plus cost/latency/tool-distribution.

    python eval/build_questions.py          # once, generates the question set
    python eval/run_ragas.py --arm baseline
    python eval/run_ragas.py --arm final
    python eval/run_ragas.py --arm both --report

Outputs
-------
    eval/results_baseline.json   raw per-question records + aggregate scores
    eval/results_final.json
    eval/comparison.md           the table to paste into REPORT.md §3

What the two arms are
---------------------
BASELINE — the system before the Block 1 improvements:
    flat 450-char chunks, dense top-k cosine only, no BM25, no RRF,
    no cross-encoder reranking, no parent expansion,
    zero-shot prompt, one greedy sample, no critic.

FINAL — the system as shipped:
    parent-child chunking, BM25 + dense + RRF, cross-encoder reranking,
    parent expansion, few-shot CoT in the EVIDENCE/ANALYSIS/CONCLUSION/
    CONFIDENCE schema, Self-Consistency k=3, critic review.

Both arms are executed by real code in this repository. Nothing in the
comparison table is remembered or estimated.

⚠️  Runtime & cost: RAGAS calls the judge model several times per question per
metric. On gpt-4o-mini expect a few minutes and a few cents per arm for 13
questions (gpt-4o judge: ~15x the cost, still under a dollar). Run it once,
commit the JSON, and quote the committed numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import reasoning  # noqa: E402
from config import settings  # noqa: E402
from guardrails import TokenBudget  # noqa: E402
from llm import LLMUnavailable, health_check  # noqa: E402
from retrieval import HybridRetriever, assemble_context  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.json"


# ===========================================================================
# Running the two arms
# ===========================================================================


def run_baseline(question: str) -> dict:
    """Flat dense top-k retrieval + zero-shot single-sample synthesis."""
    budget = TokenBudget()
    started = time.perf_counter()

    retriever = HybridRetriever(mode="baseline").load()
    passages = retriever.retrieve(question, top_n=settings.rerank_top_n)
    context = assemble_context(passages)

    answer, response = reasoning.synthesise_baseline(
        question, context, budget=budget
    )

    return {
        "question": question,
        "answer": answer,
        "contexts": [p.text for p in passages],
        "latency_s": round(time.perf_counter() - started, 2),
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "cost_usd": response.cost_usd,
        "llm_calls": 1,
        "tool_calls": {},
    }


def run_final(question: str) -> dict:
    """The shipped agent, through its real entry point."""
    import agent

    result = asyncio.run(agent.answer(question))
    metrics = result.metrics

    return {
        "question": question,
        # RAGAS scores the ANSWER, so we pass the conclusion rather than the
        # full four-section block: faithfulness over a text that already quotes
        # the context verbatim in its EVIDENCE section would be inflated, and
        # would flatter the final arm for the wrong reason.
        "answer": _conclusion_of(result.answer),
        "answer_full": result.answer,
        "contexts": result.contexts,
        "confidence": result.confidence,
        "critic_verdict": result.critic_verdict,
        "self_consistency_agreement": result.agreement,
        "latency_s": round(metrics.latency_s, 2),
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "cost_usd": metrics.cost_usd,
        "llm_calls": metrics.llm_calls,
        "tool_calls": metrics.tool_calls,
        "budget_exceeded": metrics.budget.get("exceeded", False),
    }


def _conclusion_of(markdown: str) -> str:
    parsed = reasoning.parse_structured(markdown)
    return parsed.conclusion.strip() or markdown


# ===========================================================================
# RAGAS
# ===========================================================================


def score_with_ragas(records: List[dict], references: List[str]) -> Dict[str, float]:
    """Run the four metrics. Returns {} (with a warning) if RAGAS is absent."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        print(
            f"\n[!] RAGAS stack not importable ({exc}).\n"
            f"    pip install -r requirements.txt\n"
            f"    Per-question records were still written; only the aggregate "
            f"metrics are missing.\n",
            file=sys.stderr,
        )
        return {}

    # The judge. By default it is the same gpt-4o-mini that produced the
    # answers, which still carries some self-preference bias. For a truly
    # independent judge set RAGAS_JUDGE_MODEL=gpt-4o (stronger, ~15x the cost).
    # See REPORT.md §3.
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "") or settings.chat_model

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    )

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": references,
            # Newer RAGAS releases expect these names; supplying both keeps the
            # harness working across the 0.2/0.3 rename.
            "user_input": [r["question"] for r in records],
            "response": [r["answer"] for r in records],
            "retrieved_contexts": [r["contexts"] for r in records],
            "reference": references,
        }
    )

    result = evaluate(
        dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # Extracting the aggregate scores is the one place RAGAS's API has churned
    # hard across 0.1/0.2/0.3: `dict(result)` works on some versions and raises
    # KeyError on others. The stable path is to_pandas() and average the metric
    # columns ourselves — text columns coerce to NaN and drop out.
    metric_keys = [
        "context_recall", "context_precision", "faithfulness", "answer_relevancy"
    ]
    scores: Dict[str, float] = {}

    try:
        import pandas as pd

        df = result.to_pandas()
        for key in metric_keys:
            if key in df.columns:
                series = pd.to_numeric(df[key], errors="coerce")
                if series.notna().any():
                    scores[key] = round(float(series.mean()), 4)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] to_pandas() score extraction failed ({exc}); trying dict.",
              file=sys.stderr)

    # Fallbacks for versions where the aggregate is exposed directly.
    if not scores:
        for key in metric_keys:
            try:
                scores[key] = round(float(result[key]), 4)
            except Exception:  # noqa: BLE001
                pass
    if not scores:
        try:
            for key, value in dict(result).items():
                try:
                    scores[key] = round(float(value), 4)
                except (TypeError, ValueError):
                    continue
        except Exception:  # noqa: BLE001
            pass

    return scores


# ===========================================================================
# Aggregation
# ===========================================================================


def aggregate(records: List[dict]) -> dict:
    latencies = [r["latency_s"] for r in records]
    costs = [r["cost_usd"] for r in records]
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in records]

    tool_totals: Dict[str, int] = {}
    for record in records:
        for tool, n in (record.get("tool_calls") or {}).items():
            tool_totals[tool] = tool_totals.get(tool, 0) + n

    return {
        "n_runs": len(records),
        "avg_latency_s": round(statistics.mean(latencies), 2),
        "median_latency_s": round(statistics.median(latencies), 2),
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
        "avg_cost_usd": round(statistics.mean(costs), 6),
        "total_cost_usd": round(sum(costs), 6),
        "avg_tokens_per_run": int(statistics.mean(tokens)),
        "avg_llm_calls": round(statistics.mean([r["llm_calls"] for r in records]), 2),
        "tool_call_distribution": tool_totals,
        "budget_exceeded_runs": sum(
            1 for r in records if r.get("budget_exceeded")
        ),
    }


def write_comparison(baseline: dict, final: dict) -> Path:
    """Emit the markdown table for REPORT.md §3."""
    metric_names = [
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
    ]
    b_scores, f_scores = baseline.get("ragas", {}), final.get("ragas", {})

    lines = [
        "# RAGAS comparison — baseline vs final",
        "",
        f"Question set: `eval/questions.json` "
        f"({baseline.get('aggregate', {}).get('n_runs', '?')} questions). "
        f"Model: `{settings.chat_model}`. "
        f"Judge: `{os.getenv('RAGAS_JUDGE_MODEL', '') or settings.chat_model}` "
        f"(see the judge note in REPORT.md §3).",
        "",
        "| Metric | Baseline | Final | Δ |",
        "|--------|---------|-------|---|",
    ]
    for name in metric_names:
        b = b_scores.get(name)
        f = f_scores.get(name)
        delta = f"{f - b:+.4f}" if isinstance(b, float) and isinstance(f, float) else "—"
        lines.append(
            f"| {name} | {b if b is not None else 'n/a'} | "
            f"{f if f is not None else 'n/a'} | {delta} |"
        )

    lines += ["", "## Cost, latency, tool distribution", ""]
    lines += [
        "| Measure | Baseline | Final |",
        "|---------|---------|-------|",
    ]
    ba, fa = baseline.get("aggregate", {}), final.get("aggregate", {})
    for label, key in [
        ("runs", "n_runs"),
        ("avg latency (s)", "avg_latency_s"),
        ("p95 latency (s)", "p95_latency_s"),
        ("avg tokens / run", "avg_tokens_per_run"),
        ("avg LLM calls / run", "avg_llm_calls"),
        ("avg USD / run (real spend)", "avg_cost_usd"),
        ("runs that hit the TokenBudget", "budget_exceeded_runs"),
    ]:
        lines.append(f"| {label} | {ba.get(key, '—')} | {fa.get(key, '—')} |")

    lines += ["", "### Tool call distribution (final arm)", ""]
    dist = fa.get("tool_call_distribution", {})
    if dist:
        lines += ["| Tool | Calls |", "|------|-------|"]
        lines += [f"| {tool} | {n} |" for tool, n in sorted(dist.items())]
    else:
        lines.append("_No tool calls recorded._")

    note = (
        f"> USD figures are REAL spend on `{settings.chat_model}` (chat) + "
        f"`{settings.openai_embedding_model}` (embeddings), at the rates pinned "
        "in `src/config.py` (verified July 2026). Re-check them before quoting."
    )
    lines += ["", note, ""]

    path = ROOT / "eval" / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ===========================================================================
# CLI
# ===========================================================================


def run_arm(arm: str, questions: List[dict], limit: int | None) -> dict:
    items = questions[:limit] if limit else questions
    runner = run_baseline if arm == "baseline" else run_final

    records: List[dict] = []
    for i, item in enumerate(items, start=1):
        print(f"[{arm}] {i}/{len(items)}  {item['question'][:70]}...", flush=True)
        try:
            records.append(runner(item["question"]))
        except LLMUnavailable as exc:
            print(f"  ! aborting: {exc}", file=sys.stderr)
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed: {exc}", file=sys.stderr)
            records.append(
                {
                    "question": item["question"],
                    "answer": "",
                    "contexts": [],
                    "error": str(exc),
                    "latency_s": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "llm_calls": 0,
                    "tool_calls": {},
                }
            )

    usable = [r for r in records if r.get("answer")]
    references = [
        item["reference"]
        for item, record in zip(items, records)
        if record.get("answer")
    ]

    print(f"[{arm}] scoring {len(usable)} records with RAGAS...", flush=True)
    payload = {
        "arm": arm,
        "model": settings.chat_model,
        "agent_version": settings.agent_version,
        "records": records,
        "aggregate": aggregate(records),
        "ragas": score_with_ragas(usable, references) if usable else {},
    }

    out = ROOT / "eval" / f"results_{arm}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{arm}] wrote {out.relative_to(ROOT)}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation harness")
    parser.add_argument(
        "--arm", choices=["baseline", "final", "both"], default="both"
    )
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument(
        "--report", action="store_true", help="write eval/comparison.md"
    )
    args = parser.parse_args()

    if not QUESTIONS.exists():
        print("Run `python eval/build_questions.py` first.", file=sys.stderr)
        return 2

    try:
        health_check()
    except LLMUnavailable as exc:
        print(f"\n[setup] {exc}\n", file=sys.stderr)
        return 2

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]

    results: Dict[str, Any] = {}
    for arm in (["baseline", "final"] if args.arm == "both" else [args.arm]):
        results[arm] = run_arm(arm, questions, args.limit)

    if args.report or args.arm == "both":
        for arm in ("baseline", "final"):
            path = ROOT / "eval" / f"results_{arm}.json"
            if arm not in results and path.exists():
                results[arm] = json.loads(path.read_text(encoding="utf-8"))
        if "baseline" in results and "final" in results:
            out = write_comparison(results["baseline"], results["final"])
            print(f"\nWrote {out.relative_to(ROOT)} — paste into REPORT.md §3")
        else:
            print("\nRun both arms before generating the comparison.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
