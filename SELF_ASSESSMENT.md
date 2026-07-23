# Self-assessed estimated grade

Group N · Topic 6: Drug resistance. Scored against the shared rubric. Each line
cites the evidence in the repo so the estimate is checkable, not asserted.

## Repository quality gate (pass/fail)

- [x] Repository is public and accessible
- [x] `pip install -r requirements.txt` completes without errors
- [x] `python src/agent.py` runs and produces output following the README
- [x] `python -m pytest tests/test_security.py` runs without import errors (36 passed)

**Gate: PASS.**

---

## Technical implementation — /50

| # | Criterion | Max | Est. | Justification (evidence) |
|---|-----------|-----|------|--------------------------|
| A | Retrieval pipeline | 15 | **14** | Hybrid search BM25 + dense + RRF(k=60), cross-encoder reranking, parent-child chunking — all in `src/retrieval.py`. RAGAS `context_recall` improves +0.147 (0.699→0.846) vs the flat-baseline arm. Not 15 only because precision/faithfulness traded down (explained in report §3). |
| B | MCP server | 10 | **9** | 4 tools in `src/mcp_server.py`, all with full docstrings (Use when / Do NOT use / Args / Returns / Example), all return `{"error":...}` instead of raising. Runnable standalone for MCP Inspector. |
| C | Security stack | 10 | **10** | L1 filter (8 injection patterns + NFKC/homoglyph/zero-width normalisation), L4 `ACTION_RISK_MATRIX` covering every tool (deny-by-default), `TokenBudget` integrated. 5/5 injection classes blocked, 36/36 tests in `tests/test_security.py`. |
| D | Reasoning strategy | 10 | **9** | Few-shot CoT in EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE format (`reasoning.SYNTHESIS_SYSTEM`), Self-Consistency k=3 on the synthesis step with embedding-cluster voting and confidence recalibration. Confidence tagging throughout. |
| E | Observability | 5 | **4** | Langfuse instrumentation in `src/observability.py`: span per agent run / LLM call / tool call, `agent_version` logged, a monitoring alert described in README. Estimate 4 pending a captured trace screenshot; 5 once the trace is shown live. |

**Subtotal: 46 / 50**

---

## Evaluation & measurements — /20

| # | Criterion | Max | Est. | Justification |
|---|-----------|-----|------|---------------|
| F | RAGAS baseline & improvement | 12 | **11** | RAGAS on 13 questions (≥10), all 4 metrics reported, baseline documented before Block-1 improvements, final shows measurable improvement (recall +0.147), every change linked to its technique in report §3 (including honest explanation of the two drops). |
| G | Cost & latency | 8 | **7** | Avg cost/run ($0.0021) and latency (19.9 s) over 13 runs, tool-call distribution reported, TokenBudget trigger documented via `eval/budget_demo.py` (overflow at 9,397/9,000). |

**Subtotal: 18 / 20**

---

## Report quality — /20

| # | Criterion | Max | Est. | Justification |
|---|-----------|-----|------|---------------|
| H | Problem statement & architecture | 8 | **7** | Specific user (stewardship pharmacist) + concrete scenario; architecture diagram matches running code (`docs/architecture.md`); one non-obvious design decision explained with its trade-off (L1 on retrieved passages). |
| I | EU AI Act assessment | 6 | **6** | Risk tier (limited, Art. 50) with justification referencing Art. 5 / Art. 6(1) / MDR / Annex III; obligation derived and its implementation described in a table. |
| J | Limitations & what's next | 6 | **5** | Four specific limitations with the conditions under which they manifest; concrete next-sprint items (retrieval abstention, faithfulness constraint, semantic injection classifier), not "improve the agent". |

**Subtotal: 18 / 20**

---

## AI use transparency — /10

| # | Criterion | Max | Est. | Justification |
|---|-----------|-----|------|---------------|
| K | Disclosure + code ownership | 10 | **8** | Disclosure table filled honestly (report §7): most code AI-generated under the group's direction, then reviewed; two load-bearing decisions made jointly. Estimate 8 rather than 10 because the top band requires every function to survive individual questioning — conditional on the oral. |

**Subtotal: 8 / 10**

---

## Estimated total: **90 / 100** — band: *Exceptional (85–100)*

Realistic range **85–90** depending on two orals-dependent factors: showing a live
Langfuse trace (E) and each member explaining their zone under questioning (K).
If the Langfuse trace is not shown, subtract ~1; if code ownership is shaky under
questioning, K can fall to 5–6. Conservative floor: ~82.
