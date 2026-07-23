# AMR Surveillance Agent

A production research agent for antimicrobial resistance surveillance. It
answers questions about resistance trends by retrieving from a document corpus
and a structured surveillance table, reasoning in an auditable format, and
having a second agent role review the answer before it is returned.

Chat and embeddings run on the **OpenAI API** (`gpt-4o-mini` +
`text-embedding-3-small`). The cross-encoder reranker is the one local model —
OpenAI has no reranking endpoint.

---

## Quickstart

```bash
git clone <this-repo>
cd amr-agent

# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configuration — add your OpenAI key
cp .env.example .env
#   then edit .env and set:  OPENAI_API_KEY=sk-...

# 3. Build the seed dataset (synthetic — see data/README.md)
python scripts/build_dataset.py

# 4. Run
python src/agent.py
```

The first run downloads the local cross-encoder reranker (~90 MB) and takes
noticeably longer than subsequent runs. Chat and embeddings hit the OpenAI API,
so a run costs a fraction of a cent — see §Evaluation in `REPORT.md`.

### Other entry points

```bash
python src/agent.py "How has MRSA changed in Southern Europe?"
python src/agent.py --interactive        # question loop, approves HIGH-risk tools
python src/agent.py --question "..." --json

python -m pytest tests/test_security.py -v   # 5 injection tests + invariants

python eval/build_questions.py               # generate the eval set
python eval/run_ragas.py --arm both --report # RAGAS baseline vs final
python eval/budget_demo.py                   # force a TokenBudget trigger

python src/mcp_server.py                     # MCP server standalone (stdio)
npx @modelcontextprotocol/inspector python src/mcp_server.py
```

### Optional web UI

A Streamlit front end is available in `ui/`. It is optional and isolated — the
core agent runs without it, so the clean-clone path above is unaffected.

```bash
pip install -r ui/requirements-ui.txt
streamlit run ui/app.py            # opens http://localhost:8501
```

It calls the exact same `agent.answer()` pipeline as the CLI and renders the
conclusion, cited evidence, confidence, critic verdict, and per-run metrics
(latency, cost, tokens, tool calls). One of the example buttons is a live
injection attempt, to show the L1 guardrail blocking it in the UI.

---

## Architecture

```
                        ┌──────────────────────────┐
   user question ──────►│  L1 input filter          │──► blocked → refusal
                        │  normalise + patterns     │
                        └────────────┬──────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │  ANALYST · planner (LLM) │  → 1-3 tool calls
                        └────────────┬──────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │  L4 action gate           │  ACTION_RISK_MATRIX
                        │  deny-by-default          │  HIGH → needs approval
                        └────────────┬──────────────┘
                                     ▼  MCP / stdio
        ┌────────────────────────────────────────────────────────┐
        │  MCP SERVER  (src/mcp_server.py)                        │
        │   search_amr_literature   get_resistance_profile        │
        │   compare_regions         export_situation_report       │
        └───────┬────────────────────────────────┬───────────────┘
                ▼                                ▼
   ┌────────────────────────┐        ┌────────────────────────┐
   │ RETRIEVAL              │        │ SURVEILLANCE TABLE     │
   │  parent-child chunking │        │  data/surveillance.json│
   │  BM25 ─┐               │        └────────────────────────┘
   │  dense ─┴─► RRF (k=60) │
   │  cross-encoder rerank  │
   │  parent expansion      │
   └───────────┬────────────┘
               ▼
   ┌──────────────────────────┐
   │  L1 filter on EVERY       │  ← indirect injection defence
   │  retrieved passage        │
   └───────────┬───────────────┘
               ▼
   ┌──────────────────────────────────────────┐
   │  ANALYST · synthesis (LLM × k=3)          │
   │   few-shot CoT:                           │
   │   EVIDENCE / ANALYSIS / CONCLUSION /       │
   │   CONFIDENCE                               │
   │   Self-Consistency vote over embedding     │
   │   clusters of the CONCLUSION               │
   └───────────┬───────────────────────────────┘
               ▼
   ┌──────────────────────────────────────────┐
   │  CRITIC (LLM) — second agent role         │
   │   grounding · citation · fabrication ·     │
   │   calibration · safety                     │
   │   → PASS / REVISE / FAIL + JSON issues     │
   └───────────┬───────────────────────────────┘
               ▼
        answer + confidence + critic verdict

   TokenBudget debits every LLM and tool call throughout.
   Langfuse spans wrap every box above.
```

Full component descriptions: [`docs/architecture.md`](docs/architecture.md).

---

## Repository layout

```
├── README.md                  ← you are here
├── REPORT.md                  the written report
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py               main loop, MCP client, critic, orchestration
│   ├── mcp_server.py          MCP server, 4 tools
│   ├── retrieval.py           parent-child chunking, hybrid search, RRF, rerank
│   ├── guardrails.py          L1 filter, L4 gate, TokenBudget
│   ├── reasoning.py           few-shot CoT prompts, Self-Consistency vote
│   ├── surveillance.py        structured record store behind two tools
│   ├── llm.py                 OpenAI chat wrapper, token accounting
│   ├── embeddings.py          OpenAI dense embeddings (shared)
│   ├── observability.py       Langfuse adapter (v2/v3) + no-op fallback
│   └── config.py              every tunable, in one place
├── tests/
│   └── test_security.py       5 injection tests + guardrail invariants
├── eval/
│   ├── build_questions.py     generates the eval set from the data
│   ├── run_ragas.py           baseline vs final, cost, latency, tool mix
│   └── budget_demo.py         forces a TokenBudget trigger
├── scripts/
│   └── build_dataset.py       generates the synthetic corpus + table
├── docs/
│   └── architecture.md
└── data/
    ├── README.md              what to put here, how to use real data
    ├── seed/                  corpus (.md) — generated, synthetic
    ├── adversarial/           poisoned document used by the security tests
    └── surveillance.json      structured records — generated, synthetic
```

---

## The four MCP tools

| Tool | Risk | What it does |
|------|------|--------------|
| `search_amr_literature` | LOW | Hybrid search over the corpus; returns reranked, parent-expanded passages |
| `get_resistance_profile` | LOW | Structured lookup of resistance percentages for one organism |
| `compare_regions` | LOW | Year-aligned numeric comparison across two or more regions |
| `export_situation_report` | **HIGH** | Writes a markdown report to `reports/` — requires explicit approval |

`export_situation_report` is denied by default. It only runs under
`--interactive`, where a human is prompted. This is deliberate: it is the one
tool with an effect outside the process, and a poisoned corpus document that
talked the model into calling it is exactly the attack the gate exists for.

---

## Observability

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`. Each run produces
one trace with at least eight spans:

```
amr-agent-run                     (trace, tagged agent_version)
├── guardrail.l1_input_filter
├── plan                          (generation)
├── guardrail.l4_action_gate
├── tool.search_amr_literature
│   ├── retrieval.hybrid_search
│   └── retrieval.rerank
├── synthesis.sample_1..3         (generations)
├── synthesis.vote
└── critic.review                 (generation)
```

Without keys the agent runs normally and tracing becomes a no-op — observability
is never allowed to be the reason a run fails.

**Suggested monitoring alert.** Alert when the rolling 1-hour rate of
`critic_verdict ∈ {REVISE, FAIL}` exceeds 30% of runs. That is the earliest
signal that retrieval quality has degraded — a corpus change, a model swap, or
an embedding-model version drift shows up as the critic rejecting groundings
long before it shows up in user complaints.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `OPENAI_API_KEY is not set` | Add your key to `.env` |
| `OpenAI is not reachable...` | Wrong key, no billing set up, or the model name is not available to your account |
| `No .md/.txt documents found in data/seed` | `python scripts/build_dataset.py` |
| `surveillance.json not found` | same — the build script writes both |
| First run hangs for ~30s | the local cross-encoder reranker is downloading |
| `ImportError: No module named mcp` / `openai` | `pip install -r requirements.txt` |

---

## Data

The seed corpus and surveillance table are **synthetic** — generated by
`scripts/build_dataset.py` so that the repository runs from a clean clone with
no network access and no manual data step. The figures are plausible in shape
but fabricated, and are labelled as such in every generated file.

`data/README.md` describes how to substitute real WHO GLASS and ECDC EARS-Net
data. The retrieval pipeline is source-agnostic: drop `.md` or `.txt` files into
`data/seed/` and they are indexed on the next run.

---

## Licence and intended use

Research and coursework. This agent reports population-level surveillance
patterns. It is not a clinical decision support tool and must not be used to
select therapy for an individual patient — see REPORT.md §5 for the regulatory
reasoning behind that boundary.
