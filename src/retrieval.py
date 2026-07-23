"""Production retrieval pipeline: parent-child chunking → hybrid search → RRF
→ cross-encoder reranking → parent expansion.

Pipeline in one picture
-----------------------

    corpus (.md/.txt)
        │
        ├─ parent chunks  (~1800 chars — the unit the LLM READS)
        │      └─ child chunks (~450 chars — the unit we SEARCH)
        │
    query ──┬── BM25 over children ────► ranked list A
            └── dense cosine over children ► ranked list B
                        │
                     RRF fusion (k=60) ──► fused candidate list
                        │
                cross-encoder rerank ────► top-N children
                        │
                 parent expansion ──────► deduplicated parent passages
                        │
                    context assembly

Why parent-child
----------------
Small children make the embedding specific enough to match a narrow query
("carbapenem resistance in K. pneumoniae in 2023"). But a 450-char window
routinely cuts a sentence in half and drops the qualifier that makes the number
meaningful. So we match on children and hand the LLM the parent. This is the
single change that moved context_recall the most (see REPORT.md §3).

BASELINE MODE
-------------
``HybridRetriever(mode="baseline")`` disables chunk hierarchy, BM25 and
reranking, leaving plain top-k dense cosine over flat chunks. This exists so
the RAGAS baseline column in the report is produced by real code rather than
by a remembered number.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np

import observability as obs
from config import settings

log = logging.getLogger(__name__)

Mode = Literal["baseline", "final"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Parent:
    id: str
    text: str
    source: str          # filename
    title: str           # first heading of the document
    ordinal: int         # position of this parent within its document


@dataclass
class Child:
    id: str
    text: str
    parent_id: str
    source: str


@dataclass
class Passage:
    """What leaves the retriever and enters the prompt."""

    text: str
    source: str
    title: str
    score: float
    matched_child: str
    citation: str = field(default="")

    def __post_init__(self) -> None:
        if not self.citation:
            self.citation = f"{self.source}"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_PARA = re.compile(r"\n\s*\n")


def _split_to_size(text: str, size: int, overlap: int = 0) -> List[str]:
    """Paragraph-aware greedy packing, with a hard character cap.

    We prefer paragraph boundaries; only if a single paragraph exceeds the cap
    do we cut mid-paragraph (with overlap so a claim spanning the cut is still
    recoverable by one of the two children).
    """
    paragraphs = [p.strip() for p in _PARA.split(text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    for para in paragraphs:
        if len(para) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            step = max(1, size - overlap)
            for i in range(0, len(para), step):
                piece = para[i : i + size]
                if piece.strip():
                    chunks.append(piece.strip())
            continue

        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= size:
            buf = f"{buf}\n\n{para}"
        else:
            chunks.append(buf)
            buf = para

    if buf:
        chunks.append(buf)
    return chunks


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]


def build_hierarchy(
    documents: Sequence[tuple[str, str]],
) -> tuple[Dict[str, Parent], List[Child]]:
    """documents: sequence of (filename, raw_text). Returns (parents, children)."""
    parents: Dict[str, Parent] = {}
    children: List[Child] = []

    for source, raw in documents:
        heading = next(
            (
                line.lstrip("# ").strip()
                for line in raw.splitlines()
                if line.strip().startswith("#")
            ),
            source,
        )
        for ordinal, ptext in enumerate(_split_to_size(raw, settings.parent_chars)):
            pid = _hash_id(source, str(ordinal))
            parents[pid] = Parent(
                id=pid, text=ptext, source=source, title=heading, ordinal=ordinal
            )
            for ctext in _split_to_size(
                ptext, settings.child_chars, settings.child_overlap
            ):
                children.append(
                    Child(
                        id=_hash_id(pid, ctext[:40]),
                        text=ctext,
                        parent_id=pid,
                        source=source,
                    )
                )

    return parents, children


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[str]], k: int = 60
) -> List[tuple[str, float]]:
    """RRF: score(d) = Σ_lists 1 / (k + rank(d)).

    Rank-based rather than score-based, which is the whole point: BM25 scores
    are unbounded and corpus-dependent while cosine sits in [-1, 1], so the two
    cannot be added directly. Ranks are comparable by construction.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "what", "which", "how", "does", "do", "did", "with", "by",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords dropped.

    Deliberately keeps digits and hyphens joined ("mrsa", "esbl", "2023",
    "third-generation"), because AMR queries are dense in codes and years and
    those are exactly the tokens BM25 earns its keep on.
    """
    tokens = re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


class HybridRetriever:
    """Loads a corpus and answers queries. Models are loaded lazily on first use."""

    def __init__(
        self,
        corpus_dir: Optional[Path] = None,
        mode: Mode = "final",
    ) -> None:
        self.corpus_dir = Path(corpus_dir or settings.corpus_dir)
        self.mode = mode
        self.parents: Dict[str, Parent] = {}
        self.children: List[Child] = []
        self._bm25 = None
        self._reranker = None
        self._embeddings: Optional[np.ndarray] = None
        self._loaded = False

    # -- loading ------------------------------------------------------------

    def _read_corpus(self) -> List[tuple[str, str]]:
        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory {self.corpus_dir} does not exist. "
                "See data/README.md for how to populate it."
            )
        docs: List[tuple[str, str]] = []
        for path in sorted(self.corpus_dir.rglob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            if path.name.upper().startswith("README"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append((path.name, text))
        if not docs:
            raise FileNotFoundError(
                f"No .md/.txt documents found in {self.corpus_dir}. "
                "See data/README.md."
            )
        return docs

    def load(self) -> "HybridRetriever":
        if self._loaded:
            return self
        docs = self._read_corpus()

        if self.mode == "baseline":
            # Flat chunks, no hierarchy: each chunk is its own parent.
            self.parents, self.children = {}, []
            for source, raw in docs:
                for ordinal, ctext in enumerate(
                    _split_to_size(raw, settings.child_chars)
                ):
                    pid = _hash_id(source, "flat", str(ordinal))
                    self.parents[pid] = Parent(
                        id=pid, text=ctext, source=source, title=source,
                        ordinal=ordinal,
                    )
                    self.children.append(
                        Child(id=pid, text=ctext, parent_id=pid, source=source)
                    )
        else:
            self.parents, self.children = build_hierarchy(docs)

        log.info(
            "Loaded %d documents → %d parents, %d children (mode=%s)",
            len(docs), len(self.parents), len(self.children), self.mode,
        )
        self._loaded = True
        return self

    # -- lazy models --------------------------------------------------------

    @property
    def reranker(self):
        # Always a local cross-encoder — OpenAI has no reranking endpoint.
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.reranker_model)
        return self._reranker

    @property
    def embeddings(self) -> np.ndarray:
        """Child embeddings, computed once via the shared embedder.

        A single batched embeddings API call over the whole corpus at load time,
        so per-query retrieval only embeds the query — the corpus is not
        re-embedded on every tool call.
        """
        if self._embeddings is None:
            from embeddings import embed

            self._embeddings = embed([c.text for c in self.children])
        return self._embeddings

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.children])
        return self._bm25

    # -- retrieval arms -----------------------------------------------------

    def _dense_arm(self, query: str, n: int) -> List[str]:
        from embeddings import embed_one

        qvec = embed_one(query)                 # normalised by the embedder
        sims = self.embeddings @ qvec           # cosine, vectors are normalised
        top = np.argsort(-sims)[:n]
        return [self.children[i].id for i in top]

    def _sparse_arm(self, query: str, n: int) -> List[str]:
        # asarray because get_scores' return type is not guaranteed across
        # rank_bm25 versions, and unary minus on a plain list raises.
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float64)
        top = np.argsort(-scores)[:n]
        # Drop zero-score hits: BM25 pads the tail with documents sharing no
        # term with the query, and feeding those into RRF is pure noise.
        return [self.children[i].id for i in top if scores[i] > 0]

    # -- public -------------------------------------------------------------

    def retrieve(self, query: str, top_n: Optional[int] = None) -> List[Passage]:
        """Return reranked, parent-expanded passages for ``query``."""
        self.load()
        top_n = top_n or settings.rerank_top_n
        by_id = {c.id: c for c in self.children}

        with obs.span(
            "retrieval.hybrid_search", input=query, mode=self.mode
        ) as sp:
            if self.mode == "baseline":
                ordered = self._dense_arm(query, top_n)
                passages = [
                    Passage(
                        text=by_id[cid].text,
                        source=by_id[cid].source,
                        title=self.parents[by_id[cid].parent_id].title,
                        score=float(len(ordered) - i),
                        matched_child=cid,
                    )
                    for i, cid in enumerate(ordered)
                ]
                sp.update(output={"n_passages": len(passages), "arm": "dense-only"})
                return passages

            n = settings.candidates_per_arm
            dense_ids = self._dense_arm(query, n)
            sparse_ids = self._sparse_arm(query, n)
            fused = reciprocal_rank_fusion(
                [dense_ids, sparse_ids], k=settings.rrf_k
            )
            sp.update(
                output={
                    "dense_hits": len(dense_ids),
                    "sparse_hits": len(sparse_ids),
                    "fused_candidates": len(fused),
                }
            )

        # Rerank a bounded candidate window — the cross-encoder is the
        # expensive step (it runs one forward pass per pair).
        candidate_ids = [cid for cid, _ in fused[: n * 2]]
        with obs.span(
            "retrieval.rerank", input={"candidates": len(candidate_ids)}
        ) as sp:
            pairs = [(query, by_id[cid].text) for cid in candidate_ids]
            if not pairs:
                sp.update(output={"n": 0})
                return []
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(
                zip(candidate_ids, [float(s) for s in scores]),
                key=lambda kv: kv[1],
                reverse=True,
            )
            sp.update(output={"top_score": ranked[0][1] if ranked else None})

        # Parent expansion + dedupe: two children of the same parent must not
        # put the parent in the context twice.
        passages: List[Passage] = []
        seen_parents: set[str] = set()
        for cid, score in ranked:
            child = by_id[cid]
            if child.parent_id in seen_parents:
                continue
            seen_parents.add(child.parent_id)
            parent = self.parents[child.parent_id]
            passages.append(
                Passage(
                    text=parent.text,
                    source=parent.source,
                    title=parent.title,
                    score=score,
                    matched_child=child.text,
                )
            )
            if len(passages) >= top_n:
                break

        return passages


def assemble_context(passages: Sequence[Passage], max_chars: int = 9000) -> str:
    """Format passages into a numbered, citable context block.

    Numbering matters: the synthesis prompt requires every EVIDENCE line to
    carry a [n] marker, which is what makes faithfulness checkable.
    """
    blocks, used = [], 0
    for i, p in enumerate(passages, start=1):
        block = f"[{i}] source: {p.source} — {p.title}\n{p.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


# A module-level singleton so the MCP server does not re-embed the corpus on
# every tool call. Embedding the seed corpus takes a few seconds; doing it per
# call would dominate latency and make the latency figure meaningless.
_default: Optional[HybridRetriever] = None


def get_retriever(mode: Mode = "final") -> HybridRetriever:
    global _default
    if _default is None or _default.mode != mode:
        _default = HybridRetriever(mode=mode).load()
    return _default
