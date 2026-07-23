"""Dense embeddings via the OpenAI embeddings API (text-embedding-3-small).

One module so retrieval.py and the Self-Consistency vote in reasoning.py embed
through the same code path and cannot silently diverge — e.g. searching with one
model while clustering conclusions with another, which would compare vectors
from incommensurable spaces.

The cross-encoder RERANKER is deliberately NOT here — it stays local in
retrieval.py, because OpenAI has no reranking endpoint and a reranker is a
different kind of model (it scores query-passage pairs jointly rather than
producing a vector per text).

All vectors are L2-normalised on the way out, so downstream code can use a plain
dot product as cosine similarity.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

import observability as obs
from config import embedding_cost, settings

log = logging.getLogger(__name__)


def _rough_token_count(texts: List[str]) -> int:
    # ~4 chars/token is close enough for the cost line; we do not tokenise
    # precisely just to bill an embedding call.
    return sum(len(t) for t in texts) // 4


def embed(texts: List[str], *, budget: Optional[object] = None) -> np.ndarray:
    """Return an (n, d) array of L2-normalised embeddings for ``texts``.

    Every call is traced as its own Langfuse span with a ``cost_usd`` field, so
    embedding spend is visible in the trace. If a ``budget`` is passed the token
    count is also debited; retrieval does not pass one because embedding cost at
    $0.02/1M is negligible next to chat (a whole-corpus embed rounds to a
    fraction of a cent), and the corpus is embedded once at load, not per query.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    from llm import _get_client

    with obs.span("embeddings.embed", input={"n": len(texts)}) as sp:
        client = _get_client()
        resp = client.embeddings.create(
            model=settings.openai_embedding_model, input=texts
        )
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        tokens = int(
            getattr(getattr(resp, "usage", None), "total_tokens", 0)
            or _rough_token_count(texts)
        )

        # Normalise so a dot product is cosine similarity.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        cost = embedding_cost(tokens)
        sp.update(output={"dim": int(vecs.shape[1]), "tokens": tokens,
                          "cost_usd": round(cost, 6)})

    if budget is not None:
        budget.charge(tokens, label="embeddings")

    return vecs


def embed_one(text: str) -> np.ndarray:
    """Convenience: embed a single string, return a 1-D vector."""
    return embed([text])[0]
