"""The single retrieval entry point for arms 2 and 4.

One function, one configuration. Arms 2 (base + RAG) and 4 (fine-tuned + RAG)
call exactly this, so any difference between them is the adaptation method and
nothing else.

**Retrieval sees the whole corpus, including eval-unseen sections. This is
deliberate and it matters.**

The section split governs two things: what the fine-tuned model trains on, and
where eval questions come from. It does *not* restrict what the retriever may
index -- a real RAG system indexes its whole corpus, and hiding eval sections
from it would cripple the RAG arms to flatter the parametric ones.

That asymmetry *is* the phenomenon under study. On an eval-unseen question the
retrieval arms can look the answer up while the no-retrieval arms must
generalise, and that is precisely the structural advantage retrieval has. The
benchmark measures it rather than designing it away; `parametric_answerable`
stratification is what keeps the comparison interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from ragft.retrieval.index import embedder, get_client
from ragft.settings import RetrievalConfig, Settings, load_retrieval_config


@dataclass(frozen=True)
class Retrieved:
    """One retrieved chunk, with the provenance needed to cite it."""

    text: str
    score: float
    section_id: str
    citation: str
    act_name: str
    section_label: str
    licence_basis: str
    source_url: str

    @classmethod
    def from_hit(cls, hit: Any) -> Retrieved:
        m = hit.payload or {}
        return cls(
            text=str(m.get("text", "")),
            score=float(hit.score),
            section_id=str(m.get("section_id", "")),
            citation=str(m.get("citation", "")),
            act_name=str(m.get("act_name", "")),
            section_label=str(m.get("section_label", "")),
            licence_basis=str(m.get("licence_basis", "")),
            source_url=str(m.get("source_url", "")),
        )


class Retriever:
    def __init__(
        self,
        client: QdrantClient | None = None,
        cfg: RetrievalConfig | None = None,
        collection: str | None = None,
    ) -> None:
        settings = Settings(_env_file=None)
        self.cfg = cfg or load_retrieval_config()
        self.client = client or get_client()
        self.collection = collection or settings.qdrant_collection
        self.model: TextEmbedding = embedder(settings)

    def retrieve(self, query: str) -> list[Retrieved]:
        """Fetch `top_k_retrieve` candidates, return `top_k_context`.

        The wider pool exists so a reranker has something to rerank. The frozen
        baseline profile has no reranker, so this reduces to the top
        `top_k_context` -- kept explicit rather than collapsed, because the
        config is mirrored from VidyaRAG and should read the same.
        """
        vector = next(iter(self.model.embed([query]))).tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=self.cfg.top_k_retrieve,
            with_payload=True,
        )
        return [Retrieved.from_hit(h) for h in response.points[: self.cfg.top_k_context]]

    def format_context(self, results: list[Retrieved]) -> str:
        """Render retrieved chunks for a prompt.

        Each block carries its citation, so the model can cite what it actually
        read instead of reconstructing a reference from memory -- and so a
        fabricated citation in a RAG arm is unambiguously the model's doing.
        """
        return "\n\n".join(f"[{i}] {r.citation}\n{r.text}" for i, r in enumerate(results, 1))


@lru_cache(maxsize=1)
def retriever() -> Retriever:
    return Retriever()
