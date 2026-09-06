"""Build the vector index: chunk, embed, store in Qdrant.

Deliberate deviation from VidyaRAG, recorded rather than buried: VidyaRAG
embeds with OpenAI `text-embedding-3-small`. This project runs at zero API cost
by choice, so it embeds with `BAAI/bge-small-en-v1.5` via fastembed -- local,
ONNX, no torch, and already a transitive dependency through
`qdrant-client[fastembed]`.

Everything else about the retrieval arm is mirrored exactly: 512/64 chunking,
top_k 20 -> 5, dense only, no reranker, no corrective loop. The embedder is the
one component that differs, and it is the one that a reader should know about
when comparing numbers across the two repositories.

Qdrant runs in embedded mode: no server, no network hop. The corpus is ~3k
chunks and VidyaRAG's own store client warns past ~20k points, so this is
comfortable.

Embedding is done explicitly through fastembed rather than through
qdrant-client's convenience wrapper. That wrapper (`set_model` / `add` /
`query`) was removed in qdrant-client 1.19, and depending on a convenience
layer that moves between minor versions is how a benchmark stops reproducing.
The explicit path is more verbose and does not move.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from ragft.corpus.parse import load_sections
from ragft.corpus.split import load_splits
from ragft.retrieval.chunker import Chunk, chunk_sections
from ragft.settings import REPO_ROOT, Settings, load_retrieval_config

INDEX_DIR = REPO_ROOT / "data" / "index"
CHUNKS_PATH = REPO_ROOT / "data" / "corpus" / "chunks.jsonl"
REPORTS = REPO_ROOT / "reports"


def build_chunks(force: bool = False) -> list[Chunk]:
    """Chunk the corpus, caching to disk -- splitting takes several minutes."""
    if CHUNKS_PATH.exists() and not force:
        with CHUNKS_PATH.open(encoding="utf-8") as fh:
            return [Chunk(**json.loads(line)) for line in fh if line.strip()]

    cfg = load_retrieval_config()
    t0 = time.perf_counter()
    chunks = chunk_sections(load_sections(), cfg, load_splits())
    print(f"chunked {len(chunks):,} chunks in {time.perf_counter() - t0:.0f}s")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.payload(), ensure_ascii=False) + "\n")
    return chunks


def embedder(settings: Settings) -> TextEmbedding:
    return TextEmbedding(settings.fastembed_model)


def build_index(force: bool = False) -> dict[str, Any]:
    settings = Settings(_env_file=None)
    chunks = build_chunks(force=force)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(INDEX_DIR))
    collection = settings.qdrant_collection

    if client.collection_exists(collection):
        if not force:
            info = client.get_collection(collection)
            print(f"collection {collection!r} exists with {info.points_count:,} points")
            client.close()
            return {"collection": collection, "points": info.points_count, "rebuilt": False}
        client.delete_collection(collection)

    model = embedder(settings)
    dim = len(next(iter(model.embed(["dimension probe"]))))
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    t0 = time.perf_counter()
    vectors = list(model.embed([c.text for c in chunks]))
    embed_seconds = time.perf_counter() - t0

    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(id=i, vector=vec.tolist(), payload=chunk.payload())
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
    elapsed = time.perf_counter() - t0
    count = client.get_collection(collection).points_count
    print(f"embedded + indexed {count:,} chunks in {elapsed:.0f}s (embed {embed_seconds:.0f}s)")

    cfg = load_retrieval_config()
    summary: dict[str, Any] = {
        "collection": collection,
        "points": count,
        "rebuilt": True,
        "embedding_model": settings.fastembed_model,
        "embedding_dim": dim,
        "elapsed_seconds": round(elapsed, 1),
        "chunking": {
            "chunk_size": cfg.chunking.chunk_size,
            "chunk_overlap": cfg.chunking.chunk_overlap,
        },
        "top_k_retrieve": cfg.top_k_retrieve,
        "top_k_context": cfg.top_k_context,
        "deviation_from_vidyarag": (
            "VidyaRAG embeds with OpenAI text-embedding-3-small; this project uses "
            f"{settings.fastembed_model} locally to run at zero API cost. All other "
            "retrieval parameters are mirrored exactly."
        ),
        "by_split": {
            split: sum(c.split == split for c in chunks)
            for split in ("train", "val", "eval_unseen")
        },
        "note": (
            "The index covers the WHOLE corpus including eval_unseen sections. A real "
            "RAG system indexes everything; hiding eval sections would cripple the "
            "retrieval arms to flatter the parametric ones."
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "index_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    client.close()
    return summary


def get_client() -> QdrantClient:
    return QdrantClient(path=str(INDEX_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild from scratch")
    args = parser.parse_args()
    s = build_index(force=args.force)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
