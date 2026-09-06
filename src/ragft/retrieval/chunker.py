"""Chunk statute sections for retrieval.

The chunking parameters (512 tokens, 64 overlap, sentence-aligned) are carried
over unchanged from the biology run of this benchmark. **They are no longer a
mirror of VidyaRAG** -- that project indexes OpenStax biology, so with the domain
switched to Indian statutes the apples-to-apples claim it supported no longer
applies. They are retained for a different and still-real reason: holding
retrieval fixed is what lets the two domains be compared to each other.

In practice the parameters barely bite here. Statute sections have a median of
~550 characters against ~13,000 for a textbook section, so most sections fall
inside a single chunk. That is a happy accident rather than a design: chunk,
section, and citation unit coincide, which removes the usual RAG failure of
retrieving half a provision.

Every chunk carries `act_name`, `licence_basis` and `source_url` in its payload,
so provenance survives retrieval instead of being reattached at the presentation
layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llama_index.core.node_parser import SentenceSplitter

from ragft.corpus.acts import ACTS_BY_SLUG
from ragft.corpus.parse import Section
from ragft.settings import RetrievalConfig

SOURCE_URL = "https://indiacode.gov.in"
# Not a licence grant. India Code publishes no machine-readable licence, so
# reuse rests on a statutory exemption -- recorded verbatim on every chunk so
# the basis travels with the text rather than living only in a README.
LICENCE_BASIS = "Indian Copyright Act 1957, s.52(1)(q)(ii) - bare legislative text"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    section_id: str
    section_label: str
    section_title: str
    act_slug: str
    act_name: str
    act_year: int
    era: str
    citation: str
    licence_basis: str
    source_url: str
    split: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def build_splitter(cfg: RetrievalConfig) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )


def chunk_sections(
    sections: list[Section],
    cfg: RetrievalConfig,
    assignment: dict[str, str],
) -> list[Chunk]:
    """Split sections into retrievable chunks, preserving provenance."""
    splitter = build_splitter(cfg)
    chunks: list[Chunk] = []

    for section in sections:
        act = ACTS_BY_SLUG[section.act_slug]
        for i, text in enumerate(splitter.split_text(section.text)):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{section.section_id}#{i}",
                    text=text,
                    section_id=section.section_id,
                    section_label=section.label,
                    section_title=section.title,
                    act_slug=section.act_slug,
                    act_name=act.citation_name,
                    act_year=section.act_year,
                    era=section.era,
                    citation=section.citation,
                    licence_basis=LICENCE_BASIS,
                    source_url=SOURCE_URL,
                    split=assignment.get(section.section_id, "unknown"),
                )
            )
    return chunks
