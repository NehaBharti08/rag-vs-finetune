"""Chunk corpus sections exactly as VidyaRAG's frozen `baseline` profile does.

512 tokens, 64 overlap, sentence-aligned. These are not chosen here -- they are
mirrored from VidyaRAG so the retrieval arms of the two projects are the same
pipeline rather than a lookalike, which is the entire apples-to-apples claim of
this benchmark.

Chunking is token-based, not character-based, because the config is stated in
tokens and because the 5-chunk context budget (~2.6k tokens) only holds if the
unit is tokens.

Every chunk carries `book_title`, `license` and `source_url` in its payload, so
CC BY attribution survives retrieval instead of being bolted on at the
presentation layer -- and carries `section_id` and page range, so a retrieved
chunk can produce a real citation rather than a plausible-looking one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llama_index.core.node_parser import SentenceSplitter

from ragft.corpus.books import BOOKS_BY_SLUG
from ragft.corpus.parse import Section
from ragft.settings import RetrievalConfig

SOURCE_URLS = {
    "biology": "https://openstax.org/details/books/biology",
    "anatomy-and-physiology": "https://openstax.org/details/books/anatomy-and-physiology",
}
LICENSE = "CC BY 4.0"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    section_id: str
    section_label: str
    section_title: str
    book_slug: str
    book_title: str
    printed_page_start: int
    printed_page_end: int
    citation: str
    license: str
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
        book = BOOKS_BY_SLUG[section.book_slug]
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
                    book_slug=section.book_slug,
                    book_title=book.citation_name,
                    printed_page_start=section.printed_page_start,
                    printed_page_end=section.printed_page_end,
                    citation=section.citation,
                    license=LICENSE,
                    source_url=SOURCE_URLS[section.book_slug],
                    split=assignment.get(section.section_id, "unknown"),
                )
            )
    return chunks
