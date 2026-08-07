"""Turn OpenStax PDFs into numbered sections with real page numbers.

Sections come from the PDF outline, not from text heuristics. OpenStax ships a
clean two-level outline -- level 1 chapters, level 2 numbered sections -- so
guessing at headings from font sizes would be strictly worse.

The fiddly part is page numbers. The citation format this project trains on is
``Biology, S7.3, p.214``, where 214 is the number **printed on the page**, not
the PDF page index. These differ by the front matter (12 pages in Biology, 10
in Anatomy and Physiology), and the PDFs carry no page-label dictionary. So the
offset is measured from the page text and checked for consistency rather than
assumed -- a wrong offset would put a plausible-looking but incorrect page in
every single citation, including the ones used to score hallucination.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from ragft.corpus.books import BOOKS, BOOKS_BY_SLUG, Book
from ragft.settings import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# Level-2 outline entries look like "7.3.\xa0Oxidation of Pyruvate...*".
SECTION_RE = re.compile(r"^(\d+)\.(\d+)\.?\s*(.*)$")
# A line that is nothing but a number: the printed page number in the footer.
STANDALONE_NUM_RE = re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE)
# Body text below this length is a stub (figure-only page, section divider).
MIN_SECTION_CHARS = 400


# End-of-chapter apparatus. The last numbered section of each chapter is
# followed by these rather than by another numbered heading, so without an
# explicit cut it absorbs them -- in Biology 7.7 that was 24% of the section.
#
# Excluding them is a data-quality decision, not tidiness. `REVIEW QUESTIONS`
# and friends are existing textbook questions: left in, the generator would
# produce synthetic QA derived from them, and the "synthetic" training set
# would partly be the book's own question bank. `KEY TERMS` is a glossary list,
# not prose, and generates degenerate definitional pairs.
END_OF_CHAPTER_RE = re.compile(
    r"^[ \t]*(KEY TERMS|CHAPTER SUMMARY|REVIEW QUESTIONS|CRITICAL THINKING QUESTIONS"
    r"|VISUAL CONNECTION QUESTIONS|ART CONNECTION QUESTIONS|INTERACTIVE LINK QUESTIONS"
    r"|MULTIPLE CHOICE|FREE RESPONSE)[ \t]*$",
    re.MULTILINE,
)


def heading_pattern(chapter: int, number: int) -> re.Pattern[str]:
    """Match OpenStax's in-body section heading, e.g. ``7.3 | Oxidation of...``.

    Sections do not start at page boundaries, so slicing purely by page sweeps
    the tail of the previous section into this one. That matters more here than
    it would in a normal ingest: every generated QA pair records the section it
    came from, and decontamination trusts that record. Text attributed to the
    wrong section would put a training pair and an eval question on opposite
    sides of a split that believes it separated them.
    """
    return re.compile(rf"^[ \t]*{chapter}\.{number}[ \t]*\|", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    """One numbered textbook section: the atomic unit of this whole project.

    It is the unit of citation (``S7.3``), the unit of the train/eval split,
    and the unit of provenance for every generated QA pair.
    """

    book_slug: str
    book_title: str
    section_id: str  # "biology:7.3" - globally unique
    chapter: int
    number: int
    label: str  # "7.3"
    title: str
    text: str
    pdf_page_start: int  # 1-indexed PDF page
    pdf_page_end: int
    printed_page_start: int  # what a reader sees, used in citations
    printed_page_end: int
    char_count: int

    @property
    def citation(self) -> str:
        return f"{self.book_title}, §{self.label}, p.{self.printed_page_start}"


def clean_title(raw: str) -> str:
    """Strip non-breaking spaces and OpenStax's trailing asterisk marker."""
    return raw.replace("\xa0", " ").rstrip("*").strip().rstrip(".").strip()


def detect_page_offset(doc: pymupdf.Document, samples: int = 40) -> int:
    """Measure `pdf_page_1indexed - printed_page`, and verify it is constant.

    Sampling across the body rather than trusting one page: a single figure
    page with a stray standalone number would otherwise shift every citation in
    the book. Raises if the offset is not stable, because silently picking the
    mode of an inconsistent mapping is how you get citations that are wrong in
    a way nobody notices.
    """
    n = doc.page_count
    # Sample the middle 60% - front and back matter are numbered differently.
    lo, hi = int(n * 0.2), int(n * 0.8)
    step = max(1, (hi - lo) // samples)

    offsets: list[int] = []
    for idx in range(lo, hi, step):
        matches = STANDALONE_NUM_RE.findall(doc[idx].get_text())
        if len(matches) != 1:
            continue
        printed = int(matches[0])
        if printed <= 0:
            continue
        offsets.append((idx + 1) - printed)

    if not offsets:
        raise ValueError("could not read any printed page numbers to derive the offset")

    counts = Counter(offsets)
    offset, freq = counts.most_common(1)[0]
    agreement = freq / len(offsets)
    if agreement < 0.9:
        raise ValueError(
            f"printed-page offset is not stable ({agreement:.0%} agreement over "
            f"{len(offsets)} samples, distribution={dict(counts)}). Refusing to "
            f"guess: a wrong offset silently corrupts every citation."
        )
    return int(offset)


def parse_book(book: Book, pdf_path: Path) -> list[Section]:
    doc = pymupdf.open(pdf_path)
    offset = detect_page_offset(doc)
    toc = doc.get_toc()

    # Candidate sections: level-2 entries numbered "C.S.". This excludes front
    # matter ("1. About OpenStax"), which is numbered but not "C.S.", and
    # per-chapter "Glossary"/"Key Terms" entries, which are unnumbered.
    candidates: list[tuple[int, int, str, int]] = []
    for level, raw_title, page in toc:
        if level != 2:
            continue
        m = SECTION_RE.match(raw_title.replace("\xa0", " ").strip())
        if not m:
            continue
        candidates.append((int(m.group(1)), int(m.group(2)), clean_title(m.group(3)), int(page)))

    # A section ends where the next outline entry of ANY level begins, so
    # trailing chapter material (Key Terms, Review Questions) is not swept into
    # the last section of a chapter.
    all_starts = sorted({int(page) for _, _, page in toc} | {doc.page_count + 1})

    sections: list[Section] = []
    refined_start = refined_end = trimmed_eoc = 0
    for chapter, number, title, start in candidates:
        nxt = next((p for p in all_starts if p > start), doc.page_count + 1)
        end = max(start, nxt - 1)

        # Read one page PAST the nominal end, then cut at the next section's
        # heading. A section's tail usually spills onto the page where the next
        # section begins; stopping at `end` would silently truncate it, while
        # keeping the whole page would import the next section's opening. The
        # heading is the only boundary that is actually correct.
        text_end = min(nxt, doc.page_count)
        text = "\n".join(doc[p - 1].get_text() for p in range(start, text_end + 1))

        # Trim to the real section boundaries rather than the page boundaries.
        own = heading_pattern(chapter, number).search(text)
        if own:
            text = text[own.start() :]
            refined_start += 1

        following = next(
            ((c, n) for c, n, _, _ in candidates if (c, n) > (chapter, number)),
            None,
        )
        if following is not None:
            nxt_match = heading_pattern(*following).search(text)
            if nxt_match:
                text = text[: nxt_match.start()]
                refined_end += 1

        # Chapter-final sections are followed by end-of-chapter apparatus
        # rather than by the next numbered heading, so cut there instead.
        eoc = END_OF_CHAPTER_RE.search(text)
        if eoc:
            text = text[: eoc.start()]
            trimmed_eoc += 1

        # Drop the footer page numbers we used for the offset - they are
        # navigation furniture, not content, and would pollute generation.
        text = STANDALONE_NUM_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) < MIN_SECTION_CHARS:
            continue

        sections.append(
            Section(
                book_slug=book.slug,
                book_title=book.citation_name,
                section_id=f"{book.slug}:{chapter}.{number}",
                chapter=chapter,
                number=number,
                label=f"{chapter}.{number}",
                title=title,
                text=text,
                pdf_page_start=start,
                pdf_page_end=end,
                printed_page_start=start - offset,
                printed_page_end=end - offset,
                char_count=len(text),
            )
        )

    doc.close()
    n = len(sections)
    print(
        f"[{book.slug}] boundary refinement: "
        f"{refined_start}/{n} start headings, {refined_end}/{n} end headings, "
        f"{trimmed_eoc}/{n} end-of-chapter apparatus trimmed"
    )
    return sections


def parse_all() -> list[Section]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    all_sections: list[Section] = []
    for book in BOOKS:
        pdf_path = RAW_DIR / f"{book.slug}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} missing - run ragft.corpus.download first")
        sections = parse_book(book, pdf_path)
        chars = [s.char_count for s in sections]
        print(
            f"[{book.slug}] {len(sections)} sections, "
            f"{sum(chars):,} chars, median {statistics.median(chars):,.0f}, "
            f"pages {min(s.printed_page_start for s in sections)}-"
            f"{max(s.printed_page_end for s in sections)}"
        )
        all_sections.extend(sections)

    out = CORPUS_DIR / "sections.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for s in all_sections:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    print(f"\n{len(all_sections)} sections -> {out}")
    return all_sections


def load_sections() -> list[Section]:
    path = CORPUS_DIR / "sections.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run ragft.corpus.parse first")
    with path.open(encoding="utf-8") as fh:
        return [Section(**json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show", type=str, default=None, help="print one section, e.g. biology:7.3"
    )
    args = parser.parse_args()

    sections = parse_all()
    if args.show:
        match = next((s for s in sections if s.section_id == args.show), None)
        if match is None:
            raise SystemExit(f"no section {args.show!r}")
        print(f"\n--- {match.section_id} ---")
        print("citation:", match.citation)
        print("title   :", match.title)
        print("pages   :", f"pdf {match.pdf_page_start}-{match.pdf_page_end}")
        print("text    :", match.text[:400].replace("\n", " "), "...")
    _ = BOOKS_BY_SLUG


if __name__ == "__main__":
    main()
