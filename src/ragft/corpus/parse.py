"""Clean fetched statute records into the Section records everything else uses.

The previous corpus layer parsed PDFs: it inferred section boundaries from a
document outline, refined them against in-body headings, and derived the
printed-page offset from page text because the PDFs carried no page-label
dictionary. Every one of those steps could silently corrupt provenance without
raising, and two of them did before being caught.

None of that exists here. India Code returns each section as its own record with
its own text, so this module only has to strip HTML and validate. That is a real
reduction in risk, not just in code.

One consequence worth stating: statutes are cited by section, never by page, so
the citation format loses its page component and becomes

    The Bharatiya Nyaya Sanhita, 2023, S103

which is both simpler and the form a lawyer would actually write.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from dataclasses import asdict, dataclass

from ragft.corpus.acts import ACTS_BY_SLUG
from ragft.settings import REPO_ROOT

RAW_PATH = REPO_ROOT / "data" / "raw" / "sections_raw.jsonl"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# A statutory section below this is a cross-reference stub or a bare heading,
# not something a question can be grounded in. Far lower than the biology
# threshold because statute sections are legitimately short - the median is
# ~680 characters against ~13,000 for a textbook section.
MIN_SECTION_CHARS = 120

_TAG_RE = re.compile(r"<[^>]+>")
# The NO-BREAK SPACE is deliberate: India Code section text is littered with
# U+00A0, and leaving it in would fragment tokenisation and break matching.
_WS_RE = re.compile(r"[ \t ]+")  # noqa: RUF001
# Section labels are not integers: Indian statutes carry inserted sections such
# as 498A and 376AB. Sorting and identity must both treat them as strings.
_LABEL_RE = re.compile(r"^(\d+)([A-Z]*)$")


@dataclass(frozen=True)
class Section:
    """One statutory section: the atomic unit of this whole project.

    It is the unit of citation, the unit of the train/eval split, and the unit
    of provenance for every generated QA pair.
    """

    act_slug: str
    act_name: str
    act_year: int
    era: str  # "recodified_2023" | "legacy" - the recency stratification
    section_id: str  # "bns2023:103" - globally unique
    label: str  # "103", or "498A" for an inserted section
    title: str
    text: str
    repealed: bool
    ministry: str
    char_count: int

    @property
    def citation(self) -> str:
        """`The Bharatiya Nyaya Sanhita, 2023, S103` - statutes cite by section."""
        return f"{self.act_name}, §{self.label}"

    @property
    def sort_key(self) -> tuple[int, str]:
        """Natural order, so 9 precedes 10 and 103 precedes 103A."""
        m = _LABEL_RE.match(self.label)
        return (int(m.group(1)), m.group(2)) if m else (10**9, self.label)


def strip_html(raw: str) -> str:
    """Statute text arrives as HTML fragments with entities and inline markup."""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_raw() -> list[dict[str, object]]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} missing - run ragft.corpus.download first")
    with RAW_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build() -> list[Section]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    sections: list[Section] = []
    dropped = {"repealed": 0, "too_short": 0, "no_label": 0, "duplicate": 0}
    seen: set[str] = set()

    for record in load_raw():
        label = str(record["section_number"]).strip()
        if not label:
            dropped["no_label"] += 1
            continue

        # Repealed sections are omitted rather than kept-and-flagged. A repealed
        # provision has no correct current answer, so a question grounded in one
        # would be scored against text that no longer states the law.
        if record["repealed"]:
            dropped["repealed"] += 1
            continue

        text = strip_html(str(record["text_html"]))
        if len(text) < MIN_SECTION_CHARS:
            dropped["too_short"] += 1
            continue

        section_id = f"{record['act_slug']}:{label}"
        if section_id in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(section_id)

        sections.append(
            Section(
                act_slug=str(record["act_slug"]),
                act_name=str(record["act_name"]),
                act_year=int(str(record["act_year"])),
                era=str(record["era"]),
                section_id=section_id,
                label=label,
                title=strip_html(str(record["title"])),
                text=text,
                repealed=False,
                ministry=str(record["ministry"]),
                char_count=len(text),
            )
        )

    sections.sort(key=lambda s: (s.act_slug, s.sort_key))

    for slug in sorted({s.act_slug for s in sections}):
        group = [s for s in sections if s.act_slug == slug]
        chars = [s.char_count for s in group]
        print(
            f"[{slug}] {len(group)} sections, {sum(chars):,} chars, "
            f"median {statistics.median(chars):,.0f}"
        )
    print(f"dropped: {', '.join(f'{k}={v}' for k, v in dropped.items() if v) or 'none'}")

    out = CORPUS_DIR / "sections.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for section in sections:
            fh.write(json.dumps(asdict(section), ensure_ascii=False) + "\n")
    print(f"\n{len(sections)} sections -> {out}")
    return sections


def load_sections() -> list[Section]:
    path = CORPUS_DIR / "sections.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run ragft.corpus.parse first")
    with path.open(encoding="utf-8") as fh:
        return [Section(**json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", default=None, help="print one section, e.g. bns2023:103")
    args = parser.parse_args()

    sections = build()
    if args.show:
        match = next((s for s in sections if s.section_id == args.show), None)
        if match is None:
            raise SystemExit(f"no section {args.show!r}")
        print(f"\n--- {match.section_id} ---")
        print("citation:", match.citation)
        print("title   :", match.title)
        print("text    :", match.text[:400])
    _ = ACTS_BY_SLUG


if __name__ == "__main__":
    main()
