"""The corpus: which Indian statutes, and why these four.

Three of them are the 2023 criminal-law recodification, in force from July 2024:

    Bharatiya Nyaya Sanhita          replaces the Indian Penal Code, 1860
    Bharatiya Nagarik Suraksha Sanhita  replaces the Code of Criminal Procedure, 1973
    Bharatiya Sakshya Adhiniyam      replaces the Indian Evidence Act, 1872

They were chosen for a measured reason, not for topicality. The biology run this
project started from hit a ceiling: Qwen2.5 already answered 75.7% of questions
with no retrieval, because OpenStax is public and heavily crawled, which
compressed every cell of the 2x2. Legislation that came into force in mid-2024
sits at or past the model's knowledge boundary, so the same measurement should
have far more room to move.

They also interlock by construction -- an offence in the Sanhita, its procedure
in the Suraksha Sanhita, its evidentiary rule in the Adhiniyam -- which is what
makes genuine multi-hop questions possible rather than contrived. That was hard
to obtain honestly from two biology textbooks.

The fourth title is deliberately old:

    The Indian Contract Act, 1872    unchanged since 1872, and famous

It is the control. The model is near-certain to know it well, so enactment
recency becomes a **second stratification variable** alongside parametric
answerability. "The model knows 1872 contract law and not 2023 criminal law" is
a result the biology corpus could not produce.

LICENSING -- read this before adding a title.

Unlike OpenStax, India Code exposes no machine-readable licence field; every
terms route returns the same client-side application shell. The basis for reuse
is therefore statutory rather than a publisher's grant:

    Indian Copyright Act, 1957, s.52(1)(q)(ii) -- the reproduction or
    publication of any Act of a Legislature is not an infringement of
    copyright, provided it is reproduced without any commentary.

That is a weaker guarantee than the biology corpus had, and it is recorded as
such in ATTRIBUTION.md rather than glossed. What the ingest CAN still enforce is
the condition the exemption actually turns on: every item must be a bare
statutory SECTION, never commentary.
"""

from __future__ import annotations

from dataclasses import dataclass

# DSpace 9.1 REST API. Section-level items carry their own full text, so there
# is no PDF to download, no outline to parse, and no printed-page offset to
# infer -- the three most failure-prone steps of the previous corpus pipeline.
INDIA_CODE_API = "https://indiacode.gov.in/server/api/discover/search/objects"

# The condition s.52(1)(q)(ii) actually turns on: the item must be a bare
# statutory SECTION rather than commentary. Ingest fails closed on this.
#
# An earlier version of this file also required the Legislative Department, which
# was wrong and rejected all 358 sections of the Bharatiya Nyaya Sanhita on the
# first run. Criminal law is administered by the Ministry of Home Affairs; the
# Contract Act by Law and Justice. Which ministry administers a statute has no
# bearing on whether its text is bare legislative text, so the department is
# recorded as metadata and never gated on.
REQUIRED_COLLECTION = "SECTION"


@dataclass(frozen=True)
class Act:
    """One statute, pinned by its exact India Code title."""

    slug: str
    # Must match `dc.title.act_name` EXACTLY. The DSpace phrase query is fuzzy
    # and will happily return amendment acts alongside the principal act, so
    # results are filtered on this string rather than trusted from the query.
    exact_name: str
    short_name: str
    year: int
    expected_sections: int
    replaces: str | None
    era: str  # "recodified_2023" | "legacy" - the recency stratification

    @property
    def citation_name(self) -> str:
        """Act name as it appears in a citation: `<name>, S103`."""
        return self.exact_name


ACTS: tuple[Act, ...] = (
    Act(
        slug="bns2023",
        exact_name="The Bharatiya Nyaya Sanhita, 2023",
        short_name="Bharatiya Nyaya Sanhita",
        year=2023,
        expected_sections=358,
        replaces="The Indian Penal Code, 1860",
        era="recodified_2023",
    ),
    Act(
        slug="bnss2023",
        exact_name="The Bharatiya Nagarik Suraksha Sanhita, 2023",
        short_name="Bharatiya Nagarik Suraksha Sanhita",
        year=2023,
        expected_sections=531,
        replaces="The Code of Criminal Procedure, 1973",
        era="recodified_2023",
    ),
    Act(
        slug="bsa2023",
        exact_name="The Bharatiya Sakshya Adhiniyam, 2023",
        short_name="Bharatiya Sakshya Adhiniyam",
        year=2023,
        expected_sections=170,
        replaces="The Indian Evidence Act, 1872",
        era="recodified_2023",
    ),
    Act(
        slug="contract1872",
        exact_name="The Indian Contract Act, 1872",
        short_name="Indian Contract Act",
        year=1872,
        expected_sections=238,
        replaces=None,
        era="legacy",
    ),
)

ACTS_BY_SLUG: dict[str, Act] = {a.slug: a for a in ACTS}
ACTS_BY_NAME: dict[str, Act] = {a.exact_name: a for a in ACTS}
