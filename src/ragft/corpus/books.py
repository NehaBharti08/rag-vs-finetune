"""The corpus: which titles, which editions, and why only these.

Edition matters more than title. OpenStax relicensed much of its catalog to
CC BY-NC-SA on second editions, and the change is invisible from the title
alone. *Biology 1st ed* is CC BY 4.0; *Biology 2e* is not. NonCommercial and
ShareAlike terms would restrict downstream reuse of an MIT-licensed repo and of
a publicly published adapter, so only the first editions are usable.

These two are not a preference among many -- they are the complete set of CC BY
biology titles OpenStax publishes. Corpus selection follows VidyaRAG's
ATTRIBUTION.md so the two projects index the same text.
"""

from __future__ import annotations

from dataclasses import dataclass

OPENSTAX_API = "https://openstax.org/apps/cms/api/v2/pages/"

# What a title must report to be usable here. Verified per book at ingest time
# against the publisher's own API, never assumed from a sibling edition.
REQUIRED_LICENSE_NAME = "Creative Commons Attribution License"
REQUIRED_LICENSE_VERSION = "4.0"


@dataclass(frozen=True)
class Book:
    """One OpenStax title, pinned by slug and UUID."""

    slug: str
    title: str
    short_title: str
    edition: str
    uuid: str
    print_isbn_13: str
    publish_date: str

    @property
    def citation_name(self) -> str:
        """Book name as it appears in a citation: `Biology, S7.3, p.214`."""
        return self.short_title


# Pinned by UUID as well as slug: a slug could in principle be repointed at a
# different edition, and that would silently swap the corpus underneath every
# result in this repository.
BOOKS: tuple[Book, ...] = (
    Book(
        slug="biology",
        title="Biology",
        short_title="Biology",
        edition="1st",
        uuid="185cbf87-c72e-48f5-b51e-f14f21b5eabd",
        print_isbn_13="978-1-938168-09-3",
        publish_date="2016-10-21",
    ),
    Book(
        slug="anatomy-and-physiology",
        title="Anatomy and Physiology",
        short_title="Anatomy and Physiology",
        edition="1st",
        uuid="14fb4ad7-39a1-4eee-ab6e-3ef2482e3e22",
        print_isbn_13="978-1-938168-13-0",
        publish_date="2013-04-25",
    ),
)

BOOKS_BY_SLUG: dict[str, Book] = {b.slug: b for b in BOOKS}
