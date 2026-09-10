"""The upstream contract this project's corpus ingestion depends on.

Marked `integration` because it hits the live India Code API, so it is excluded
from the default run and from CI.

**It exists because the contract broke silently.** India Code renamed the act
metadata field from `dc.title.act_name` to `dc.identifier.act_name`. The query
kept returning HTTP 200 with zero results, and the corpus ingestion produced no
sections at all. It was caught only by an end-to-end reproduction run from a
clean clone, months after the numbers were published -- meaning anyone who
cloned the repo in between could not rebuild the corpus.

The ingest failing closed (`CorpusIntegrityError`) is what kept this from being
worse: it refused to build a partial corpus rather than quietly shipping one.
This test turns the same breakage into a fast, explicit signal.

Run it whenever the corpus will not rebuild::

    uv run pytest tests/test_indiacode_contract.py -m integration -v
"""

from __future__ import annotations

import httpx
import pytest

from ragft.corpus.acts import ACTS, INDIA_CODE_API, REQUIRED_COLLECTION

pytestmark = pytest.mark.integration

# Every metadata field the ingest reads. A rename in any of these breaks the
# corpus the same way the act_name rename did.
REQUIRED_FIELDS = (
    "dc.identifier.act_name",
    "dc.identifier.collection",
    "dc.identifier.section_page_note",
    "dc.identifier.section_number",
    "dc.identifier.order_number",
    "dc.title",
)


@pytest.fixture(scope="module")
def sample() -> dict[str, object]:
    act = ACTS[0]
    query = (
        f"dc.identifier.collection:{REQUIRED_COLLECTION} "
        f'AND dc.identifier.act_name:"{act.short_name}"'
    )
    with httpx.Client(follow_redirects=True) as client:
        resp = client.get(
            INDIA_CODE_API,
            params={"query": query, "size": 1, "page": 0},
            headers={"Accept": "application/json"},
            timeout=60.0,
        )
    resp.raise_for_status()
    result = resp.json()["_embedded"]["searchResult"]
    total = int(result["page"]["totalElements"])
    assert total > 0, (
        "India Code returned ZERO results for the pinned query. This is the exact "
        "failure that broke corpus ingestion once before: the API answers 200 with "
        "an empty result set when a queried field no longer exists. Check whether "
        "the metadata field names have changed again."
    )
    objects = result["_embedded"]["objects"]
    md: dict[str, object] = objects[0]["_embedded"]["indexableObject"]["metadata"]
    return {"total": total, "metadata": md}


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_metadata_field_still_exists(sample: dict[str, object], field: str) -> None:
    md = sample["metadata"]
    assert isinstance(md, dict)
    assert field in md, (
        f"{field} is gone from India Code's response. The ingest reads it, so the "
        f"corpus cannot be rebuilt until download.py is updated. "
        f"Available fields: {sorted(md)}"
    )


def test_each_act_returns_at_least_its_expected_sections() -> None:
    """Fewer hits than expected means the pinned title stopped matching."""
    shortfalls = []
    with httpx.Client(follow_redirects=True) as client:
        for act in ACTS:
            query = (
                f"dc.identifier.collection:{REQUIRED_COLLECTION} "
                f'AND dc.identifier.act_name:"{act.short_name}"'
            )
            resp = client.get(
                INDIA_CODE_API,
                params={"query": query, "size": 1, "page": 0},
                headers={"Accept": "application/json"},
                timeout=60.0,
            )
            resp.raise_for_status()
            total = int(resp.json()["_embedded"]["searchResult"]["page"]["totalElements"])
            # The query is fuzzy and returns amendment acts too, so the hit count
            # is an upper bound on what survives exact-title filtering. It must
            # still be at least what the repo expects to keep.
            if total < act.expected_sections:
                shortfalls.append(f"{act.slug}: {total} hits < {act.expected_sections} expected")
    assert not shortfalls, "India Code returned fewer sections than pinned: " + "; ".join(
        shortfalls
    )
