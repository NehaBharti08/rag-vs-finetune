"""Fetch statute sections from the India Code DSpace API.

Replaces the OpenStax PDF pipeline. Section-level items carry their own full
text in metadata, so this fetches structured records rather than downloading and
parsing PDFs -- which removes the three most failure-prone steps of the previous
corpus layer: outline parsing, in-body heading refinement, and printed-page
offset inference.

**Fails closed on the licensing condition.** India Code publishes no
machine-readable licence, so reuse rests on s.52(1)(q)(ii) of the Indian
Copyright Act, which exempts reproduction of bare legislative text. What can
still be enforced is the condition that exemption turns on, and it is enforced
here: every ingested item must be a SECTION-collection item, i.e. bare
statutory text rather than commentary. Anything else is rejected rather than
warned about.

**Fails closed on identity too.** The DSpace phrase query is fuzzy and returns
amendment acts alongside the principal act, so every record is filtered on an
exact `dc.title.act_name` match against the pinned title in `acts.py`. Trusting
the query would silently mix "The Indian Contract (Amendment) Act, 1996" into
the corpus.

Usage::

    uv run python -m ragft.corpus.download
    uv run python -m ragft.corpus.download --verify-only
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx

from ragft.corpus.acts import ACTS, INDIA_CODE_API, REQUIRED_COLLECTION, Act
from ragft.settings import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / "sections_raw.jsonl"
MANIFEST_PATH = RAW_DIR / "manifest.json"
PAGE_SIZE = 100


class CorpusIntegrityError(RuntimeError):
    """Raised when fetched content fails a licensing or identity condition."""


def _md(item: dict[str, Any], key: str, default: str = "") -> str:
    values = item.get("metadata", {}).get(key) or []
    return str(values[0].get("value", default)) if values else default


def fetch_page(act: Act, page: int, client: httpx.Client) -> dict[str, Any]:
    query = (
        f'dc.identifier.collection:{REQUIRED_COLLECTION} AND dc.title.act_name:"{act.short_name}"'
    )
    resp = client.get(
        INDIA_CODE_API,
        params={"query": query, "size": PAGE_SIZE, "page": page},
        headers={"Accept": "application/json"},
        timeout=60.0,
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def fetch_act(act: Act, client: httpx.Client) -> list[dict[str, Any]]:
    """All sections of one act, filtered to exact-title matches."""
    first = fetch_page(act, 0, client)
    total_pages = int(first["_embedded"]["searchResult"]["page"]["totalPages"])

    records: list[dict[str, Any]] = []
    rejected = {"wrong_act": 0, "wrong_collection": 0, "no_text": 0}

    for page in range(total_pages):
        payload = first if page == 0 else fetch_page(act, page, client)
        for obj in payload["_embedded"]["searchResult"]["_embedded"]["objects"]:
            item = obj["_embedded"]["indexableObject"]

            if _md(item, "dc.title.act_name") != act.exact_name:
                rejected["wrong_act"] += 1
                continue
            if _md(item, "dc.identifier.collection") != REQUIRED_COLLECTION:
                rejected["wrong_collection"] += 1
                continue
            if not _md(item, "dc.identifier.section_page_note").strip():
                rejected["no_text"] += 1
                continue

            records.append(
                {
                    "act_slug": act.slug,
                    "act_name": act.exact_name,
                    "act_year": int(_md(item, "dc.date.act_year", str(act.year)) or act.year),
                    "act_number": _md(item, "dc.identifier.act_number"),
                    "era": act.era,
                    "uuid": item.get("uuid", ""),
                    "section_number": _md(item, "dc.identifier.section_number"),
                    "order_number": _md(item, "dc.identifier.order_number"),
                    "title": _md(item, "dc.title"),
                    "text_html": _md(item, "dc.identifier.section_page_note"),
                    "footnote_html": _md(item, "dc.identifier.section_footnote"),
                    "repealed": _md(item, "dc.identifier.repealed") == "true",
                    "department": _md(item, "dc.identifier.department_name"),
                    "ministry": _md(item, "dc.identifier.ministry_name"),
                }
            )

    print(
        f"[{act.slug}] {len(records)} sections kept "
        f"(rejected: {', '.join(f'{k}={v}' for k, v in rejected.items() if v)or 'none'})"
    )
    if not records:
        raise CorpusIntegrityError(
            f"{act.slug}: no sections survived filtering. The pinned title "
            f"{act.exact_name!r} may no longer match India Code."
        )
    return records


def ingest(verify_only: bool = False) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    with httpx.Client(follow_redirects=True) as client:
        for act in ACTS:
            records = fetch_act(act, client)
            # A large shortfall against the expected count means the pinned
            # title has drifted or the API changed shape. Better to stop than
            # to silently benchmark on a partial statute.
            ratio = len(records) / max(1, act.expected_sections)
            if ratio < 0.8:
                raise CorpusIntegrityError(
                    f"{act.slug}: got {len(records)} sections, expected about "
                    f"{act.expected_sections}. Refusing to ingest a partial statute."
                )
            entries.append(
                {
                    "slug": act.slug,
                    "act_name": act.exact_name,
                    "year": act.year,
                    "era": act.era,
                    "replaces": act.replaces,
                    "sections_fetched": len(records),
                    "sections_expected": act.expected_sections,
                    "repealed_sections": sum(r["repealed"] for r in records),
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            all_records.extend(records)

    if not verify_only:
        with RAW_PATH.open("w", encoding="utf-8") as fh:
            for record in all_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source": "India Code (indiacode.gov.in), DSpace REST API",
        "licence_basis": (
            "Indian Copyright Act, 1957 s.52(1)(q)(ii) - reproduction of bare "
            "legislative text is not an infringement. India Code publishes no "
            "machine-readable licence field, so this is a statutory exemption "
            "rather than a publisher's grant. Weaker than the OpenStax corpus "
            "this replaced; see ATTRIBUTION.md."
        ),
        "enforced_conditions": {
            "collection": REQUIRED_COLLECTION,
            "exact_act_name_match": True,
            "note": (
                "Department is recorded but NOT gated on. Criminal law is "
                "administered by Home Affairs and contract law by Law and "
                "Justice; neither bears on whether the text is bare legislative "
                "text, which is what s.52(1)(q)(ii) turns on."
            ),
        },
        "total_sections": len(all_records),
        "acts": entries,
    }
    if not verify_only:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"\n{len(all_records)} sections -> {RAW_PATH}")
        print(f"Wrote {MANIFEST_PATH}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    ingest(verify_only=args.verify_only)


if __name__ == "__main__":
    main()
