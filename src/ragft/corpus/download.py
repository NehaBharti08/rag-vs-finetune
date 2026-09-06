"""Fetch the corpus, verifying its license first.

Two properties this module is responsible for:

1. **Fail closed on licensing.** Every title's license is re-verified against
   OpenStax's own content API at ingest time. If a book does not report
   CC BY 4.0 -- because OpenStax repointed a slug, or because someone added a
   title without checking the edition -- ingestion aborts. It does not warn and
   continue. A NonCommercial corpus silently entering an MIT-licensed project
   is not a recoverable mistake once an adapter trained on it is published.

2. **Reproducible provenance.** SHA-256, source URL, byte size, page count and
   fetch timestamp for every PDF are recorded in ``data/raw/manifest.json`` so
   an ingest run can be audited or repeated. The PDFs themselves are never
   committed.

Usage::

    uv run python -m ragft.corpus.download
    uv run python -m ragft.corpus.download --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from ragft.corpus.books import (
    BOOKS,
    OPENSTAX_API,
    REQUIRED_LICENSE_NAME,
    REQUIRED_LICENSE_VERSION,
    Book,
)
from ragft.settings import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"
CHUNK_BYTES = 1 << 20


class LicenseVerificationError(RuntimeError):
    """Raised when a title does not report the license this project requires."""


def fetch_book_record(slug: str, client: httpx.Client) -> dict[str, Any]:
    """Pull one book's full record from the publisher's API."""
    resp = client.get(
        OPENSTAX_API,
        params={"type": "books.Book", "fields": "*", "slug": slug},
        timeout=30.0,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        raise LicenseVerificationError(f"OpenStax API returned no book for slug {slug!r}")
    record: dict[str, Any] = items[0]
    return record


def verify_license(book: Book, record: dict[str, Any]) -> dict[str, str]:
    """Assert the API reports CC BY 4.0 for this exact book. Raise otherwise."""
    name = record.get("license_name", "")
    version = str(record.get("license_version", ""))
    uuid = record.get("book_uuid", "")

    if uuid != book.uuid:
        raise LicenseVerificationError(
            f"{book.slug}: UUID mismatch. Expected {book.uuid}, API reports {uuid!r}. "
            f"The slug may have been repointed at a different edition -- which is exactly "
            f"the failure this check exists to catch, since editions differ in license."
        )
    if name != REQUIRED_LICENSE_NAME or version != REQUIRED_LICENSE_VERSION:
        raise LicenseVerificationError(
            f"{book.slug}: license is {name!r} {version!r}, not "
            f"{REQUIRED_LICENSE_NAME!r} {REQUIRED_LICENSE_VERSION!r}. "
            f"Aborting: a NonCommercial or ShareAlike corpus would restrict reuse of "
            f"this MIT-licensed repository and of any adapter trained on it."
        )
    return {
        "license_name": name,
        "license_version": version,
        "license_url": record.get("license_url", ""),
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def download_pdf(url: str, dest: Path, client: httpx.Client) -> None:
    """Stream to a temp file, then move into place.

    Downloading to `.part` first means an interrupted fetch never leaves a
    truncated PDF that looks complete on the next run -- this box is shared and
    long jobs do get killed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, timeout=120.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(CHUNK_BYTES):
                fh.write(chunk)
    tmp.replace(dest)


def page_count(path: Path) -> int | None:
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            return int(doc.page_count)
    except Exception:  # noqa: BLE001 - page count is provenance, not correctness
        return None


def ingest(verify_only: bool = False, force: bool = False) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    with httpx.Client(follow_redirects=True) as client:
        for book in BOOKS:
            print(f"[{book.slug}] verifying license via OpenStax API...")
            record = fetch_book_record(book.slug, client)
            license_info = verify_license(book, record)
            print(
                f"[{book.slug}] OK: {license_info['license_name']} "
                f"{license_info['license_version']}"
            )

            pdf_url = record.get("pdf_url") or record.get("high_resolution_pdf_url")
            if not pdf_url:
                raise LicenseVerificationError(f"{book.slug}: API exposes no pdf_url")

            dest = RAW_DIR / f"{book.slug}.pdf"
            if verify_only:
                entries.append(
                    {"slug": book.slug, "pdf_url": pdf_url, "verified_only": True, **license_info}
                )
                continue

            if dest.exists() and not force:
                print(f"[{book.slug}] already present, skipping download")
            else:
                print(f"[{book.slug}] downloading {pdf_url}")
                download_pdf(pdf_url, dest, client)

            size = dest.stat().st_size
            print(f"[{book.slug}] hashing {size / 1e6:.1f} MB...")
            entries.append(
                {
                    "slug": book.slug,
                    "title": book.title,
                    "edition": book.edition,
                    "uuid": book.uuid,
                    "print_isbn_13": book.print_isbn_13,
                    "publish_date": book.publish_date,
                    "pdf_url": pdf_url,
                    "path": str(dest.relative_to(REPO_ROOT)),
                    "bytes": size,
                    "sha256": sha256_of(dest),
                    "page_count": page_count(dest),
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    **license_info,
                }
            )

    manifest = {
        "note": (
            "Provenance for the ingest corpus. PDFs are NOT committed; this file is. "
            "Licenses are re-verified against the OpenStax API on every ingest and "
            "ingestion aborts if any title is not CC BY 4.0."
        ),
        "books": entries,
    }
    if not verify_only:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {MANIFEST_PATH}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true", help="check licenses without downloading"
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    ingest(verify_only=args.verify_only, force=args.force)


if __name__ == "__main__":
    main()
