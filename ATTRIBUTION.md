# Corpus Attribution & Licensing

This project trains on and retrieves from openly licensed textbooks published
by **OpenStax**, a nonprofit educational initiative of Rice University.

The source code is MIT licensed (see [LICENSE](LICENSE)). **The corpus is not**,
and neither are the artifacts derived from it. Textbook content — passages
returned by retrieval, quoted in citations, paraphrased in generated answers,
rewritten into synthetic training pairs, or absorbed into adapter weights —
remains under the Creative Commons license below, and the attribution
requirement travels with it.

Corpus selection and licensing follow
[VidyaRAG's ATTRIBUTION.md](https://github.com/NehaBharti08/VidyaRAG/blob/main/ATTRIBUTION.md),
because using the same corpus is what makes the two projects comparable.

---

## Indexed titles

| Title | Edition | Published | License | Source |
|---|---|---|---|---|
| Biology | 1st | 2016-10-21 | **CC BY 4.0** | https://openstax.org/details/books/biology |
| Anatomy and Physiology | 1st | 2013-04-25 | **CC BY 4.0** | https://openstax.org/details/books/anatomy-and-physiology |

Print ISBNs: Biology `978-1-938168-09-3` · Anatomy and Physiology `978-1-938168-13-0`.
OpenStax book UUIDs: `185cbf87-c72e-48f5-b51e-f14f21b5eabd` · `14fb4ad7-39a1-4eee-ab6e-3ef2482e3e22`.

### ⚠️ Edition matters more than title

**Most OpenStax second editions are _not_ CC BY.** OpenStax relicensed much of
its catalog to CC BY-NC-SA 4.0, and the change is invisible from the title
alone:

| Title | License |
|---|---|
| Biology **1st ed** | CC BY 4.0 ✅ |
| Biology **2e** | CC BY-**NC-SA** 4.0 ❌ |
| Anatomy and Physiology **1st ed** | CC BY 4.0 ✅ |
| Anatomy and Physiology **2e** | CC BY-**NC-SA** 4.0 ❌ |
| Microbiology | CC BY-**NC-SA** 4.0 ❌ |
| Concepts of Biology | CC BY-**NC-SA** 4.0 ❌ |

NonCommercial and ShareAlike terms would restrict downstream reuse of an
MIT-licensed repository — and, here, of a publicly published adapter. **Any
title added later must have its license checked individually, by edition,
against the OpenStax content API — not assumed from a sibling edition and not
taken from a search result.**

Verification is re-run at ingest time by `ragft.corpus.download`:

```
curl "https://openstax.org/apps/cms/api/v2/pages/?type=books.Book&fields=*&slug=biology"
```

Both titles must return `license_name: Creative Commons Attribution License`,
`license_version: 4.0`. Ingest fails closed if they do not.

---

## Attribution obligations, and where they land

CC BY requires credit on redistribution. This project redistributes the corpus
in more forms than a typical RAG app does, so the obligation shows up in more
places:

| Artifact | How attribution is satisfied |
|---|---|
| This file | Canonical, machine-readable record of titles, editions, and licenses |
| Every retrieved chunk | Carries `book_title`, `license`, `source_url` in its payload, so attribution survives retrieval instead of being bolted on at the UI layer |
| Every generated answer | Renders citations as `Biology, §7.3, p.214 (OpenStax, CC BY 4.0)` |
| **The synthetic QA dataset** | Derived from CC BY content. Its Hugging Face dataset card must credit OpenStax and name both titles and editions |
| **The published LoRA adapter** | *Trained on* that derivative. Its model card's training-data provenance section is a licensing requirement, not merely good practice |

The last two are easy to forget, because a model card reads like documentation
rather than compliance. It is both.

---

## Ingestion provenance

Raw PDFs are **not committed**. They are fetched and verified by
`ragft.corpus.download`, which records SHA-256 checksums, source URLs,
retrieval timestamps, and page counts to `data/raw/manifest.json` so any ingest
run is reproducible and auditable.

Checksums and page counts are populated by the Phase 1 ingestion run.

---

## Disclaimer

This is an independent student project. It is **not affiliated with, endorsed
by, or sponsored by OpenStax or Rice University.** Generated answers are
produced by a language model and may be incorrect; they are not a substitute
for the source textbooks.
