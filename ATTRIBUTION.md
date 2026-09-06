# Corpus Attribution & Licensing

This project indexes and trains on the text of Indian statutes, retrieved from
**[India Code](https://indiacode.gov.in)**, the Government of India's official
repository of central legislation, maintained by the Legislative Department,
Ministry of Law and Justice.

The source code is MIT licensed (see [LICENSE](LICENSE)). **The corpus is not
MIT**, and the basis on which it is reused is materially different from the
CC BY textbook corpus this project previously used. That difference is set out
below rather than glossed over.

---

## Indexed statutes

| Act | Year | Sections ingested | Replaces |
|---|---|---|---|
| The Bharatiya Nyaya Sanhita | 2023 | 355 | The Indian Penal Code, 1860 |
| The Bharatiya Nagarik Suraksha Sanhita | 2023 | 523 | The Code of Criminal Procedure, 1973 |
| The Bharatiya Sakshya Adhiniyam | 2023 | 161 | The Indian Evidence Act, 1872 |
| The Indian Contract Act | 1872 | 198 | — |

**1,237 sections.** Repealed sections are excluded at ingest: a repealed
provision has no correct current answer, so a question grounded in one would be
scored against text that no longer states the law.

## ⚠️ The licensing basis is statutory, not a publisher's grant

The previous corpus (OpenStax) published a machine-readable licence field, so
ingestion could fail closed by *asking the publisher* whether the content was
CC BY 4.0.

**India Code publishes no such field.** Every terms, about and end-user-agreement
route returns the same client-side application shell. The basis for reuse is
therefore a provision of Indian law rather than a licence:

> **Indian Copyright Act, 1957, s.52(1)(q)(ii)** — the reproduction or
> publication of *any Act of a Legislature* does not constitute an infringement
> of copyright, provided it is reproduced or published together with any
> commentary thereon or other original matter.

Bare statutory text is therefore freely reproducible. This is settled and
uncontroversial, but it is **an assertion of a legal position rather than a
verified licence**, and it is a weaker guarantee than the corpus it replaced.
Anyone relying on this repository should satisfy themselves independently.

### What ingestion still enforces

The exemption turns on the material being *bare legislative text*. That
condition is machine-checkable and is enforced, failing closed:

| Condition | Enforced in |
|---|---|
| Item is a `SECTION`-collection item — statutory text, not commentary | `ragft.corpus.download` |
| `dc.title.act_name` matches a pinned title **exactly** | `ragft.corpus.download` |
| Section count within 80% of expected, else abort | `ragft.corpus.download` |

The exact-title check is not pedantry. The India Code search is a fuzzy phrase
match and will happily return *The Indian Contract (Amendment) Act, 1996*
alongside the principal Act; trusting the query would silently mix amending
statutes into the corpus.

Administering ministry is recorded but **not** gated on. An earlier version
required the Legislative Department and rejected all 358 sections of the
Bharatiya Nyaya Sanhita, because criminal law is administered by the Ministry of
Home Affairs. Which ministry administers a statute has no bearing on whether its
text is bare legislative text.

## Attribution obligations, and where they land

| Artifact | How attribution is satisfied |
|---|---|
| This file | Canonical record of statutes, years, and the reuse basis |
| Every retrieved chunk | Carries `act_name`, `licence_basis` and `source_url` in its payload, so provenance survives retrieval |
| Every generated answer | Renders a citation as `The Bharatiya Nyaya Sanhita, 2023, §103` |
| The synthetic QA dataset | Derived from statutory text; its dataset card names India Code as the source |
| The published LoRA adapter | *Trained on* that derivative; its model card records the same provenance |

## Not legal advice

This is a machine-learning benchmark that measures whether a language model can
state and correctly cite statutory provisions. **Nothing it produces is legal
advice.** Generated answers are model output and may be wrong, may cite
provisions that do not exist, and may cite statutes repealed in 2023 — measuring
exactly that failure is the point of the benchmark, not an incidental risk.

The 2023 recodification is recent and subject to amendment and judicial
interpretation. For any real question, consult the current authoritative text
and a qualified advocate.

## Ingestion provenance

Raw records are **not committed**. They are fetched by `ragft.corpus.download`,
which records per-act section counts, expected counts, repealed counts and fetch
timestamps to `data/raw/manifest.json` so any ingest run is auditable.
