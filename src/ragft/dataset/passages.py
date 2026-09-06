"""Split sections into passages small enough to ground a question precisely.

For the textbook corpus this project began with, sections averaged ~13,000
characters and had to be broken up: asking a model to ground a question in that
much text produces vague grounding, which weakens both the training signal and
the provenance record decontamination relies on.

**Statutes barely need splitting at all.** A statutory section has a median of
~550 characters, so in most cases the section *is* the passage — and that is the
right unit anyway, because section is also the unit of citation and of the
train/eval split. Splitting only kicks in for the long procedural sections in
the Suraksha Sanhita.

That difference forced two changes, and the first was a genuine bug the previous
corpus could never have surfaced.
"""

from __future__ import annotations

import re

from ragft.corpus.parse import Section

PASSAGE_CHARS = 1400

# A passage below this cannot ground a question. It is set to match the section
# floor in `ragft.corpus.parse` rather than sitting above it: at the previous
# value of 500 this silently discarded roughly half the statute corpus, because
# the median section is ~550 characters and anything shorter vanished. A
# threshold tuned for 13,000-character textbook sections is simply wrong here.
MIN_PASSAGE_CHARS = 120

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def to_passages(section: Section, target_chars: int = PASSAGE_CHARS) -> list[str]:
    """Greedy sentence packing up to ``target_chars``."""
    sentences = _SENTENCE_END.split(section.text)
    passages: list[str] = []
    buf: list[str] = []
    size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if size + len(sentence) > target_chars and buf:
            passages.append(" ".join(buf))
            buf, size = [], 0
        buf.append(sentence)
        size += len(sentence) + 1

    if buf:
        passages.append(" ".join(buf))

    # A trailing stub carries too little context to ground a question; fold it
    # into its predecessor rather than emitting a passage that will only
    # produce weak pairs.
    #
    # The pop MUST be sequenced before the index. Written as
    # `passages[-2] = passages[-2] + " " + passages.pop()`, Python evaluates the
    # right-hand side first, so with exactly two passages the pop leaves one
    # element and `passages[-2]` raises IndexError. The textbook corpus never
    # hit it because 13,000-character sections always produced three or more
    # passages; statutes produce exactly two often enough to crash immediately.
    if len(passages) >= 2 and len(passages[-1]) < MIN_PASSAGE_CHARS:
        tail = passages.pop()
        passages[-1] = passages[-1] + " " + tail

    return [p for p in passages if len(p) >= MIN_PASSAGE_CHARS]
