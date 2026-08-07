"""Split sections into passages small enough to ground a question precisely.

Sections average ~13k characters. Asking a model to generate a question
"grounded in" that much text produces vague grounding: there is no single
passage the answer traces to, which weakens both the training signal and the
provenance record decontamination relies on.

Passages are sentence-aligned so a question never depends on half a sentence.
"""

from __future__ import annotations

import re

from ragft.corpus.parse import Section

PASSAGE_CHARS = 1400
MIN_PASSAGE_CHARS = 500

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
    if len(passages) >= 2 and len(passages[-1]) < MIN_PASSAGE_CHARS:
        passages[-2] = passages[-2] + " " + passages.pop()

    return [p for p in passages if len(p) >= MIN_PASSAGE_CHARS]
