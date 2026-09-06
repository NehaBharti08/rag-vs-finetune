You are checking whether an answer is supported by the passages it was given.

  2 = GROUNDED. Every factual claim is supported by the passages.
  1 = PARTIAL. Mostly supported, but at least one claim goes beyond them.
  0 = UNGROUNDED. Central claims are absent from the passages, or contradict them.

Judge support, not correctness. A claim that happens to be true but does not
appear in the passages is NOT grounded — that distinction is the whole point of
this metric.

If the answer declines to answer, score 2.

PASSAGES:
{context}

ANSWER:
{response}

Reply with ONLY a JSON object, no other text:
{{"score": <0, 1, or 2>, "reason": "<one short sentence>"}}
