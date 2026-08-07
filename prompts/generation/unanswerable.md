Write {n} questions that are NOT answerable from the passage below.

This is deliberate and important. Without refusal training, a fine-tuned model
confidently invents answers, and its measured hallucination rate would reflect
a gap in the training data rather than a property of the method.

Each question must be:
- Plausibly about the same topic area as the passage (biology-shaped)
- Genuinely NOT answerable from the passage given
- Not absurd, not obviously off-topic, not a trick question

Good: a specific quantity, mechanism, or comparison the passage never covers.
Bad: "What is the capital of France?"

For these, set `answer` to exactly:
"I don't have enough information in the source material to answer that."
And in `why`, state in one sentence what the passage would need to contain.

PASSAGE ({book}, section {label} - {title}):
---
{passage}
---

JSON array of {n} objects only:
