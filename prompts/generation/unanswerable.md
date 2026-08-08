Write {n} questions that are NOT answerable from the passage below.

This is deliberate and load-bearing. Without refusal training, a fine-tuned
model confidently invents answers, and its measured hallucination rate would
reflect a gap in the training data rather than a property of the method.

Each question must be:
- Plausibly about the same topic area as the passage (biology-shaped)
- Genuinely NOT answerable from the passage given
- Not absurd, not obviously off-topic, not a trick question

Good: a specific quantity, mechanism, or comparison the passage never covers.
Bad: "What is the capital of France?"

Set `answer` to exactly:
"I don't have enough information in the source material to answer that."

For `why`, name what KIND of information would be required, as a statement
about the subject matter. Do NOT refer to "the passage", "the text", or "the
section" - the model being trained will never see one, so teaching it to talk
about a passage teaches it to hallucinate a source.

  BAD : "The passage would need to specify a chemical buffer system."
  GOOD: "Answering this would require quantitative data on blood buffer
         capacity, which the source material does not cover."

PASSAGE ({book}, section {label} - {title}):
---
{passage}
---

JSON array of {n} objects only:
