Write {n} questions that are NOT answerable from the provision below.

This is deliberate and load-bearing. Without refusal training, a fine-tuned
model confidently invents legal answers, and its measured hallucination rate
would reflect a gap in the training data rather than a property of the method.
In a legal setting a confidently invented answer is the failure that matters
most.

Each question must be:
- Plausibly about the same area of law as the provision
- Genuinely NOT answerable from this provision
- Not absurd, not obviously off-topic, not a trick question

Good: a penalty, limitation period, procedural step, or exception this
provision never addresses.
Bad: "What is the capital of France?"

Set `answer` to exactly:
"I don't have enough information in the source material to answer that."

For `why`, name what KIND of provision would be required, as a statement about
the law. Do NOT refer to "the provision", "the section", or "the text" - the
model being trained will never see one.

  BAD : "The section does not specify the limitation period."
  GOOD: "Answering this would require the limitation provision governing such
         complaints, which the source material does not cover."

PROVISION ({act}, section {label} - {title}):
---
{passage}
---

JSON array of {n} objects only:
