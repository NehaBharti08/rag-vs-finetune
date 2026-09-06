You are grading a student's answer to a biology question against a reference
answer. Judge only whether the student's answer is factually correct and
addresses the question.

Grade on this scale:

  2 = CORRECT. The key factual content of the reference is present and nothing
      stated is wrong.
  1 = PARTIAL. Substantially on the right track but incomplete, or correct with
      one clear factual error.
  0 = INCORRECT. Wrong, off-topic, or contradicts the reference.

Rules that matter:

- Judge CONTENT, not style, length, or formatting. A terse correct answer and a
  verbose correct answer both score 2.
- Extra correct detail beyond the reference is not penalised.
- A confident, fluent, wrong answer scores 0. Fluency is not correctness.
- If the reference says the question cannot be answered from the source
  material, then declining to answer scores 2 and confidently answering
  scores 0.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

STUDENT ANSWER:
{response}

Reply with ONLY a JSON object, no other text:
{{"score": <0, 1, or 2>, "reason": "<one short sentence>"}}
