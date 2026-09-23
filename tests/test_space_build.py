"""The deployed demo must not disagree with the published numbers.

The first version of the Space hand-ported the citation regex into JavaScript.
It drifted: A2 graded 254 correct / 2 out-of-corpus against the published
251 / 5, because the hand-port missed typographic apostrophes in statute titles,
the optional year group, and the `Code of ...` prefix form.

Three items out of 1,200 is small, and that is exactly why it would have
survived a glance. A demo whose numbers quietly differ from the report it links
to is worse than no demo.

`ragft.analysis.build_space` now ships `CITATION_RE.pattern` to the browser
verbatim and grades with the repo's own registry, so there is nothing left to
drift. These tests pin that.
"""

from __future__ import annotations

import json
import re

import pytest

from ragft.analysis.build_space import classify
from ragft.corpus.toc import CITATION_RE
from ragft.settings import REPO_ROOT

RESPONSES = REPO_ROOT / "data" / "eval" / "responses"
FAILURES = REPO_ROOT / "reports" / "failures.json"

# build_space's vocabulary vs the failure taxonomy's, for the no-retrieval arms.
EQUIVALENT = {"near": "right_act_wrong_section", "none": "no_citation"}


@pytest.mark.skipif(not FAILURES.exists(), reason="failure report not built")
@pytest.mark.parametrize("arm", ["A1_base_zeroshot", "A3_ft_zeroshot"])
def test_demo_grading_matches_the_failure_report(arm: str) -> None:
    """Only the no-retrieval arms: the RAG arms' taxonomy keys on retrieval hits."""
    from collections import Counter

    rows = [
        json.loads(line)
        for line in (RESPONSES / f"{arm}.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    counts = Counter(
        EQUIVALENT.get(v, v)
        for v, _, _ in (classify(r["response"], r["source_section_ids"]) for r in rows)
    )
    published = json.loads(FAILURES.read_text(encoding="utf-8"))["arms"][arm]["counts"]

    assert dict(counts) == published, (
        f"{arm}: the demo grades differently from reports/failures.md.\n"
        f"  demo      {dict(counts)}\n  published {published}\n"
        "A demo whose numbers disagree with the report it links to is worse than none."
    )


class TestRegexShipsIntact:
    def test_pattern_translates_to_valid_javascript(self) -> None:
        """Python uses (?P<x>...), JavaScript uses (?<x>...). Nothing else differs."""
        js = re.sub(r"\(\?P<(\w+)>", r"(?<\1>", CITATION_RE.pattern)
        assert "(?P<" not in js
        assert js.count("(?<") == 3, "expected named groups: act, year, section"

    def test_pattern_still_has_the_shapes_that_drifted(self) -> None:
        """The three things the hand-port lost, pinned so they cannot be lost again."""
        pattern = CITATION_RE.pattern
        assert "Code\\s+of" in pattern, "the `Code of Criminal Procedure` prefix form"
        assert "’" in pattern, "typographic apostrophe in statute titles"  # noqa: RUF001
        assert "?P<year>" in pattern, "the optional year group"
