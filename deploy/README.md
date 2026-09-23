# Deployment

Two Hugging Face Spaces come out of one pipeline, because HF prices them
differently.

| | SDK | Cost | What it does |
|---|---|---|---|
| [rag-vs-finetune-demo](https://huggingface.co/spaces/nehabharti0802/rag-vs-finetune-demo) | static | free | Client-side citation checker, 1,200 measured answers, real retrieved context |
| `space/app.py` | gradio | **HF PRO** | Adds live retrieval for any question you type |

Free `cpu-basic` returns `402 Payment Required` for Gradio and Docker Spaces —
only static Spaces are free. That is why the deployed demo is static.

## Build and deploy the static Space

```bash
uv run python -m ragft.analysis.build_space   # -> deploy/build/index.html
```

Then upload `deploy/build/index.html` plus `deploy/space/SPACE_README.md`
(renamed `README.md`) to the Space.

## Run the Gradio app locally

```bash
uv run python -m ragft.analysis.build_space   # writes the data files
cd deploy/space && pip install -r requirements.txt && python app.py
```

## The one rule

**Verdicts are computed by the repo's own validator and the regex ships to the
browser verbatim.** An earlier version hand-ported `CITATION_RE` into JavaScript
and it drifted — A2 graded 254 correct / 2 out-of-corpus against the published
251 / 5, because the port lost typographic apostrophes, the optional year group
and the `Code of ...` prefix form.

Three items in 1,200 is small enough to survive a glance, which is precisely the
problem. `tests/test_space_build.py` pins the agreement.
