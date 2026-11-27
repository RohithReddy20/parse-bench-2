"""Jurors: other readers whose page readings help choose between the pipeline's own two readings.

A juror's reading is never shown; it only votes (`docparse.fixes.select.jury_choice`). A juror can be a model the
pipeline runs itself, or a stand-in: the reading that model's authors published for a benchmark page, stored in the
model-call cache under the page's pixel hash by `evalkit.standin.import_juror`. A stand-in that has no reading for a
page is simply an absent juror.
"""

from __future__ import annotations

from docparse.models.calls import ModelCall, ModelRef, ModelResult

PUBLISHED_RUNTIME = "published"
PROMPT_ID = "juror/page"


def published_ref(name: str) -> ModelRef:
    return ModelRef(repo=f"published/{name}", revision="parsebench-2-results", dtype="na")


def published_call(name: str, page_hash: str) -> ModelCall:
    """The stand-in call of juror `name` for the page with this pixel hash."""
    return ModelCall(
        model=published_ref(name),
        runtime=PUBLISHED_RUNTIME,
        prompt_id=PROMPT_ID,
        prompt_sha="published",
        params={},
        inputs=[page_hash],
    )


def read_juror(result: ModelResult) -> str:
    return str(result.output.get("markdown") or "")
