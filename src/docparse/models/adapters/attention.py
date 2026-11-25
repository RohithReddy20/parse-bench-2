"""OCRia-1's attention over the page image while it wrote each word: the positions of words on pages that have
no PDF text layer. One teacher-forced forward pass per page (`ocria_grounding.capture.Grounder.unit_maps`),
stored as the top cells per word (`ocria_grounding.ground.sparsify`).

The call is identified by the page image, the answer text, the model and the attention heads. The words
("units") are not stored: both sides derive them from the answer with the same pinned code.
"""

from __future__ import annotations

from dataclasses import dataclass

from docparse.core.hashing import sha256_hex
from docparse.models.calls import ModelCall, ModelRef, ModelResult

RUNTIME = "transformers-5.22.2"
PROMPT_ID = "ocria-1/attention"
# (layer, head) pairs from the pinned snapshot's ocria_grounding/heads.json; part of the call's identity
HEADS = [[27, 13], [19, 12], [23, 14], [15, 13], [23, 0], [19, 13], [27, 1], [27, 2]]
PARAMS = {"heads": HEADS, "k_word": 64, "k_image": 512}


@dataclass(frozen=True)
class Attention:
    grid: tuple[int, int]  # rows, columns of image tokens
    idx: list[list[int]]  # per word: indices of its strongest cells in the flattened grid
    val: list[list[float]]
    has: list[bool]  # False: the word produced no tokens of its own

    def dense(self):
        """(maps [words, rows, columns] float32, has [words] bool), cells outside the kept ones being 0."""
        import numpy as np

        rows, cols = self.grid
        maps = np.zeros((len(self.idx), rows * cols), dtype=np.float32)
        for u, (cells, values) in enumerate(zip(self.idx, self.val, strict=True)):
            if cells:
                maps[u, cells] = values
        return maps.reshape(-1, rows, cols), np.array(self.has, dtype=bool)


def attention_call(page_hash: str, answer: str, model: ModelRef, prompt_sha: str) -> tuple[ModelCall, dict[str, bytes]]:
    """The call for one page and the answer blob it needs (the page image blob is the page read's own)."""
    data = answer.encode("utf-8")
    answer_hash = sha256_hex(data)
    call = ModelCall(
        model=model,
        runtime=RUNTIME,
        prompt_id=PROMPT_ID,
        prompt_sha=prompt_sha,
        params=dict(PARAMS),
        inputs=[page_hash, answer_hash],
    )
    return call, {answer_hash: data}


def read_attention(result: ModelResult) -> Attention:
    out = result.output
    rows, cols = (int(v) for v in out["grid"])
    return Attention(
        grid=(rows, cols),
        idx=[[int(i) for i in cells] for cells in out["idx"]],
        val=[[float(v) for v in values] for values in out["val"]],
        has=[bool(h) for h in out["has"]],
    )
