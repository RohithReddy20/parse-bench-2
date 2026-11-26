"""PaddlePaddle PP-DocLayoutV3 (RT-DETR layout detector): build cached call and read detections.

Loads natively via `transformers` (`AutoModelForObjectDetection`, `AutoImageProcessor`), zero remote code.
Provides fine-grained layout boxes (including explicit `chart`, `image`, `table`, `seal`, `formula`,
`paragraph_title`, `doc_title`, `header`, `footer`, `footnote`) mapped cleanly to Docling classes.
"""

from __future__ import annotations

from docparse.models.adapters.heron import Detection
from docparse.models.calls import ModelCall, ModelRef, ModelResult

PP_DOCLAYOUT_V3 = ModelRef(
    repo="PaddlePaddle/PP-DocLayoutV3_safetensors",
    revision="e4f489dc536556fc1be02b973ba06756e0d4a2ac",
    dtype="fp32",
)
RUNTIME = "transformers"
PROMPT_ID = "pp-doclayout-v3/detect"
PARAMS = {"threshold": 0.25}

CATEGORY = {
    "abstract": "Text",
    "algorithm": "Code",
    "aside_text": "Text",
    "chart": "Chart",
    "content": "Text",
    "doc_title": "Title",
    "figure_title": "Caption",
    "footer": "Page-footer",
    "footnote": "Footnote",
    "formula": "Formula",
    "formula_number": "Formula",
    "header": "Page-header",
    "image": "Picture",
    "number": "Page-footer",
    "paragraph_title": "Section-header",
    "reference": "Text",
    "reference_content": "List-item",
    "seal": "Picture",
    "table": "Table",
    "text": "Text",
    "vision_footnote": "Footnote",
}


def detect_call(page_hash: str, model: ModelRef = PP_DOCLAYOUT_V3) -> ModelCall:
    """Detection call for the page image whose `pixel_hash` is `page_hash`."""
    return ModelCall(
        model=model,
        runtime=RUNTIME,
        prompt_id=PROMPT_ID,
        prompt_sha="",
        params=dict(PARAMS),
        inputs=[page_hash],
    )


def read_detections(result: ModelResult, min_score: float = 0.35) -> list[Detection]:
    """Read raw PP-DocLayoutV3 detections mapped to Docling classes (filtering below `min_score`)."""
    out: list[Detection] = []
    for d in result.output.get("detections") or []:
        score = float(d["score"])
        if score < min_score:
            continue
        x0, y0, x1, y1 = (float(v) for v in d["box"])
        if x1 > x0 and y1 > y0:
            raw_lbl = str(d["label"]).strip().lower()
            out.append(Detection(label=CATEGORY.get(raw_lbl, "Text"), score=score, box=(x0, y0, x1, y1)))
    return out


def post_process_boxes(
    proc,
    outputs,
    threshold: float,
    target_sizes: list[tuple[int, int]],
) -> list[dict]:
    """Decode PP-DocLayoutV3 boxes, labels, scores, and reading order in pure PyTorch (no OpenCV needed)."""
    import torch

    boxes = outputs.pred_boxes
    logits = outputs.logits
    order_seqs = proc._get_order_seqs(outputs.order_logits)

    box_centers, box_dims = torch.split(boxes, 2, dim=-1)
    boxes = torch.cat([box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1)

    img_height, img_width = torch.as_tensor(target_sizes, dtype=boxes.dtype, device=boxes.device).unbind(1)
    scale_factor = torch.stack([img_width, img_height, img_width, img_height], dim=1)
    boxes = boxes * scale_factor[:, None, :]

    num_top_queries = logits.shape[1]
    num_classes = logits.shape[2]
    scores = torch.nn.functional.sigmoid(logits)
    scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
    labels = index % num_classes
    index = index // num_classes
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    order_seqs = order_seqs.gather(dim=1, index=index)

    results = []
    for score, label, box, order_seq in zip(scores, labels, boxes, order_seqs, strict=True):
        keep = score >= threshold
        kept_order = order_seq[keep]
        sorted_order, indices = torch.sort(kept_order)
        results.append(
            {
                "scores": score[keep][indices],
                "labels": label[keep][indices],
                "boxes": box[keep][indices],
                "order_seq": sorted_order,
            }
        )
    return results


def read_raw_detections(result: ModelResult, min_score: float = 0.35) -> list[Detection]:
    """Read raw PP-DocLayoutV3 detections keeping native fine-grained labels (`chart`, `seal`, `table`, ...)."""
    out: list[Detection] = []
    for d in result.output.get("detections") or []:
        score = float(d["score"])
        if score < min_score:
            continue
        x0, y0, x1, y1 = (float(v) for v in d["box"])
        if x1 > x0 and y1 > y0:
            out.append(Detection(label=str(d["label"]).strip().lower(), score=score, box=(x0, y0, x1, y1)))
    return out
