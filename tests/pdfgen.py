"""Minimal hand-built PDFs for tests: text at given positions, optional /Rotate and /CropBox."""

from __future__ import annotations


def _text_ops(texts, font: str = "F1") -> bytes:
    ops = []
    for x, y, size, text in texts:
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"BT /{font} {size} Tf {x} {y} Td ({esc}) Tj ET")
    return "\n".join(ops).encode("latin-1")


def make_pdf(pages: list[dict]) -> bytes:
    """pages: [{"size": (w, h), "rotate": 0, "crop": (x0, y0, x1, y1) | None, "texts": [(x, y, size, text)],
    "overlay": [(x, y, size, text)]}]. Text coordinates are PDF user space (origin bottom-left). "overlay" text
    is drawn the way a fill tool adds it: the page wrapped in q ... Q, then one more stream in its own font."""
    objects: list[bytes] = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    stream = lambda body: b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream"  # noqa: E731
    kids = []
    for page in pages:
        w, h = page.get("size", (612, 792))
        content = _text_ops(page.get("texts", []))
        page_no = len(objects) + 1
        kids.append(f"{page_no} 0 R")
        extra = ""
        if page.get("rotate"):
            extra += f" /Rotate {page['rotate']}"
        if page.get("crop"):
            extra += " /CropBox [{} {} {} {}]".format(*page["crop"])
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}]{extra} "
                f"/Resources << /Font << /F1 3 0 R{' /F2 ' + str(page_no + 5) + ' 0 R' if page.get('overlay') else ''} >> >> "
                f"/Contents {('[' + ' '.join(f'{page_no + k} 0 R' for k in range(1, 5)) + ']') if page.get('overlay') else f'{page_no + 1} 0 R'} >>"
            ).encode()
        )
        if page.get("overlay"):
            objects += [stream(b"q"), stream(content), stream(b"Q"), stream(_text_ops(page["overlay"], "F2"))]
            objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
        else:
            objects.append(stream(content))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def simple_pdf(text: str = "Hello world", **page) -> bytes:
    return make_pdf([{"texts": [(72, 700, 12, text)], **page}])
