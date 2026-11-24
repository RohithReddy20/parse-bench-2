"""OCRia-1's raw answer -> the markdown that is scored.

Ported verbatim from the ParseBench harness (Apache-2.0), so our own runs are post-processed exactly as the
published `ocria_1` pipeline's were:
  run-llama/ParseBench @ 6e4479ad9bda4347a81f7fe86dcf674071c352f9
  src/parse_bench/inference/providers/parse/ocria_1.py  (`ocria_postprocess` and its helpers)
`tests/base/test_postprocess.py` checks this copy against the pinned harness function.
"""

from __future__ import annotations

import re

_TABLE_SPLIT = re.compile(r"(<table\b.*?</table>)", re.IGNORECASE | re.DOTALL)
_PAGE_TAG_RE = re.compile(r"</?page(?:_full)?_transcription\s*>", re.IGNORECASE)
_TSV_THEAD = "[thead]"
_TSV_COLSPAN_RE = re.compile(r"^\[colspan=(\d+)\]")
_TSV_ROWSPAN = "ROW_SPAN_CONTINUE"


def _strip_markdown_wrapper(text: str) -> str:
    text = re.sub(r"^```(?:markdown)?\s*\n", "", text)
    return re.sub(r"\n```\s*$", "", text)


def _emphasis_segment_to_markdown(s: str) -> str:
    """HTML emphasis/heading tags -> the markdown forms the formatting rules credit (tables handled by the caller)."""
    s = re.sub(
        r"<h([1-6])\b[^>]*>(.*?)</h\1>",
        lambda m: f"\n{'#' * int(m.group(1))} {m.group(2).strip()}\n",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(r"</?(?:strong|b)\b[^>]*>", "**", s, flags=re.IGNORECASE)
    s = re.sub(r"</?(?:em|i)\b[^>]*>", "*", s, flags=re.IGNORECASE)
    s = re.sub(r"</?(?:s|strike|del)\b[^>]*>", "~~", s, flags=re.IGNORECASE)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</(?:p|div)>", "\n", s, flags=re.IGNORECASE)
    return re.sub(r"<(?:p|div)\b[^>]*>", "\n", s, flags=re.IGNORECASE)


def _html_emphasis_to_markdown(text: str) -> str:
    if not text:
        return text
    return "".join(
        part if i % 2 else _emphasis_segment_to_markdown(part) for i, part in enumerate(_TABLE_SPLIT.split(text))
    )


def _tsv_cell(cell: str) -> tuple[str, int]:
    text = cell[len(_TSV_THEAD) :] if cell.startswith(_TSV_THEAD) else cell
    colspan = 1
    m = _TSV_COLSPAN_RE.match(text)
    if m:
        colspan, text = int(m.group(1)), text[m.end() :]
    if text.strip() == "EMPTY":
        text = ""
    return text.strip(), colspan


def _tsv_rows_to_html(rows: list[list[str]]) -> str:
    header_rows = [r for r in rows if r and r[0].startswith(_TSV_THEAD)]
    body_rows = [r for r in rows if not (r and r[0].startswith(_TSV_THEAD))]
    emitted: list[list[dict]] = [[] for _ in body_rows]
    owner: dict[int, dict] = {}
    for ri, cells in enumerate(body_rows):
        col = 0
        for c in cells:
            if c == _TSV_ROWSPAN:
                o = owner.get(col)
                if o is not None:
                    o["rowspan"] += 1
                    col += o["colspan"]
                else:
                    emitted[ri].append({"text": "", "colspan": 1, "rowspan": 1, "tag": "td"})
                    col += 1
                continue
            text, colspan = _tsv_cell(c)
            d = {"text": text, "colspan": colspan, "rowspan": 1, "tag": "td"}
            emitted[ri].append(d)
            owner[col] = d
            for k in range(col + 1, col + colspan):
                owner.pop(k, None)
            col += colspan

    def tag(d: dict) -> str:
        attrs = (f' colspan="{d["colspan"]}"' if d["colspan"] > 1 else "") + (
            f' rowspan="{d["rowspan"]}"' if d["rowspan"] > 1 else ""
        )
        return f"<{d['tag']}{attrs}>{d['text']}</{d['tag']}>"

    parts = ["<table>"]
    if header_rows:
        parts.append("<thead>")
        for cells in header_rows:
            row = []
            for c in cells:
                text, colspan = _tsv_cell(c)
                row.append(tag({"text": text, "colspan": colspan, "rowspan": 1, "tag": "th"}))
            parts.append("<tr>" + "".join(row) + "</tr>")
        parts.append("</thead>")
    parts.append("<tbody>")
    parts += ["<tr>" + "".join(tag(d) for d in row) + "</tr>" for row in emitted]
    parts += ["</tbody>", "</table>"]
    return "\n".join(parts)


def _tsv_tables_to_html(md: str) -> str:
    """```tsv blocks (closed, or unclosed up to the next blank line) -> HTML tables."""
    if not md or "```tsv" not in md:
        return md
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        if lines[i].strip() == "```tsv":
            close = next((k for k in range(i + 1, len(lines)) if lines[k].strip() == "```"), None)
            if close is not None:
                body, nxt = lines[i + 1 : close], close + 1
            else:
                start = i + 1
                while start < len(lines) and lines[start].strip() == "":
                    start += 1
                end = next((k for k in range(start, len(lines)) if lines[k].strip() == ""), len(lines))
                body, nxt = lines[start:end], end
            rows = [ln.split("\t") for ln in body if ln.strip() != ""]
            if rows:
                out.append(_tsv_rows_to_html(rows))
            i = nxt
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _sanitize_html_attributes(text: str) -> str:
    """Quote bare HTML attribute values (as QwenProvider._sanitize_html_attributes)."""

    def _quote_attrs(match: re.Match) -> str:
        return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', match.group(0))

    return re.sub(r"<[^>]+>", _quote_attrs, text)


def _convert_md_tables_to_html(content: str) -> str:
    """Markdown pipe tables -> HTML <table> (as QwenProvider._convert_md_tables_to_html, without its aiohttp import)."""
    import markdown2

    def flush(table_lines: list[str], out: list[str]) -> None:
        if len(table_lines) >= 2:
            html = markdown2.markdown("\n".join(table_lines), extras=["tables"]).strip()
            out.extend([html] if "<table>" in html.lower() else table_lines)
        else:
            out.extend(table_lines)

    out: list[str] = []
    table: list[str] = []
    in_table = False
    for line in content.split("\n"):
        if "|" in line and line.strip().startswith("|"):
            table = [line] if not in_table else table + [line]
            in_table = True
        else:
            if in_table:
                flush(table, out)
                table, in_table = [], False
            out.append(line)
    if in_table:
        flush(table, out)
    return "\n".join(out)


def ocria_postprocess(raw: str) -> str:
    """OCRia-1's raw answer -> the markdown that is scored (the model card's post-processing)."""
    s = re.sub(r"<!--.*?-->", "", raw or "", flags=re.S).strip()  # analysis comments
    s = re.sub(r"<!--(?:(?!-->).)*$", "", s, flags=re.S).strip()  # an unclosed runaway comment
    s = _PAGE_TAG_RE.sub("", s).strip()  # <page_full_transcription> / <page_transcription> wrappers
    # page furniture: drop the tags, keep the text inside
    s = re.sub(r"</?(?:page_header|page_footer|page_number|watermark)\s*>", " ", s, flags=re.IGNORECASE)
    s = _tsv_tables_to_html(s)
    s = re.sub(r"(?:&nbsp;[ \t]*)+", " ", s)  # runaway &nbsp; padding
    s = re.sub(r"(?:\.[ \t]*){6,}", " ", s)  # runaway dot leaders
    s = _strip_markdown_wrapper(s).strip()  # the ```markdown page wrapper
    s = _html_emphasis_to_markdown(s)
    s = re.sub(r"<mark\b[^>]*>", "<mark>", s, flags=re.IGNORECASE)  # bare <mark>
    s = re.sub(r"<(page_header|page_footer|watermark)>.*?</\1\s*>\s*", "", s, flags=re.S | re.I)
    s = re.sub(r"</?page_number\s*>", "", s, flags=re.I)
    s = re.sub(r"!\[[^\]\n]{0,200}\]", "", s)  # image references
    s = re.sub(r"```mermaid\s*\n.*?\n```\s*", "", s, flags=re.S)  # diagrams drawn as mermaid
    s = re.sub(r"^[ \t]*<hr\s*/?>[ \t]*$\n?", "", s, flags=re.M | re.I)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if s:
        s = _convert_md_tables_to_html(s)
        s = _sanitize_html_attributes(s)
    return s
