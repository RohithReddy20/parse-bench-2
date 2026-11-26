"""Page-image identities across machines.

A request key holds the hash of a page image's pixels. PDFium draws some fonts a shade differently on Linux and
macOS, so the same page can hash differently on the two (it looks the same). To let one model-call cache answer on
both, each machine lists the hashes of its page images by identity:

    uv run python -m evalkit.pixel_alias list  --out hashes.jsonl [--fold all|dev|check] [--workers N]
    uv run python -m evalkit.pixel_alias join  --here hashes_here.jsonl --canonical hashes_other.jsonl --out alias.json

`join` writes {hash here: canonical hash} for the images whose hashes differ; point `$DOCPARSE_PIXEL_ALIAS` at it on
the machine that should speak the other one's keys. Identity is (document sha256, page number, derivation):
the upright page, its 2048-px copy (Chandra, Tier-2 reads) and the Tier-2 table strips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evalkit import paths
from evalkit.explode import split_example_id
from evalkit.fold_eval import fold_ids
from evalkit.standin import source_file

DPI = 300


def _hashes(path: str) -> list[dict]:
    from docparse.models.adapters.qwen_tier2 import _cap_image, make_n_horizontal_strips
    from docparse.pdf.render import raw_pixel_hash, render_pages
    from docparse.pdf.upright import page_turn, turn_page

    data = Path(path).read_bytes()
    doc = hashlib.sha256(data).hexdigest()
    try:
        rendered = render_pages(data, DPI)
    except Exception:
        return []
    rows = []
    for index, stored in enumerate(rendered):
        image = turn_page(stored, page_turn(stored)).image
        kinds = {"full": image, "cap2048": _cap_image(image)}
        for n in (2, 3):
            for i, strip in enumerate(make_n_horizontal_strips(image, n=n, overlap=0.14)):
                kinds[f"strip{n}_{i}"] = _cap_image(strip)
        rows += [{"doc": doc, "page": index, "kind": k, "hash": raw_pixel_hash(im)} for k, im in kinds.items()]
    return rows


def cmd_list(args: argparse.Namespace) -> int:
    if args.paths:
        files = [ln.strip() for ln in args.paths.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        files = []
        for eid in fold_ids(args.fold):
            sub, stem = split_example_id(eid)
            src = source_file(paths.gt_dir(), sub, stem)
            if src is not None:
                files.append(str(src))
    n = 0
    with args.out.open("w", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(_hashes, files, chunksize=8):
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            n += 1
            if n % 500 == 0:
                print(n, flush=True)
    print(f"{n} documents -> {args.out}")
    return 0


def _load(path: Path) -> dict[tuple, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[(row["doc"], row["page"], row["kind"])] = row["hash"]
    return out


def cmd_join(args: argparse.Namespace) -> int:
    here, canon = _load(args.here), _load(args.canonical)
    alias = {h: canon[k] for k, h in here.items() if k in canon and canon[k] != h}
    args.out.write_text(json.dumps(alias), encoding="utf-8")
    pages = {k[:2] for k, h in here.items() if k in canon and canon[k] != h}
    print(f"identities here={len(here)} shared={len(set(here) & set(canon))} differing images={len(alias)} "
          f"on {len(pages)} pages -> {args.out}")  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalkit.pixel_alias")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", default="all", choices=["all", "dev", "check"])
    p.add_argument("--paths", type=Path, help="a file of source paths instead of a ParseBench fold")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("join")
    p.add_argument("--here", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return cmd_list(args) if args.cmd == "list" else cmd_join(args)


if __name__ == "__main__":
    raise SystemExit(main())
