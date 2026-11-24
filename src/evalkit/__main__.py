"""`python -m evalkit <command>`."""

from __future__ import annotations

import argparse
import sys


def _cmd_fetch(args: argparse.Namespace) -> int:
    from evalkit import fetch

    if args.what == "gt":
        print(fetch.fetch_ground_truth())
    elif args.what == "published":
        print(fetch.fetch_published(args.pipeline))
    else:
        print(fetch.fetch_ocria_code())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download pinned data into $DOCPARSE_DATA")
    f.add_argument("what", choices=["gt", "published", "ocria"])
    f.add_argument("--pipeline", default="ocria_1")
    f.set_defaults(func=_cmd_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
