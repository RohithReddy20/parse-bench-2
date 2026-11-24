"""`python -m docparse cache plan --manifest DIR --cache DIR`: what a GPU session would run, before it runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docparse.models.cache import Cache
from docparse.models.manifest import Rate, format_plan, plan


def _rate(text: str) -> tuple[str, Rate]:
    repo, seconds, usd = text.rsplit(":", 2)
    return repo, Rate(float(seconds), float(usd))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docparse")
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache").add_subparsers(dest="action", required=True)
    p = cache.add_parser("plan", help="new calls per model and their estimated cost")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--rate", type=_rate, action="append", default=[], metavar="REPO:SECONDS_PER_CALL:USD_PER_HOUR")
    args = parser.parse_args(argv)
    print(format_plan(plan(args.manifest, Cache(args.cache), dict(args.rate))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
