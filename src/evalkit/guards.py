"""Integrity guards on the evaluation side (design §8.4).

`forbid_answer_keys` proves the parser never reads an answer key: inside the context, opening any
`*.test.json` raises. Audit hooks cannot be removed, so one hook is installed per process and armed by a flag.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager

_state = threading.local()
_installed = False
_install_lock = threading.Lock()


class AnswerKeyAccess(RuntimeError):
    pass


def _hook(event: str, args: tuple) -> None:
    if event == "open" and getattr(_state, "armed", 0) and args:
        path = args[0]
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        if isinstance(path, str) and path.endswith(".test.json"):
            raise AnswerKeyAccess(f"answer key opened during parsing: {path}")


@contextmanager
def forbid_answer_keys():
    global _installed
    with _install_lock:
        if not _installed:
            sys.addaudithook(_hook)
            _installed = True
    _state.armed = getattr(_state, "armed", 0) + 1
    try:
        yield
    finally:
        _state.armed -= 1
