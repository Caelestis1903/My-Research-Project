"""Static-tripwire check for the Stage R2 runner module.

This module contains the *names* that the training flow must not
reference.  Splitting them out of the runner module guarantees that a
``grep`` on ``ao04_v12_runner.py`` returns zero matches — the runner
itself does not see these identifiers.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

# Names that must NOT appear in the runner module body.
_FORBIDDEN_IDENTIFIER_LIST: Tuple[str, ...] = (
    "_evaluate_deferred_checkpoints",
    "write_outputs",
    "test_loader",
    "predictions_test",
)


def runner_static_tripwire(runner_path: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ao04_v12_runner.py"),
        *,
        forbidden: Sequence[str] = _FORBIDDEN_IDENTIFIER_LIST) -> List[str]:
    """Grep the runner module for forbidden identifiers."""

    if not os.path.exists(runner_path):
        return ["missing_file:{}".format(runner_path)]
    with open(runner_path, encoding="utf-8") as handle:
        source = handle.read()
    matches: List[str] = []
    for ident in forbidden:
        if ident in source:
            matches.append(ident)
    return matches


def run_static_check_and_assert_clean() -> Dict[str, object]:
    matches = runner_static_tripwire()
    if matches:
        raise AssertionError(
            "forbidden identifiers present in ao04_v12_runner.py: {}".format(matches)
        )
    return {"clean": True, "forbidden": list(_FORBIDDEN_IDENTIFIER_LIST)}


__all__ = ["runner_static_tripwire", "run_static_check_and_assert_clean",
           "_FORBIDDEN_IDENTIFIER_LIST"]
