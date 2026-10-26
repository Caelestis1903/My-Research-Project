"""Stage R2 Hpost master switch.

Hpost is **disabled** for the duration of Stage R2.  This module exists so
that future stages can detect the switch and fail fast with a uniform
message.  No grid scan, no regularizer, no per-epoch branch lives here.

Future enablement requires an explicit authorization log entry, a new
design specification amendment, and a fresh Implementation Review pass.
"""

from __future__ import annotations

import sys


HPOST_ENABLED: bool = False


def run_hpost_scan(*args, **kwargs) -> None:
    """Placeholder.  Any actual call must raise NotImplementedError."""

    raise NotImplementedError(
        "Hpost is disabled in Stage R2.  See research_R2/R2_Design_Specification.md "
        "section 4.5.  Enabling requires re-authorization and review."
    )


def ensure_hpost_disabled() -> None:
    """Fail-fast guard for stages that load the runner indirectly."""

    if HPOST_ENABLED:
        raise RuntimeError("HPOST_ENABLED unexpectedly True after import")


def frozen_hpost_constant() -> bool:
    """A read-only accessor that should always return False in Stage R2."""

    return HPOST_ENABLED


# Sanity: assert at import time that HPOST_ENABLED equals False.
if HPOST_ENABLED is not False:
    raise RuntimeError("HPOST_ENABLED must remain False in Stage R2")


__all__ = ["HPOST_ENABLED", "run_hpost_scan", "ensure_hpost_disabled", "frozen_hpost_constant"]
