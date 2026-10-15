"""Reviewer check: decision module behaviour on missing/invalid inputs.

- No runs -> valid=False, decision=NO, no crash.
- Counter gate: bumping the runner's TEST_PREDICTION_CALLS must make
  compute_r2a_decision refuse (RuntimeError).
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

import ao04_v12_runner as runner_mod  # noqa
import ao04_v12_decision as dec  # noqa


def main() -> int:
    ok = True

    res = dec.compute_r2a_decision("movielens", {})
    print("empty run_dirs -> valid={} decision={}".format(res.valid, res.decision))
    print("  note:", res.validity_note[:120])
    ok = ok and (not res.valid) and res.decision == "NO"

    bad = dec.compute_r2a_decision(
        "kuairand", {("B0", 2022): "/nonexistent/dir"})
    print("garbage run_dir -> valid={} decision={}".format(bad.valid, bad.decision))
    print("  note:", bad.validity_note[:160])
    ok = ok and (not bad.valid) and bad.decision == "NO"

    runner_mod._bump_test_counter()
    try:
        dec.compute_r2a_decision("movielens", {})
        print("counter gate: NOT raised (BAD)")
        ok = False
    except RuntimeError as exc:
        print("counter gate raised as expected:", str(exc)[:80])
    finally:
        runner_mod.TEST_PREDICTION_CALLS -= 1  # restore for a clean exit

    print("DECISION CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
