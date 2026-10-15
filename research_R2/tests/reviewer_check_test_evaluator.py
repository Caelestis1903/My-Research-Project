"""Reviewer check: final_test_evaluator unlock gate + counter plumbing.

Usage (from project root):
    venv/Scripts/python.exe research_R2/tests/reviewer_check_test_evaluator.py <run_dir>

<run_dir> must be a completed (or 1-epoch smoke) run directory holding
selected_checkpoint.pth / final_checkpoint.pth / common_epoch_*.pth.

Verifies:
1. Missing unlock manifest -> FileNotFoundError.
2. With a (temporary) unlock manifest the evaluator runs end-to-end,
   writes per-run metrics_test.csv / predictions_test.npz into the run
   dir, and bumps ao04_v12_runner.TEST_PREDICTION_CALLS so the decision
   module subsequently refuses to publish.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

import ao04_v12_runner as runner_mod  # noqa
import ao04_v12_decision as dec  # noqa
from ao04_v12_test_evaluator import final_test_evaluator  # noqa


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: reviewer_check_test_evaluator.py <run_dir>")
        return 2
    run_dir = sys.argv[1]
    dataset = "movielens" if "movielens" in run_dir.lower() else "kuairand"
    setting = "d0_large" if dataset == "movielens" else "natural"
    ok = True

    # Work on a scratch copy of the run dir: a 1-epoch smoke run lacks the
    # formal common-epoch checkpoints (9/19/29/39/49), which the evaluator
    # correctly requires.  For this plumbing check we synthesize them from
    # final_checkpoint.pth inside the COPY — the goal is to exercise the
    # unlock gate, model construction, per-run artifact writing and the
    # counter plumbing, not to validate any number.
    import shutil
    from ao04_v12_protocol import COMMON_EPOCHS
    scratch = tempfile.mkdtemp(prefix="r2_eval_check_")
    work_dir = os.path.join(scratch, "runs", dataset, "smoke")
    shutil.copytree(run_dir, work_dir)
    final_ck = os.path.join(work_dir, "final_checkpoint.pth")
    for e in COMMON_EPOCHS:
        p = os.path.join(work_dir, "common_epoch_{}.pth".format(e))
        if not os.path.exists(p):
            shutil.copyfile(final_ck, p)
    run_dir = work_dir

    # 1. Unlock gate.
    try:
        final_test_evaluator(
            {("B0", 2022, "m3oe"): run_dir},
            unlock_manifest_path=os.path.join(tempfile.gettempdir(),
                                              "definitely_missing_unlock.json"),
            dataset=dataset, setting=setting)
        print("unlock gate: NOT raised (BAD)")
        ok = False
    except FileNotFoundError:
        print("unlock gate: FileNotFoundError as expected")

    # 2. Real pass with a temporary unlock manifest.
    assert runner_mod.TEST_PREDICTION_CALLS == 0
    unlock = os.path.join(tempfile.mkdtemp(prefix="r2_unlock_"),
                          "global_test_unlock.json")
    with open(unlock, "w", encoding="utf-8") as h:
        json.dump({"global_test_unlock": True}, h)
    artifacts = final_test_evaluator(
        {("B0", 2022, "m3oe"): run_dir},
        unlock_manifest_path=unlock, dataset=dataset, setting=setting)
    print("evaluator ran: csv={} npz={}".format(
        os.path.exists(artifacts.metrics_test_csv),
        os.path.exists(artifacts.predictions_test_npz)))
    per_run_csv = os.path.join(run_dir, "metrics_test.csv")
    per_run_npz = os.path.join(run_dir, "predictions_test.npz")
    per_run_ok = os.path.exists(per_run_csv) and os.path.exists(per_run_npz)
    print("per-run artifacts written:", per_run_ok)
    ok = ok and per_run_ok
    bumped = runner_mod.TEST_PREDICTION_CALLS
    print("TEST_PREDICTION_CALLS after evaluator:", bumped)
    ok = ok and bumped > 0

    # 3. Decision module must now refuse to publish.
    try:
        dec.compute_r2a_decision(dataset, {})
        print("decision after test eval: NOT raised (BAD)")
        ok = False
    except RuntimeError as exc:
        print("decision refuses after test eval:", str(exc)[:70])

    print("TEST EVALUATOR CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
