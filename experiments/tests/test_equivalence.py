"""P4 acceptance test: with ALL components off, the runner must be bit-identical
to the main baseline (as-published). Also unit-tests the curriculum degenerate
property (rho=1 -> exact original parametrization).

Run:  venv/Scripts/python.exe experiments/tests/test_equivalence.py
(exit 0 = pass)
"""
import csv
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
sys.path.insert(0, BASE)

import torch  # noqa: E402

P0_LOG = os.path.join(BASE, "p0_seed2022.log")
EQ0_STREAM = os.path.join(BASE, "results", "p1", "loss_stream_eq0.txt")
TEST_EPOCHS = 3
STEPS = 20  # per epoch at batch 40960


def run_all_off(outdir):
    cmd = [PY, os.path.join(BASE, "p4_run.py"), "--seed", "2022", "--epoch", str(TEST_EPOCHS),
           "--tag", "alloff_test", "--outdir", outdir,
           "--loss_stream", os.path.join(outdir, "loss_stream_alloff.txt")]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(ROOT, "m3oe", "examples"))
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        raise SystemExit("all-off run failed")
    return res.stdout


def val_auc_stream(text):
    return [(int(m.group(1)), float(m.group(2)))
            for m in re.finditer(r"epoch:\s*(\d+)\s+validation:\s+auc:\s*([\d.]+)", text)]


def p0_val_auc_stream():
    out = []
    for line in open(P0_LOG, encoding="utf-8", errors="replace"):
        m = re.search(r"epoch:\s*(\d+)\s+validation:\s+auc:\s*([\d.]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out[:TEST_EPOCHS]


def test_1_all_off_matches_baseline():
    # NOTE (P0-1 review, 2026-09-08): (1)(2) compare against reference logs from
    # the P0/P1 GPU window. exp.md documents a known cross-window drift (same
    # seed, epoch-0 val AUC 0.71866 in the P0 window vs 0.71232 in later
    # windows; bit-reproducible within a window, cause: GPU desktop load ->
    # cuBLASLt kernel selection). In a non-P0 window these two asserts FAIL for
    # environmental reasons even with a bit-correct code path; the within-window
    # check is `drift_ml_orig_baseline_2022*.log` vs the current run.
    outdir = os.path.join(BASE, "results", "p4", "test_alloff")
    os.makedirs(outdir, exist_ok=True)
    text = run_all_off(outdir)
    a = p0_val_auc_stream()
    b = val_auc_stream(text)
    assert len(b) >= TEST_EPOCHS, f"expected {TEST_EPOCHS} epochs, got {len(b)}"
    assert a == b[:TEST_EPOCHS], f"val AUC streams differ:\n{a}\n{b[:TEST_EPOCHS]}"
    print(f"[PASS] (1) all-off val AUC stream == P0 for {TEST_EPOCHS} epochs (bit-exact)")

    s0 = [float(x) for x in open(EQ0_STREAM) if x.strip()][:TEST_EPOCHS * STEPS]
    s1 = [float(x) for x in open(os.path.join(outdir, "loss_stream_alloff.txt")) if x.strip()]
    assert len(s1) >= len(s0), f"loss stream too short: {len(s1)} vs {len(s0)}"
    max_diff = max(abs(x - y) for x, y in zip(s0, s1[:len(s0)]))
    assert max_diff == 0.0, f"per-step loss streams differ by {max_diff}"
    print(f"[PASS] (2) all-off per-step loss stream == P0 baseline for "
          f"{TEST_EPOCHS * STEPS} steps (bit-exact)")


def test_2_curriculum_degenerate():
    """rho(tau)=1 must make MDMTRecBeta.forward exactly MDMTRec.forward."""
    sys.path.insert(0, os.path.join(ROOT, "m3oe"))
    sys.path.insert(0, os.path.join(ROOT, "m3oe", "examples"))
    from MDMTRec.models.multi_domain import MDMTRec
    from MDMTRec.basic.features import SparseFeature, DenseFeature

    torch.manual_seed(0)
    feats = [DenseFeature(3)] + [SparseFeature(i, vocab_size=50, embed_dim=16) for i in (0, 1, 2, 4, 5, 6, 7)]

    common = dict(domain_num=3, task_num=2, fcn_dims=[512, 256, 256, 64], expert_num=4,
                  exp_d=1, exp_t=1, bal_d=1, bal_t=1)
    m1 = MDMTRec(feats, **common)
    m2 = __import__("p4_run").MDMTRecBeta(feats, domain_p=[0.21, 0.40, 0.39],
                                          kappa0=3.0, curriculum_beta_t=False, **common)
    # copy all shared weights
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    for k in sd1:
        if k in sd2 and sd1[k].shape == sd2[k].shape:
            sd2[k] = sd1[k].clone()
    # per-domain beta init must equal the original scalar 1.0
    sd2["_weight_bal_d"] = torch.ones(3)
    sd2["_weight_bal_t"] = torch.ones(2)
    m2.load_state_dict(sd2)
    m1.eval()
    m2.eval()

    x = torch.zeros((8, 64), dtype=torch.long)
    x[0] = torch.randint(0, 50, (64,))
    x[1] = torch.randint(0, 50, (64,))
    x[2] = torch.randint(0, 50, (64,))
    x[4] = torch.randint(0, 50, (64,))
    x[5] = torch.randint(0, 50, (64,))
    x[6] = torch.randint(0, 50, (64,))
    x[3] = torch.rand(64).mul(100).long()
    x[7] = torch.randint(0, 3, (64,))

    with torch.no_grad():
        m2.set_epoch(10_000, 50)  # forces rho = 1
        y1 = m1(x)
        y2 = m2(x)
    diff = (y1 - y2).abs().max().item()
    assert diff == 0.0, f"rho=1 forward differs by {diff}"
    print("[PASS] (3) curriculum degenerate: rho=1 forward is bit-identical to MDMTRec")

    # and with rho<1 it must differ (curriculum is actually doing something)
    with torch.no_grad():
        m2.set_epoch(0, 50)  # rho = 0
        y3 = m2(x)
    assert (y1 - y3).abs().max().item() > 1e-6, "rho=0 should change the output"
    print("[PASS] (4) rho=0 changes the forward (curriculum active)")

    ok, off = __import__("p4_run").curriculum_degenerate_check(0.7, (0.21, 0.40, 0.39), 10_000, 50)
    assert ok == 0.0
    print(f"[PASS] (5) analytic check: max offset at rho=1 is {ok}")


if __name__ == "__main__":
    test_2_curriculum_degenerate()
    test_1_all_off_matches_baseline()
    print("ALL EQUIVALENCE TESTS PASSED")
