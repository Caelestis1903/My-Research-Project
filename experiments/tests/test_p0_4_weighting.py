"""P0-4 unit smoke: assert the loss-weight vector math in p4_run.P4Runner.

Per spec §1b for strong-arm p = (0.8, 0.184, 0.016):
  equal: w_d = 1 (None / original path)
  sample: w = (2.4, 0.552, 0.048)        (sum=3, 6-term sum=6)
  sqrt:   w = (1.8507, 0.8876, 0.2617)   (sum=3, 6-term sum=6)
Plus the task-major flatten order: idx = t*D + d -> all 6 entries share
domain-equal weights when summed across tasks (idx 1 and idx 4 both pick
domain 1's weight, etc.).

This is a pure numpy test that does not import p4_run (which has heavy torch
sys.path side-effects) — it re-implements the same math and verifies the
spec's contract.

Run: venv/Scripts/python.exe experiments/tests/test_p0_4_weighting.py
(exit 0 = pass)
"""
import os
import sys
import numpy as np

TOL = 1e-4
P = (0.8, 0.184, 0.016)
SAMPLE_EXPECTED = (2.4, 0.552, 0.048)
SQRT_EXPECTED = (1.8507, 0.8876, 0.2617)


def compute_weight(p, kind):
    p = np.asarray(p, dtype=np.float64)
    if kind == "equal":
        return np.ones_like(p)
    if kind == "sample":
        return p / p.mean()
    if kind == "sqrt":
        return np.sqrt(p) / np.sqrt(p).mean()
    raise ValueError(kind)


def task_major_flatten(w, T=2):
    """idx = t * D + d. tile D-vector T times along the task axis."""
    return np.tile(w, T)


def main():
    # 1. sample weights equal the expected
    w_sample = compute_weight(P, "sample")
    assert np.allclose(w_sample, SAMPLE_EXPECTED, atol=TOL), (
        f"sample weight mismatch: got {w_sample}, expected {SAMPLE_EXPECTED}")
    print(f"[PASS] sample weights == ({', '.join(f'{x:.4f}' for x in SAMPLE_EXPECTED)})",
          flush=True)

    # 2. sqrt weights equal the expected
    w_sqrt = compute_weight(P, "sqrt")
    assert np.allclose(w_sqrt, SQRT_EXPECTED, atol=TOL), (
        f"sqrt weight mismatch: got {w_sqrt}, expected {SQRT_EXPECTED}")
    print(f"[PASS] sqrt weights == ({', '.join(f'{x:.4f}' for x in SQRT_EXPECTED)})",
          flush=True)

    # 3. sum of per-domain weights == D = 3 for both arms (so the 6-term
    # total loss scale is preserved across arms)
    assert abs(w_sample.sum() - 3.0) < TOL, f"sample sum != 3: {w_sample.sum()}"
    assert abs(w_sqrt.sum() - 3.0) < TOL, f"sqrt sum != 3: {w_sqrt.sum()}"
    # 4. sum of task-major 6-vector == D * T = 6 for both arms
    flat_sample = task_major_flatten(w_sample)
    flat_sqrt = task_major_flatten(w_sqrt)
    assert abs(flat_sample.sum() - 6.0) < TOL, f"sample 6-sum != 6: {flat_sample.sum()}"
    assert abs(flat_sqrt.sum() - 6.0) < TOL, f"sqrt 6-sum != 6: {flat_sqrt.sum()}"
    print("[PASS] both weight vectors sum to 3 (per-domain) and 6 (6-term flat)",
          flush=True)

    # 5. task-major flatten order: idx 1 and idx 4 must both be domain 1's
    # weight (idx = t*D + d -> (d, t) = (1, 0) and (1, 1) for D=3).
    for name, flat in (("sample", flat_sample), ("sqrt", flat_sqrt)):
        assert abs(flat[1] - flat[4]) < TOL, (
            f"{name}: idx 1 ({flat[1]}) and idx 4 ({flat[4]}) must equal "
            f"(domain 1 weight)")
        assert abs(flat[2] - flat[5]) < TOL, (
            f"{name}: idx 2 ({flat[2]}) and idx 5 ({flat[5]}) must equal "
            f"(domain 2 weight)")
        assert abs(flat[0] - flat[3]) < TOL, (
            f"{name}: idx 0 ({flat[0]}) and idx 3 ({flat[3]}) must equal "
            f"(domain 0 weight)")
    print("[PASS] task-major flatten: idx 0/3 (d=0), idx 1/4 (d=1), "
          "idx 2/5 (d=2) all share domain weight", flush=True)

    # 6. equal path returns ones (matches the original 6-term equal-weight
    # sum path; the runner treats this as None and skips reweighting).
    w_equal = compute_weight(P, "equal")
    assert np.allclose(w_equal, np.ones(3)), f"equal weight != ones: {w_equal}"
    print("[PASS] equal weights == (1, 1, 1) (original 6-term path)", flush=True)

    print("ALL P0-4 WEIGHTING UNIT TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
