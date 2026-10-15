"""P0-1 sampling-rule smoke checks (spec section 7, a-f).

Drives P4Runner's data-load + caching path (no model construction) for every
(setting, seed, method-or-oracle-domain) combination needed by a-f, then
asserts:
  (a) joint cached x_tr has len 187500 and per-domain counts == role target
  (b) joint val/test tensors are byte-identical across the three settings
  (c) joint baseline and joint comp1 share the same cache file
  (d) oracle(setting X, d) cached x_tr == joint(setting X) cached x_tr
      filtered to domain d (array equality)
  (e) nesting: oracle d0 subset in setting B (d0=S) is a strict subset of
      oracle d0 in setting C (d0=M), which is a strict subset of oracle d0
      in setting A (d0=L), for the same seed
  (f) joint subsets for seed 2022 vs 2023 differ (not bit-identical)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import p4_run


def make_args(role_setting, seed, single_domain, momentum_mode="none"):
    ns = argparse.Namespace()
    ns.seed = seed
    ns.batch_size = 40960
    ns.epoch = 50
    ns.learning_rate = 1e-2
    ns.learning_rate_darts = 1e-3
    ns.weight_decay = 1e-5
    ns.expert_num = 4
    ns.device = "cpu"
    ns.dataset_path = "D:/CODE/phase0-3/m3oe/examples/data/ml-1m"
    ns.fix_optim_grouping = 1
    ns.momentum_mode = momentum_mode
    ns.gamma0 = 0.9
    ns.gamma_max = 0.99
    ns.gamma_source = "formula"
    ns.gamma_manual = ""
    ns.gamma_uniform = 0
    ns.curriculum_mode = "none"
    ns.kappa0 = 0.0
    ns.curriculum_beta_t = 0
    ns.arch_grad_ema = 0
    ns.gamma_a = 0.9
    ns.arch_loss_masked = 0
    ns.arch_update_every = 0
    ns.downsample_ratio = 0.0
    ns.downsample_seed = 777
    ns.fix_arch = 0.0
    ns.kuairand = 0
    ns.single_domain = single_domain
    ns.role_setting = role_setting
    ns.role_sizes = "150000,30000,7500"
    ns.run_dir = None
    ns.tag = "smoke"
    ns.outdir = "D:/CODE/phase0-3/experiments/tmp_p0_1_smoke/out"
    ns.loss_stream = None
    return ns


def load_data_cached(args):
    """Returns (x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num, cache_path)
    without constructing the model. Goes through the same _load_cache /
    _load_data / _store_cache path as P4Runner.__init__."""
    runner = p4_run.P4Runner.__new__(p4_run.P4Runner)
    runner.args = args
    runner.device = args.device
    path = runner._cache_path(args)
    cached = runner._load_cache(args)
    if cached is not None:
        (dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num) = cached
    else:
        (dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te,
         domain_num) = runner._load_data(args)
        runner._store_cache(args, dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va,
                            x_te, y_te, domain_num)
    dom = x_tr[:, -1].astype(int)
    counts = {int(d): int((dom == d).sum()) for d in np.unique(dom)}
    return x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num, path, counts


def header(name):
    print(f"\n=== {name} ===", flush=True)


def main():
    seed = 2022

    # (a) joint subset sizes
    header("(a) joint cache sizes and per-domain counts")
    expected = {"d0_large": {0: 150000, 1: 30000, 2: 7500},
                "d1_large": {0: 7500,  1: 150000, 2: 30000},
                "d2_large": {0: 30000, 1: 7500,  2: 150000}}
    a_pass = True
    for setting in ("d0_large", "d1_large", "d2_large"):
        args = make_args(setting, seed, single_domain=-1)
        x_tr, _, _, _, _, _, dn, path, counts = load_data_cached(args)
        total = len(x_tr)
        ok = (total == 187500) and (dn == 3)
        for d, want in expected[setting].items():
            ok = ok and counts.get(d, 0) == want
        a_pass = a_pass and ok
        print(f"  {setting}: len(x_tr)={total} domain_num={dn} counts={counts} "
              f"cache={os.path.basename(path)} -> {'PASS' if ok else 'FAIL'}",
              flush=True)
    print(f"(a) {'PASS' if a_pass else 'FAIL'}", flush=True)

    # (b) joint val/test byte-identical across settings
    header("(b) joint val/test byte-identical across settings")
    b_pass = True
    ref = None
    for setting in ("d0_large", "d1_large", "d2_large"):
        args = make_args(setting, seed, single_domain=-1)
        _, _, x_va, y_va, x_te, y_te, _, _, _ = load_data_cached(args)
        if ref is None:
            ref = (x_va, y_va, x_te, y_te)
        else:
            ok = (np.array_equal(x_va, ref[0]) and np.array_equal(y_va, ref[1])
                  and np.array_equal(x_te, ref[2]) and np.array_equal(y_te, ref[3]))
            b_pass = b_pass and ok
            print(f"  {setting} matches A? {'YES' if ok else 'NO'}", flush=True)
    print(f"(b) {'PASS' if b_pass else 'FAIL'}", flush=True)

    # (c) joint baseline and comp1 share cache file
    header("(c) joint baseline and comp1 share cache file path")
    c_pass = True
    for setting in ("d0_large", "d1_large", "d2_large"):
        a1 = make_args(setting, seed, single_domain=-1, momentum_mode="none")
        a2 = make_args(setting, seed, single_domain=-1, momentum_mode="per_domain_only")
        runner = p4_run.P4Runner.__new__(p4_run.P4Runner)
        path1 = runner._cache_path(a1)
        path2 = runner._cache_path(a2)
        ok = path1 == path2
        c_pass = c_pass and ok
        print(f"  {setting}: baseline={os.path.basename(path1)} "
              f"comp1={os.path.basename(path2)} -> {'SAME' if ok else 'DIFF'}",
              flush=True)
    print(f"(c) {'PASS' if c_pass else 'FAIL'}", flush=True)

    # (d) oracle x_tr == joint x_tr filtered to domain d
    header("(d) oracle(setting X, d) cached x_tr == joint(setting X) x_tr filtered to d")
    d_pass = True
    for setting in ("d0_large", "d1_large", "d2_large"):
        for d in (0, 1, 2):
            aj = make_args(setting, seed, single_domain=-1)
            xj, _, _, _, _, _, _, _, _ = load_data_cached(aj)
            xj_d = xj[xj[:, -1].astype(int) == d]
            ao = make_args(setting, seed, single_domain=d)
            xo, _, _, _, _, _, dn_o, _, _ = load_data_cached(ao)
            # oracle zeros the domain col -> re-stamp domain id = d for comparison
            xo_stamped = xo.copy()
            xo_stamped[:, -1] = d
            ok = (dn_o == 1) and np.array_equal(xo_stamped, xj_d)
            d_pass = d_pass and ok
            print(f"  {setting} d{d}: oracle_len={len(xo)} dn={dn_o} "
                  f"joint_d_len={len(xj_d)} -> {'MATCH' if ok else 'MISMATCH'}",
                  flush=True)
    print(f"(d) {'PASS' if d_pass else 'FAIL'}", flush=True)

    # (e) nesting: oracle d0 in B (S=7500) ⊂ oracle d0 in C (M=30000) ⊂ oracle d0 in A (L=150000)
    header("(e) nesting: d0 oracle B ⊂ C ⊂ A")
    e_pass = True
    nest_expected = [("d1_large", 7500,  "S"),   # d0 plays S in B (smallest)
                     ("d2_large", 30000, "M"),   # d0 plays M in C
                     ("d0_large", 150000, "L")]  # d0 plays L in A (largest)
    prev_oracle_x = None
    prev_setting = None
    for setting, expected_size, role_label in nest_expected:
        a = make_args(setting, seed, single_domain=0)
        xo, _, _, _, _, _, _, _, _ = load_data_cached(a)
        size = len(xo)
        ok = (size == expected_size)
        print(f"  d0 in {setting} ({role_label}): len={size} "
              f"expected={expected_size} -> {'OK' if ok else 'WRONG'}", flush=True)
        if prev_oracle_x is not None:
            common = int(np.sum(np.all(prev_oracle_x[:, None, :] == xo[None, :, :], axis=-1)))
            strict_ok = (common == prev_oracle_x.shape[0]) and (size > prev_oracle_x.shape[0])
            ok = ok and strict_ok
            print(f"    nesting {prev_setting} -> {setting}: "
                  f"{common}/{prev_oracle_x.shape[0]} matches, sizes "
                  f"{prev_oracle_x.shape[0]} < {size}: "
                  f"{'STRICT SUBSET' if strict_ok else 'NOT NESTED'}", flush=True)
        e_pass = e_pass and ok
        prev_oracle_x = xo
        prev_setting = setting
    print(f"(e) {'PASS' if e_pass else 'FAIL'}", flush=True)

    # (f) joint subsets for seed 2022 vs 2023 differ
    header("(f) seed 2022 vs 2023 joint subsets differ")
    f_pass = True
    for setting in ("d0_large", "d1_large", "d2_large"):
        a22 = make_args(setting, 2022, single_domain=-1)
        a23 = make_args(setting, 2023, single_domain=-1)
        x22, _, _, _, _, _, _, _, _ = load_data_cached(a22)
        x23, _, _, _, _, _, _, _, _ = load_data_cached(a23)
        ok = not np.array_equal(x22, x23)
        f_pass = f_pass and ok
        print(f"  {setting}: seed2022 vs seed2023 identical? "
              f"{'NO' if ok else 'YES (bad)'}", flush=True)
    print(f"(f) {'PASS' if f_pass else 'FAIL'}", flush=True)

    overall = a_pass and b_pass and c_pass and d_pass and e_pass and f_pass
    print("\n=== OVERALL ===")
    print(f"(a) {'PASS' if a_pass else 'FAIL'}")
    print(f"(b) {'PASS' if b_pass else 'FAIL'}")
    print(f"(c) {'PASS' if c_pass else 'FAIL'}")
    print(f"(d) {'PASS' if d_pass else 'FAIL'}")
    print(f"(e) {'PASS' if e_pass else 'FAIL'}")
    print(f"(f) {'PASS' if f_pass else 'FAIL'}")
    print(f"\nP0-1 SAMPLE SMOKE: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()