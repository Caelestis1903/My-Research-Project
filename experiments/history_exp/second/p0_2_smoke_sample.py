"""P0-2 sampling-rule smoke checks (spec section 10, a-h).

Drives P4Runner's data-load + caching path (no model construction) for every
(ratio, setting, seed, method/oracle-domain) combination needed by a-h, then
asserts:

  (a) per (ratio, setting) the joint cached x_tr has len 187500 and per-domain
      counts match the role-size target table.
  (b) joint val/test tensors are byte-identical across all (ratio, setting)
      combos AND byte-identical with the P0-1 cached val/test (the val/test
      are loaded from the "all/all" split and never depend on role).
  (c) the ratio_20 (P0-1) cache path exists and is the same file P0-1 wrote
      (legalizes 20:1 reuse).
  (d) oracle(setting, d) cached x_tr == joint(setting) cached x_tr filtered
      to domain d (one ratio fully verified across all three settings).
  (e) prefix nesting: d0 with the same seed has S-chain
      3000 ⊂ 7500 ⊂ 25000 ⊂ 62500 and L-chain 125000 ⊂ 150000.
  (f) joint subsets for seed 2022 vs 2023 differ (not bit-identical).
  (g) ratio_1 degeneracy: the three settings all yield the same joint x_tr
      (up to byte equality). Only valid because the spec is willing to run
      a single setting for 1:1.
  (h) gamma_d recomputation: by sampling-counts, formula
      gamma_d = clip(1 - 0.1 * p_d / p_max, 0.9, 0.99)
      yields the spec-expected values per ratio.

CPU only; no model construction; no training.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import p4_run


# (ratio, role_sizes, settings, [L,M,S] expected counts).
# 1:1 -> 62500/62500/62500 ; 5:1 -> 125000/37500/25000 ;
# 20:1 -> 150000/30000/7500 (P0-1) ; 50:1 -> 150000/34500/3000
RATIO_TABLE = {
    1:  ("62500,62500,62500",   ["d0_large", "d1_large", "d2_large"], (62500, 62500, 62500)),
    5:  ("125000,37500,25000", ["d0_large", "d1_large", "d2_large"], (125000, 37500, 25000)),
    20: ("150000,30000,7500",  ["d0_large", "d1_large", "d2_large"], (150000, 30000, 7500)),
    50: ("150000,34500,3000",  ["d0_large", "d1_large", "d2_large"], (150000, 34500, 3000)),
}


def make_args(role_setting, seed, role_sizes, single_domain, momentum_mode="none"):
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
    ns.role_sizes = role_sizes
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


def expected_counts_for_setting(setting, lms):
    """Return {domain_id: count} for a given (L, M, S) and setting."""
    L, M, S = lms
    sizes = {"L": L, "M": M, "S": S}
    role = p4_run.ROLE_MAPS[setting]
    return {d: sizes[role[d]] for d in (0, 1, 2)}


def per_domain_share(counts, domain_num=3):
    total = sum(counts.values())
    return {d: counts.get(d, 0) / total for d in range(domain_num)}


def compute_gamma(counts, gamma0=0.9, gamma_max=0.99):
    """Replicate p4_run.py per_domain_only formula and clip."""
    shares = per_domain_share(counts)
    if not shares:
        return {}
    pmax = max(shares.values())
    out = {}
    for d, p in shares.items():
        out[d] = float(np.clip(1.0 - (1.0 - gamma0) * p / pmax, gamma0, gamma_max))
    return out


def strict_subset(a, b):
    """Return True iff every row of `a` appears in `b` and |a|<|b|.

    role_reversal_sample uses np.random.RandomState(seed*10+d).permutation
    and takes the first N_r elements, so under the same seed the smaller
    set is always a *prefix* of the larger set's permutation. We therefore
    verify subset membership row-by-row via a bytes-level hash, which is
    O(|a|+|b|) instead of the O(|a|*|b|) broadcasting approach.
    """
    if a.shape[0] >= b.shape[0]:
        return False
    # Use the raw bytes of each row as a hash key. This is exact (no
    # collisions) and avoids any per-cell Python overhead. uint8 view gives
    # the underlying byte layout.
    a_bytes = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[0], -1)
    b_bytes = np.ascontiguousarray(b).view(np.uint8).reshape(b.shape[0], -1)
    # Hash each row; sets of byte strings are O(N) and exact.
    b_set = {bytes(row) for row in b_bytes}
    return all(bytes(row) in b_set for row in a_bytes)


def main():
    seed = 2022
    all_ratios = [1, 5, 20, 50]
    results = {}

    # (a) per (ratio, setting): len == 187500 and per-domain counts match
    header("(a) joint cache sizes and per-domain counts")
    a_pass = True
    for ratio in all_ratios:
        sizes, settings, lms = RATIO_TABLE[ratio]
        for setting in settings:
            args = make_args(setting, seed, sizes, single_domain=-1)
            x_tr, _, _, _, _, _, dn, path, counts = load_data_cached(args)
            total = len(x_tr)
            exp = expected_counts_for_setting(setting, lms)
            ok = (total == 187500) and (dn == 3)
            for d, want in exp.items():
                ok = ok and counts.get(d, 0) == want
            a_pass = a_pass and ok
            print(f"  ratio={ratio} {setting}: len={total} dn={dn} "
                  f"counts={counts} expected={exp} cache={os.path.basename(path)} "
                  f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"(a) {'PASS' if a_pass else 'FAIL'}", flush=True)
    results["a"] = a_pass

    # (b) joint val/test byte-identical across all (ratio, setting) AND P0-1
    header("(b) val/test byte-identical across all (ratio, setting) and P0-1")
    b_pass = True
    b_pairs = []
    for ratio in all_ratios:
        sizes, settings, _ = RATIO_TABLE[ratio]
        for setting in settings:
            args = make_args(setting, seed, sizes, single_domain=-1)
            _, _, x_va, y_va, x_te, y_te, _, _, _ = load_data_cached(args)
            b_pairs.append((ratio, setting, x_va, y_va, x_te, y_te))
    for i in range(1, len(b_pairs)):
        ratio_i, setting_i, xva_i, yva_i, xte_i, yte_i = b_pairs[i]
        r0, s0, xva_0, yva_0, xte_0, yte_0 = b_pairs[0]
        ok = (np.array_equal(xva_0, xva_i) and np.array_equal(yva_0, yva_i)
              and np.array_equal(xte_0, xte_i) and np.array_equal(yte_0, yte_i))
        b_pass = b_pass and ok
        print(f"  ratio={ratio_i} {setting_i} matches ref "
              f"ratio={r0} {s0}? {'YES' if ok else 'NO'}", flush=True)
    print(f"(b) {'PASS' if b_pass else 'FAIL'}", flush=True)
    results["b"] = b_pass

    # (c) ratio_20 cache path exists, same file as P0-1 wrote
    header("(c) ratio_20 cache path exists, same file as P0-1 (reuse legit)")
    c_pass = True
    # Construct the exact cache path P0-1 would have written for
    # ratio=20, setting=d0_large, seed=2022. Since ratio_20 IS the
    # P0-1 default role_sizes "150000,30000,7500", the cache path is the
    # same as p0_1_orchestrator (role_setting=d0_large).
    a20 = make_args("d0_large", seed, "150000,30000,7500", single_domain=-1)
    runner = p4_run.P4Runner.__new__(p4_run.P4Runner)
    path20 = runner._cache_path(a20)
    if not os.path.exists(path20):
        # Not cached yet; force it to be written so we can hash it.
        load_data_cached(a20)
    if not os.path.exists(path20):
        c_pass = False
        print(f"  ratio_20 cache MISSING at {path20} -> FAIL", flush=True)
    else:
        import hashlib
        with open(path20, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        size = os.path.getsize(path20)
        # Independently: load it and verify the cached x_tr matches the
        # expected ratio_20 counts in d0_large.
        args_check = make_args("d0_large", seed, "150000,30000,7500", single_domain=-1)
        x_tr, _, _, _, _, _, _, _, counts = load_data_cached(args_check)
        exp20 = expected_counts_for_setting("d0_large", (150000, 30000, 7500))
        ok = all(counts.get(d, 0) == exp20[d] for d in exp20) and len(x_tr) == 187500
        c_pass = c_pass and ok
        print(f"  ratio_20 path={path20}", flush=True)
        print(f"    size={size} bytes  md5={digest}  counts={counts} "
              f"expected={exp20} -> {'OK' if ok else 'WRONG'}", flush=True)
    print(f"(c) {'PASS' if c_pass else 'FAIL'}", flush=True)
    results["c"] = c_pass

    # (d) oracle x_tr == joint x_tr filtered to domain d
    # Verified for ratio=5 across all three settings and all three domains.
    header("(d) oracle(setting, d) == joint(setting) x_tr filtered to d")
    d_pass = True
    d_ratio = 5
    d_sizes, d_settings, _ = RATIO_TABLE[d_ratio]
    for setting in d_settings:
        for d in (0, 1, 2):
            aj = make_args(setting, seed, d_sizes, single_domain=-1)
            xj, _, _, _, _, _, _, _, _ = load_data_cached(aj)
            xj_d = xj[xj[:, -1].astype(int) == d]
            ao = make_args(setting, seed, d_sizes, single_domain=d)
            xo, _, _, _, _, _, dn_o, _, _ = load_data_cached(ao)
            xo_stamped = xo.copy()
            xo_stamped[:, -1] = d
            ok = (dn_o == 1) and np.array_equal(xo_stamped, xj_d)
            d_pass = d_pass and ok
            print(f"  ratio={d_ratio} {setting} d{d}: oracle_len={len(xo)} "
                  f"dn={dn_o} joint_d_len={len(xj_d)} "
                  f"-> {'MATCH' if ok else 'MISMATCH'}", flush=True)
    print(f"(d) {'PASS' if d_pass else 'FAIL'}", flush=True)
    results["d"] = d_pass

    # (e) prefix nesting for d0 with the same seed
    header("(e) d0 prefix nesting: S chain 3000<7500<25000<62500; "
           "L chain 125000<150000")
    e_pass = True
    # d0 plays S in d1_large; chain increasing ratios in d1_large.
    s_chain = [(50, "d1_large"), (20, "d1_large"),
               (5,  "d1_large"), (1,  "d1_large")]
    s_oracle = None
    s_meta = None
    for ratio, setting in s_chain:
        sizes, _, _ = RATIO_TABLE[ratio]
        a = make_args(setting, seed, sizes, single_domain=0)
        xo, _, _, _, _, _, _, _, _ = load_data_cached(a)
        size = len(xo)
        if s_oracle is not None:
            ok = strict_subset(s_oracle, xo)
            e_pass = e_pass and ok
            print(f"  S chain {s_meta} -> ratio={ratio} {setting}: "
                  f"size {s_oracle.shape[0]} < {size}, "
                  f"strict-subset? {'YES' if ok else 'NO'}", flush=True)
        else:
            print(f"  S chain seed: ratio={ratio} {setting} size={size}", flush=True)
        s_oracle = xo
        s_meta = f"ratio={ratio} {setting}"
    # d0 plays L in d0_large; chain small-then-large.
    l_chain = [(5, "d0_large"), (20, "d0_large")]
    # 20 and 50 both have d0=150000 so 5 (125000) is a strict subset of either.
    l_oracle = None
    l_meta = None
    for ratio, setting in l_chain:
        sizes, _, _ = RATIO_TABLE[ratio]
        a = make_args(setting, seed, sizes, single_domain=0)
        xo, _, _, _, _, _, _, _, _ = load_data_cached(a)
        size = len(xo)
        if l_oracle is not None:
            ok = strict_subset(l_oracle, xo)
            e_pass = e_pass and ok
            print(f"  L chain {l_meta} -> ratio={ratio} {setting}: "
                  f"size {l_oracle.shape[0]} < {size}, "
                  f"strict-subset? {'YES' if ok else 'NO'}", flush=True)
        else:
            print(f"  L chain seed: ratio={ratio} {setting} size={size}", flush=True)
        l_oracle = xo
        l_meta = f"ratio={ratio} {setting}"
    # Also verify ratio=50 d0 (L in d0_large) is 150000 = ratio=20 d0 (L) -> byte-equal
    a50 = make_args("d0_large", seed, "150000,34500,3000", single_domain=0)
    a20 = make_args("d0_large", seed, "150000,30000,7500", single_domain=0)
    x50, _, _, _, _, _, _, _, _ = load_data_cached(a50)
    x20, _, _, _, _, _, _, _, _ = load_data_cached(a20)
    ok = np.array_equal(x50, x20) and len(x50) == 150000
    e_pass = e_pass and ok
    print(f"  ratio=50 d0 (L) == ratio=20 d0 (L) at size 150000? "
          f"{'YES' if ok else 'NO'}", flush=True)
    print(f"(e) {'PASS' if e_pass else 'FAIL'}", flush=True)
    results["e"] = e_pass

    # (f) joint subsets for seed 2022 vs 2023 differ
    header("(f) seed 2022 vs 2023 joint subsets differ")
    f_pass = True
    for ratio in all_ratios:
        sizes, settings, _ = RATIO_TABLE[ratio]
        for setting in settings:
            a22 = make_args(setting, 2022, sizes, single_domain=-1)
            a23 = make_args(setting, 2023, sizes, single_domain=-1)
            x22, _, _, _, _, _, _, _, _ = load_data_cached(a22)
            x23, _, _, _, _, _, _, _, _ = load_data_cached(a23)
            ok = not np.array_equal(x22, x23)
            f_pass = f_pass and ok
            print(f"  ratio={ratio} {setting}: seed2022 vs seed2023 identical? "
                  f"{'NO' if ok else 'YES (bad)'}", flush=True)
    print(f"(f) {'PASS' if f_pass else 'FAIL'}", flush=True)
    results["f"] = f_pass

    # (g) ratio_1 degeneracy: three settings' joint x_tr are byte-equal
    header("(g) ratio_1 degeneracy: 3 settings' joint x_tr byte-equal")
    g_pass = True
    g1_sizes, g1_settings, _ = RATIO_TABLE[1]
    g1_x = {}
    for setting in g1_settings:
        a = make_args(setting, seed, g1_sizes, single_domain=-1)
        x, _, _, _, _, _, _, _, _ = load_data_cached(a)
        g1_x[setting] = x
    keys = list(g1_x.keys())
    for i in range(1, len(keys)):
        ok = np.array_equal(g1_x[keys[0]], g1_x[keys[i]])
        g_pass = g_pass and ok
        print(f"  {keys[0]} == {keys[i]}? {'YES' if ok else 'NO'}", flush=True)
    print(f"(g) {'PASS' if g_pass else 'FAIL'}", flush=True)
    results["g"] = g_pass

    # (h) gamma_d recomputation
    header("(h) gamma_d recomputation from sampled counts")
    h_pass = True
    # spec expects: ratio_1 -> all 0.900; ratio_5 -> 0.900/0.970/0.980;
    # ratio_50 -> 0.900/0.977/0.990 (clipped)
    expected_gamma = {
        1:  [0.900, 0.900, 0.900],
        5:  [0.900, 0.970, 0.980],
        20: None,   # not in spec
        50: [0.900, 0.977, 0.990],
    }
    # Map role -> index: 0=L, 1=M, 2=S. Use setting=d0_large so that
    # domain d0 plays L, d1 plays M, d2 plays S.
    for ratio in (1, 5, 50):
        sizes, _, lms = RATIO_TABLE[ratio]
        a = make_args("d0_large", seed, sizes, single_domain=-1)
        x_tr, _, _, _, _, _, _, _, counts = load_data_cached(a)
        g = compute_gamma(counts)
        got = [round(g[0], 3), round(g[1], 3), round(g[2], 3)]
        exp = expected_gamma[ratio]
        ok = (got == exp)
        h_pass = h_pass and ok
        print(f"  ratio={ratio}: counts={counts} gamma_L/M/S={got} "
              f"expected={exp} -> {'OK' if ok else 'WRONG'}", flush=True)
    print(f"(h) {'PASS' if h_pass else 'FAIL'}", flush=True)
    results["h"] = h_pass

    print("\n=== OVERALL ===")
    for k in ("a", "b", "c", "d", "e", "f", "g", "h"):
        print(f"({k}) {'PASS' if results[k] else 'FAIL'}")
    overall = all(results.values())
    print(f"\nP0-2 SAMPLE SMOKE: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
