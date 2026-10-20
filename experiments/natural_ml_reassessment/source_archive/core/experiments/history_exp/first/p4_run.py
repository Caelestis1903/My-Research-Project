"""P4 runner: three new components for M3oE, all behind config switches.

Default configuration (every switch off) runs the ORIGINAL CTRTrainer code
paths — bit-identical to the main baseline (verified by
experiments/tests/test_equivalence.py).

Component 1 — domain-size-driven per-domain momentum   (--momentum_mode)
Component 2 — domain-conditioned curriculum on fusion weights (--curriculum_mode)
Component 3 — smoothing of architecture-param updates  (--arch_grad_ema /
              --arch_loss_masked / --arch_update_every)

Dataset knobs (used by P5): --downsample_ratio (max:min target), --domain
(single-domain oracle runs). Everything else mirrors run_movielens.py.

P1-3 (Matched Momentum Control) constants:
  - GAMMA_STAR = 1 - 1/17.6 ≈ 0.9432   (frequency-weighted mean τ* of the
    baseline comp1 schedule; per_domain EMA is updated EVERY training step
    on ALL D domains — see PerDomainMomentum.backward_accumulate, which
    loops D times every step — so the per-step update frequency f_d=1 for
    every d. With γ_d = [0.9, 0.977, 0.99] (gamma0=0.9, gamma_max=0.99,
    formula source) and f_d = [150000/187500, 34500/187500, 3000/187500]
    = [0.8, 0.184, 0.016], the per-domain half-lives are τ_d = 1/(1-γ_d)
    = [10, 43.478, 100] and the frequency-weighted mean τ* is
    Σ f_d·τ_d = 0.8·10 + 0.184·43.478 + 0.016·100 = 17.6 steps.
    Therefore γ* = 1 - 1/τ* = 0.9432 (the C-arm uniform γ value). The
    alternative τ* = Στ_d = 153.5 → γ* ≈ 0.9935 (spec §0a) only applies
    under an "only update m_d on step where domain d appears" schedule,
    which is NOT what backward_accumulate implements.
  - BETA1_STAR = 0.9432  (D-arm Adam β1 override on the shared param group;
    equivalent to running one Adam step with β1=γ* — Adam's first-moment
    EMA with the same decay — for a memory-matched Adam baseline that
    lacks comp1's domain-conditioned structure).
"""
import argparse
import csv
import os
import random
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe", "examples"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe", "examples"))

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from torch.utils.data import DataLoader

from MDMTRec.trainers import CTRTrainer
from MDMTRec.models.multi_domain import MDMTRec
from MDMTRec.basic.layers import EmbeddingLayer
from MDMTRec.basic.activation import activation_layer
from MDMTRec.utils.data import DataGenerator, TorchDataset
from run_movielens import get_movielens_data_rank_multidomain
from p1_run import setup_seed, shared_param_names, ARCH_PARAMS

# P1-3: matched-momentum constants (see module docstring for derivation).
# Both arms reuse this single value because the design equates the EMA
# half-life with the Adam first-moment half-life (γ* = β1*).
GAMMA_STAR = 1.0 - 1.0 / 17.6   # ≈ 0.9431818182
BETA1_STAR = GAMMA_STAR          # exact same value
TAU_STAR = 17.6                  # spec §0a derivation, see docstring

# ----------------------------------------------------------------------------
# Component 2: model variant with per-domain / per-task fusion weights
# ----------------------------------------------------------------------------
class MDMTRecBeta(MDMTRec):
    """Replaces the scalar bal_d / bal_t with per-domain (D) / per-task (T)
    weights and applies the domain-conditioned curriculum offset in forward.

    Key property: with rho(epoch)=1 (or kappa==0) the forward is numerically
    identical to MDMTRec.forward — the curriculum only reshapes the training
    trajectory, never the model capacity (unit-tested in
    experiments/tests/test_equivalence.py::test_curriculum_degenerate).
    """

    def __init__(self, *a, domain_p=None, kappa0=0.0, curriculum_beta_t=False, **kw):
        super().__init__(*a, **kw)
        assert domain_p is not None
        self.domain_p = np.asarray(domain_p, dtype=np.float64)
        self.kappa0 = float(kappa0)
        self.curriculum_beta_t = bool(curriculum_beta_t)
        self.domain_num_c = kw["domain_num"]
        self.task_num_c = kw["task_num"]
        D, T = self.domain_num_c, self.task_num_c
        # per-domain / per-task learnable raw weights (e_beta), init = 1.0 like
        # the original scalars (sigmoid(1) ~ 0.7311)
        self._weight_bal_d = torch.nn.Parameter(torch.ones(D), requires_grad=True)
        self._weight_bal_t = torch.nn.Parameter(torch.ones(T), requires_grad=True)
        # freeze the original Weights modules so no stale scalars are trained
        self._orig_weight_bal_d = self._orig_weight_bal_t = None
        self.tau = 1.0  # no-op, Weights modules replaced
        self.register_buffer("_curriculum_rho", torch.tensor(1.0))
        pmax, pmin = self.domain_p.max(), self.domain_p.min()
        if pmax > pmin:
            log_ratio = np.log(pmax / self.domain_p) / np.log(pmax / pmin)
            self.kappa_d = self.kappa0 * log_ratio
        else:
            self.kappa_d = np.zeros(D)
        # named_parameters names for the optimizer grouping fix
        self._beta_param_names = {"_weight_bal_d", "_weight_bal_t"}

    def set_epoch(self, epoch, total_epochs):
        T_c = max(1.0, total_epochs / 2.0)
        self._curriculum_rho.fill_(min(1.0, epoch / T_c))

    def _kappa_t_offset(self):
        return 0.0  # beta_t curriculum off in this implementation switch set

    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        mask = []
        for d in range(self.domain_num):
            mask.append((domain_id == d).cpu())  # faithful to model.py:145

        input_emb = self.embedding(x, self.features, squeeze_dim=True)

        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight)) + _bias + self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb) + skip

        gate_value = [self.gate[i](emb.detach()).unsqueeze(1)
                      for i in range(self.task_num * self.domain_num)]

        out = [self.expert[i](emb) for i in range(self.expert_num)]
        domain_exp_out = [self.domain_expert[i](emb) for i in range(self.domain_num)]
        task_exp_out = [self.task_expert[i](emb) for i in range(self.task_num)]
        fea = torch.cat([o.unsqueeze(1) for o in out], dim=1)
        domain_fea = torch.cat([o.unsqueeze(1) for o in domain_exp_out], dim=1)
        task_fea = torch.cat([o.unsqueeze(1) for o in task_exp_out], dim=1)

        rho = float(self._curriculum_rho)
        # curriculum offset on the RAW weight: sigmoid(e - kappa_d*(1-rho)).
        # rho=1 => offset 0 => exactly the original parametrization.
        off_np = (self.kappa_d * (1.0 - rho)).astype(np.float32)
        off = torch.as_tensor(off_np, device=_device)
        bal_d = torch.sigmoid(self._weight_bal_d - off)
        if self.curriculum_beta_t:
            bt_off = torch.as_tensor(np.float32(self.kappa0 * (1.0 - rho)), device=_device)
            bal_t = torch.sigmoid(self._weight_bal_t - bt_off)
        else:
            bal_t = torch.sigmoid(self._weight_bal_t)

        weighted_domain_fea = []
        for i in range(self.domain_num):
            temp_ = bal_d[i] * domain_fea[:, i, :]
            for j in range(self.domain_num):
                if j != i:
                    temp_ += (1 - bal_d[i]) / (self.domain_num - 1) * domain_fea[:, j, :]
            weighted_domain_fea.append(temp_)
        weighted_task_fea = []
        for i in range(self.task_num):
            temp_ = bal_t[i] * task_fea[:, i, :]
            for j in range(self.task_num):
                if j != i:
                    temp_ += (1 - bal_t[i]) / (self.task_num - 1) * task_fea[:, j, :]
            weighted_task_fea.append(temp_)

        fused_fea = [torch.bmm(gate_value[i], fea).squeeze(1) +
                     self._weight_exp_d() * weighted_domain_fea[i % self.domain_num] +
                     self._weight_exp_t() * weighted_task_fea[i // self.domain_num]
                     for i in range(self.task_num * self.domain_num)]

        results = [torch.sigmoid(self.tower[i](fused_fea[i]).squeeze(1))
                   for i in range(self.task_num * self.domain_num)]

        output = torch.zeros((len(results[0]), len(results)))
        for i in range(len(results)):
            output[:, i] = results[i]
        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t * self.domain_num + d], result[:, t])
        return result


def curriculum_degenerate_check(kappa0=0.7, p=(0.21, 0.40, 0.39), epoch=1000, total=50):
    """rho=1 must make the curriculum offset exactly zero."""
    import numpy as np
    pmax, pmin = max(p), min(p)
    if pmax > pmin:
        kappa_d = kappa0 * (np.log(pmax / np.array(p)) / np.log(pmax / pmin))
    else:
        kappa_d = np.zeros(len(p))
    rho = min(1.0, epoch / (total / 2.0))
    offset = kappa_d * (1.0 - rho)
    return float(np.max(np.abs(offset))), offset


# ----------------------------------------------------------------------------
# Component 1: per-domain momentum trainer
# ----------------------------------------------------------------------------

def arch_named(model):
    """Map canonical arch-param names to actual parameters (handles both the
    original Weights modules (`_weight_x.deep_weights`) and replaced plain
    Parameters (`_weight_x`) as used by MDMTRecBeta)."""
    named = dict(model.named_parameters())
    out = {}
    for pn in ARCH_PARAMS:
        out[pn] = named.get(pn, named.get(f"{pn}.deep_weights"))
    return out


class PerDomainMomentum:
    """Maintains per-domain EMAs of shared-parameter gradients and replaces
    the shared params' .grad with sum_d m_d before optimizer.step().
    Private params (towers, gates, slot weights, arch scalars) take the
    gradient accumulated by the same D backwards (each private param belongs
    to exactly one domain's graph; arch scalars appear in all D and thus get
    the correct full sum)."""

    def __init__(self, model, domain_num, gamma0=0.9, gamma_max=0.99, domain_p=None,
                 device="cuda:0", gamma_source="formula", gamma_manual="",
                 gamma_uniform=None):
        self.model = model
        self.domain_num = domain_num
        self.domain_p = np.asarray(domain_p, dtype=np.float64)
        pmax = self.domain_p.max()
        self.gamma_d = np.clip(1.0 - (1.0 - gamma0) * self.domain_p / pmax, gamma0, gamma_max)
        if gamma_source == "uniform":
            # P1-3 C arm: force every domain to the same gamma value (gamma*).
            # Frequencies (f_d) and per-domain gammas (γ_d) collapse to a single
            # γ*, so the residual comp1 gain becomes attributable to the
            # matched momentum-intensity hypothesis (H1) vs the domain-
            # conditioned structure hypothesis (H2).
            if gamma_uniform is None:
                raise ValueError("[p4][ERROR] --gamma_source=uniform requires "
                                 "--gamma_uniform <float>")
            self.gamma_d = np.full(domain_num, float(gamma_uniform))
        elif gamma_uniform:
            # Legacy P6 bool semantic: any TRUTHY value collapses all
            # domains to gamma0 (preserves the existing --gamma_uniform 1
            # path). The pre-P1-3 call site passed bool(args.gamma_uniform),
            # so 0/0.0/None all meant "off"; keep that truthiness contract
            # here (the float itself is only honoured in the uniform branch).
            self.gamma_d = np.full(domain_num, gamma0)
        elif gamma_source == "manual":
            vals = np.array([float(x) for x in gamma_manual.split(",") if x != ""])
            assert len(vals) == domain_num, f"--gamma_manual needs {domain_num} values"
            self.gamma_d = vals
        elif gamma_source == "snr":
            # P1 D0.4 measured per-domain SNR (batch=40960, epoch-mean over 5 seeds):
            snr = np.array([0.5085, 0.5218, 0.5584])[:domain_num]
            snr = np.resize(snr, domain_num) if domain_num > 3 else snr
            ratio = snr / snr.max()
            self.gamma_d = np.clip(1.0 - (1.0 - gamma0) * ratio, gamma0, gamma_max)
        self.names = shared_param_names(model)
        self.named = dict(model.named_parameters())
        self.m = {d: {n: torch.zeros_like(self.named[n]) for n in self.names}
                  for d in range(domain_num)}
        print(f"[comp1] gamma_d = {np.round(self.gamma_d, 4).tolist()}", flush=True)

    def backward_accumulate(self, logloss_terms, task_num, terms_w=None):
        """D backwards (retain_graph=True). Per-domain shared grads are
        snapshotted after each backward and shared grads are zeroed in between
        (shared params appear in every domain's graph). Private params
        (towers/gates/slot weights) appear in exactly one L_d each, and the 4
        arch scalars appear in all D, so leaving them accumulating across the
        loop yields exactly their full-loss gradient.

        P0-5: optional terms_w (length D*T task-major) multiplies each term
        before the per-domain L_d is formed. The per-step gradient for any
        parameter that touches a single L_d (private params, per-domain arch)
        becomes w_d * grad(L_d); for parameters that touch all D (shared
        params, shared arch scalars) the accumulated gradient becomes
        sum_d w_d * grad(L_d^shared). Both equal the gradient of the
        per-(d, t)-weighted 6-term loss, exactly-once per parameter group.
        Passing terms_w=None keeps the legacy equal-weight path bit-identical.
        """
        model = self.model
        model.zero_grad()
        gvecs = []
        for d in range(self.domain_num):
            L_d = logloss_terms[d] + logloss_terms[self.domain_num + d]
            if terms_w is not None:
                L_d = terms_w[d] * logloss_terms[d] + \
                      terms_w[self.domain_num + d] * logloss_terms[self.domain_num + d]
            L_d.backward(retain_graph=(d < self.domain_num - 1))
            gvecs.append({n: self.named[n].grad.detach().clone()
                          if self.named[n].grad is not None
                          else torch.zeros_like(self.named[n]) for n in self.names})
            for n in self.names:  # zero ONLY shared params between domains
                self.named[n].grad = None
        # EMA update per domain
        for d in range(self.domain_num):
            g = self.gamma_d[d]
            for n in self.names:
                self.m[d][n].mul_(g).add_(gvecs[d][n], alpha=1.0 - g)
        return gvecs

    def apply_grads(self):
        """Replace shared grads with sum_d m_d (drop-in for the equal-weight
        6-term loss whose shared gradient is sum_d g_d)."""
        model = self.model
        # private params already hold accumulated grad from the D backwards
        for n in self.names:
            p = self.named[n]
            total = None
            for d in range(self.domain_num):
                md = self.m[d][n]
                total = md.clone() if total is None else total + md
            p.grad = total


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--batch_size", type=int, default=40960)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=1e-2)
    ap.add_argument("--learning_rate_darts", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--expert_num", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset_path", default="D:/CODE/phase0-3/m3oe/examples/data/ml-1m")
    ap.add_argument("--fix_optim_grouping", type=int, default=0)
    # component 1
    ap.add_argument("--momentum_mode", choices=["none", "per_domain_only", "per_domain_stacked"],
                    default="none")
    ap.add_argument("--gamma0", type=float, default=0.9)
    ap.add_argument("--gamma_max", type=float, default=0.99)
    ap.add_argument("--gamma_source", choices=["formula", "manual", "snr", "uniform"], default="formula",
                    help="P6 ablation: how gamma_d is set. formula = 1-(1-g0)*p_d/p_max; "
                         "manual = --gamma_manual list; snr = replace p_d/p_max by "
                         "SNR_d/SNR_max measured in P1 D0.4. P1-3 C-arm: uniform = "
                         "set every domain to --gamma_uniform (matched momentum).")
    ap.add_argument("--gamma_manual", type=str, default="",
                    help="comma-separated gamma_d per domain, used with --gamma_source manual")
    ap.add_argument("--gamma_uniform", type=float, default=None,
                    help="P1-3 C-arm: gamma* value used when --gamma_source=uniform. "
                         "Legacy P6 bool semantic: any non-None value under "
                         "--gamma_source!=uniform collapses all gamma_d to gamma0.")
    # P1-3 D-arm: override Adam β1 for the shared param group.
    ap.add_argument("--beta1_shared", type=float, default=None,
                    help="P1-3 D-arm: override Adam β1 for shared params (matched-"
                         "momentum Adam ablation). Default None preserves the "
                         "default β1=0.9 (or 0.0 when comp1 is active).")
    # component 2
    ap.add_argument("--curriculum_mode", choices=["none", "domain_conditioned"], default="none")
    ap.add_argument("--kappa0", type=float, default=0.0)
    ap.add_argument("--curriculum_beta_t", type=int, default=0)
    # component 3
    ap.add_argument("--arch_grad_ema", type=int, default=0)
    ap.add_argument("--gamma_a", type=float, default=0.9)
    ap.add_argument("--arch_loss_masked", type=int, default=0)
    ap.add_argument("--arch_update_every", type=int, default=0,
                    help="arch update every N steps; 0 = once per epoch (as published)")
    # dataset knobs
    ap.add_argument("--downsample_ratio", type=float, default=0.0,
                    help="target max:min imbalance, e.g. 30.0; 0 = no downsampling")
    ap.add_argument("--downsample_seed", type=int, default=777)
    ap.add_argument("--fix_arch", type=float, default=0.0,
                    help="freeze the 4 arch params at this sigmoid value (requires_grad=False); 0 = off")
    ap.add_argument("--kuairand", type=int, default=0,
                    help="load KuaiRand via examples/run_kuaishou.py preprocessing")
    ap.add_argument("--single_domain", type=int, default=-1,
                    help=">=0: train a single-domain oracle on domain d only (P5 oracle baseline)")
    # P0-3: backbone switch. Default keeps M3OE MDMTRec; mmoe loads Backbone
    # from p7_models (same trainer / data path, only model class swaps).
    ap.add_argument("--backbone", choices=["mdmoe", "mmoe"], default="mdmoe",
                    help="P0-3 backbone switch: mdmoe = M3OE MDMTRec (default, "
                         "bit-identical to P0-1/P0-2); mmoe = Backbone('mmoe') "
                         "from p7_models.py")
    # P0-1: fixed total sample size + domain size role reversal
    ap.add_argument("--role_setting", default="",
                    choices=["", "d0_large", "d1_large", "d2_large"],
                    help="P0-1 role-reversal setting; '' = no role reversal (default)")
    ap.add_argument("--role_sizes", type=str, default="150000,30000,7500",
                    help="comma-separated L,M,S sample counts used by role_reversal_sample")
    ap.add_argument("--run_dir", default=None,
                    help="P0-1: write metrics_epoch.csv + metrics_summary.csv to this dir")
    # P0-2: stage / experiment labels propagated to the per-run metrics CSVs
    # (replaces the hardcoded "P0-1" / "role_reversal" in write_p0_1_metrics).
    ap.add_argument("--stage_label", default="P0-1",
                    help="value of the experiment_stage column in metrics CSVs (default P0-1)")
    ap.add_argument("--experiment_label", default="role_reversal",
                    help="value of the experiment_name column in metrics CSVs (default role_reversal)")
    # P1-2: per-epoch gradient norm logging. Off by default (smoke tests
    # confirm the no-log path is bit-identical). When on, writes
    # grad_norm_epoch.csv under --run_dir with one row per
    # (epoch, domain-label in {shared, d0, d1, d2}).
    ap.add_argument("--log_grad_norm", type=int, default=0,
                    help="P1-2: write grad_norm_epoch.csv with per-epoch L2 "
                         "norms of shared-param gradients. Last training "
                         "step grad + D independent BCE_d backprops on a "
                         "fixed mini-batch (x_tr[0:bs]).")
    # P0-4: per-(domain, task) loss reweighting (only affects the
    # momentum_mode=='none' 6-term equal-weight branch; equal = original path).
    ap.add_argument("--loss_weighting", choices=["equal", "sample", "sqrt"], default="equal",
                    help="P0-4 loss reweighting of the 6 (domain x task) terms: "
                         "equal = w_d = 1 (original); sample = w_d = p_d / p_mean "
                         "(sample-proportional, sum to D); sqrt = w_d = sqrt(p_d) "
                         "/ sqrt(p_mean). Mutex with --single_domain>=0. P0-5 "
                         "narrow exception: equal and sample are also allowed "
                         "with --momentum_mode per_domain_only (comp1). sqrt "
                         "remains mutex with comp1.")
    # misc
    ap.add_argument("--tag", default="p4")
    ap.add_argument("--outdir", default="D:/CODE/phase0-3/experiments/results/p4")
    ap.add_argument("--loss_stream", default=None)
    return ap


def downsample_train(x_train, y_train, ratio, seed, domain_col=-1):
    """Controllable imbalance by downsampling the TRAINING split only
    (val/test untouched). Deterministic given (seed, ratio).

    - If ratio >= current max:min imbalance: keep the max domain intact and
      shrink every non-max domain by one common factor f=(n_max/ratio)/n_min,
      preserving the relative sizes of the smaller domains.
    - If ratio < current imbalance (only the 1:1 tier): shrinking the max
      domain cannot raise its share, so instead the LARGER domains are shrunk
      to the min-domain size (min intact) -> all domains ~equal.
    """
    if ratio <= 0:
        return x_train, y_train
    rng = np.random.RandomState(seed)
    dom = x_train[:, domain_col].astype(int)
    counts = sorted(((d, int((dom == d).sum())) for d in np.unique(dom)),
                    key=lambda t: -t[1])
    n_max, n_min = counts[0][1], counts[-1][1]
    keep = np.ones(len(dom), dtype=bool)

    def shrink(d, target):
        idx = np.where(dom == d)[0]
        target = max(1, min(n_of[d], int(round(target))))
        drop = rng.choice(idx, size=n_of[d] - target, replace=False)
        keep[drop] = False

    n_of = dict(counts)
    if ratio >= n_max / n_min:
        f = (n_max / ratio) / n_min  # <= 1
        for d, n in counts[1:]:
            shrink(d, n * f)
    else:
        for d, n in counts[:-1]:  # tier below current imbalance: equalize down to min
            shrink(d, n_min)
    return x_train[keep], y_train[keep]


# ----------------------------------------------------------------------------
# P0-1: fixed total sample size + domain size role reversal
# ----------------------------------------------------------------------------
ROLE_MAPS = {"d0_large": {0: "L", 1: "M", 2: "S"},
             "d1_large": {0: "S", 1: "L", 2: "M"},
             "d2_large": {0: "M", 1: "S", 2: "L"}}


def parse_role_sizes(s):
    """Parse 'L,M,S' comma-separated string into {'L':N_L,'M':N_M,'S':N_S}."""
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 3, f"--role_sizes needs 3 comma-separated values, got {s!r}"
    return {"L": parts[0], "M": parts[1], "S": parts[2]}


def role_reversal_sample(x_train, y_train, setting, seed, sizes):
    """P0-1 fixed-total role-reversal subsample.

    pool_d = { train row i : domain_indicator_i == d } (preserves row order)
    perm   = np.random.RandomState(seed*10 + d).permutation(len(pool_d))
    subset(d, role r) = pool_d[perm[:sizes[r]]]

    Returns (x_train[keep], y_train[keep]); the boolean mask preserves the
    original row order so any consumer that depends on it (e.g. domain shares)
    stays consistent with the spec's sampling rule.
    """
    roles = ROLE_MAPS[setting]
    dom = x_train[:, -1].astype(int)
    keep = np.zeros(len(x_train), dtype=bool)
    for d, role in roles.items():
        pool_idx = np.where(dom == d)[0]
        N_r = sizes[role]
        perm = np.random.RandomState(seed * 10 + d).permutation(len(pool_idx))
        keep[pool_idx[perm[:N_r]]] = True
    return x_train[keep], y_train[keep]


class P4Runner:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        setup_seed(args.seed)
        cached = self._load_cache(args)
        if cached is not None:
            (dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num) = cached
        else:
            dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
                self._load_data(args)
            self._store_cache(args, dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va,
                              x_te, y_te, domain_num)
        self.domain_num, self.task_num = domain_num, 2
        # P1-2: keep references for the fixed mini-batch used by grad norm
        # logging. The training tensor itself is NOT consumed here (the
        # orchestrator never re-runs this dataset; the DataLoader remains
        # the source of truth for the per-step sampling); only a snapshot
        # slice is pulled at grad-log time, so x_tr storage does not change
        # the training trajectory.
        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_te, self.y_te = x_te, y_te
        dom = x_tr[:, -1].astype(int)
        counts = {int(d): int((dom == d).sum()) for d in np.unique(dom)}
        tot = sum(counts.values())
        self.train_domain_p = {d: counts[d] / tot for d in counts}
        self.train_counts = counts  # P0-1: per-domain integer counts in training set
        print(f"[p4] train domain share: {self.train_domain_p}", flush=True)
        # P0-4: per-(domain, task) loss weighting vector. equal / single-domain
        # -> None (run the original 6-term sum). sample / sqrt -> length D*T
        # task-major flat vector (idx = t * D + d), normalised so sum = D = 3.
        # That keeps the 6-term total loss scale (= sum_w) constant across
        # the three weighting arms, isolating weight distribution from the
        # overall lr / step-size scale.
        if args.loss_weighting == "equal":
            self.loss_w = None
        elif args.single_domain >= 0:
            # single-domain oracle: 6-term loss collapses to 2 (one per task)
            # and per-domain weights are degenerate. Run original path.
            self.loss_w = None
        else:
            p = np.array([self.train_domain_p.get(d, 0.0) for d in range(domain_num)],
                         dtype=np.float64)
            assert p.sum() > 0, "train domain share is zero"
            if args.loss_weighting == "sample":
                w = p / p.mean()
            elif args.loss_weighting == "sqrt":
                w = np.sqrt(p) / np.sqrt(p).mean()
            else:
                raise ValueError(f"unknown loss_weighting={args.loss_weighting!r}")
            # task-major flatten: idx = t * D + d
            w_flat = np.tile(w, self.task_num)
            self.loss_w = torch.as_tensor(w_flat, dtype=torch.float32,
                                          device=self.device)
        if self.loss_w is None:
            print(f"[p4] loss_weighting={args.loss_weighting} w=None "
                  f"(equal / original 6-term path)", flush=True)
        else:
            w_per_d = self.loss_w.detach().cpu().numpy()[:domain_num]
            print(f"[p4] loss_weighting={args.loss_weighting} "
                  f"w_per_domain=[{', '.join(f'{x:.4f}' for x in w_per_d)}] "
                  f"sum={float(self.loss_w.sum()):.4f}", flush=True)
        dg = DataGenerator(x_tr, y_tr)
        self.train_dl, self.val_dl, self.test_dl = dg.generate_dataloader(
            x_val=x_va, y_val=y_va, x_test=x_te, y_test=y_te, batch_size=args.batch_size)
        self.test_dl_eval = DataLoader(TorchDataset(x_te, y_te), batch_size=args.batch_size,
                                       shuffle=False, num_workers=0)
        cls = MDMTRecBeta if (args.curriculum_mode != "none") else MDMTRec
        kw = dict(domain_num=domain_num, task_num=self.task_num, fcn_dims=[512, 256, 256, 64],
                  expert_num=args.expert_num, exp_d=1, exp_t=1, bal_d=1, bal_t=1)
        if cls is MDMTRecBeta:
            kw["domain_p"] = [self.train_domain_p.get(d, 0.0) for d in range(domain_num)]
            kw["kappa0"] = args.kappa0
            kw["curriculum_beta_t"] = bool(args.curriculum_beta_t)
        # P0-3: backbone switch. mdmoe path is the default (and the only path
        # that supports --curriculum_mode != none, since MDMTRecBeta subclasses
        # MDMTRec). mmoe path loads p7_models.Backbone with backbone="mmoe"
        # and ignores exp_d / exp_t / bal_d / bal_t (they have no analog in
        # the FuxiCTR layer).
        if args.backbone == "mmoe":
            from p7_models import Backbone
            self.model = Backbone(dense_feas + sparse_feas, domain_num=domain_num,
                                  task_num=self.task_num, backbone="mmoe",
                                  expert_num=args.expert_num)
        else:
            self.model = cls(dense_feas + sparse_feas, **kw)
        if args.fix_arch > 0:
            import math
            raw = math.log(args.fix_arch / (1 - args.fix_arch))
            prm = dict(self.model.named_parameters())
            for pn in ARCH_PARAMS:
                w = prm.get(pn, prm.get(f"{pn}.deep_weights"))
                w.data.fill_(raw)
                w.requires_grad_(False)
            print(f"[p4] arch params frozen at sigmoid={args.fix_arch}", flush=True)
        self.ctr = CTRTrainer(self.model,
                              optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
                              optimizer_params_darts={"lr": args.learning_rate_darts,
                                                      "weight_decay": args.weight_decay},
                              n_epoch=args.epoch, earlystop_patience=10, device=args.device,
                              model_path="./model_para/" + str(uuid.uuid4()),
                              scheduler_params={"step_size": 4, "gamma": 0.85})
        if args.fix_optim_grouping:
            flag = lambda name: name.startswith("_weight_")
            main_params = [p for n, p in self.model.named_parameters() if not flag(n)]
            darts_params = [p for n, p in self.model.named_parameters() if flag(n)]
            # component 1 per_domain_only: shared group beta1=0. P0-3: the
            # Backbone has its own shared_param_names() that excludes
            # core.gate.* / tower.* correctly. p1's shared_param_names treats
            # core.gate.* as shared (name doesn't start with "gate."), which
            # would be wrong on a Backbone.
            if args.momentum_mode == "per_domain_only":
                if args.backbone == "mmoe":
                    shared = set(self.model.shared_param_names())
                else:
                    shared = set(shared_param_names(self.model))
                shared_params = [p for n, p in self.model.named_parameters() if n in shared]
                private_params = [p for n, p in self.model.named_parameters()
                                  if n not in shared and not flag(n)]
                self.ctr.optimizer = torch.optim.Adam(
                    [{"params": shared_params, "betas": (0.0, 0.999)},
                     {"params": private_params, "betas": (0.9, 0.999)}],
                    lr=args.learning_rate, weight_decay=args.weight_decay)
            else:
                # P1-3 D-arm: under momentum_mode=none there is normally a
                # single param group (default β1=0.9). To honour a non-default
                # --beta1_shared we must first split shared/private. Default
                # path (beta1_shared=None) is left bit-identical: a single
                # group with the default betas=(0.9, 0.999).
                if args.beta1_shared is not None:
                    if args.backbone == "mmoe":
                        shared = set(self.model.shared_param_names())
                    else:
                        shared = set(shared_param_names(self.model))
                    shared_params = [p for n, p in self.model.named_parameters()
                                     if n in shared and not flag(n)]
                    private_params = [p for n, p in self.model.named_parameters()
                                      if n not in shared and not flag(n)]
                    self.ctr.optimizer = torch.optim.Adam(
                        [{"params": shared_params, "betas": (args.beta1_shared, 0.999)},
                         {"params": private_params, "betas": (0.9, 0.999)}],
                        lr=args.learning_rate, weight_decay=args.weight_decay)
                    print(f"[p1_3] D-arm split: shared β1={args.beta1_shared:.6f}, "
                          f"private β1=0.9", flush=True)
                else:
                    self.ctr.optimizer = torch.optim.Adam(main_params, lr=args.learning_rate,
                                                          weight_decay=args.weight_decay)
            self.ctr.optimizer_loss_weight = torch.optim.Adam(darts_params, lr=args.learning_rate_darts,
                                                              weight_decay=args.weight_decay)
        self.momentum = None
        if args.momentum_mode != "none":
            self.momentum = PerDomainMomentum(self.model, domain_num,
                                              gamma0=args.gamma0, gamma_max=args.gamma_max,
                                              domain_p=[self.train_domain_p.get(d, 0.0)
                                                        for d in range(domain_num)],
                                              device=args.device,
                                              gamma_source=args.gamma_source,
                                              gamma_manual=args.gamma_manual,
                                              gamma_uniform=args.gamma_uniform)
            # P0-3: Backbone's _weight_dummy is registered with no in-graph
            # consumer, and (more importantly) p1's shared_param_names
            # (PerDomainMomentum.__init__ default) treats Backbone's
            # `core.gate.*` as shared. Override with the Backbone-aware
            # shared-name set, mirroring p7_run.py:194-197. mdmoe path is
            # unchanged (p1's filter is correct on MDMTRec).
            if args.backbone == "mmoe":
                self.momentum.names = [n for n in self.model.shared_param_names()
                                       if n in dict(self.model.named_parameters())]
                self.momentum.named = dict(self.model.named_parameters())
                self.momentum.m = {
                    d: {n: torch.zeros_like(self.momentum.named[n]) for n in self.momentum.names}
                    for d in range(domain_num)}
        self.arch_ema = {pn: None for pn in ARCH_PARAMS}
        self.loss_stream = []
        self.arch_iter = None
        # P1-2: per-epoch grad norm logging state. Populated only when
        # --log_grad_norm=1; otherwise stays empty and write_grad_norm_csv
        # is a no-op. The snapshot of shared-param .grad from the last
        # training step lives here between train_one_epoch and the
        # arch_update call (which would otherwise zero .grad).
        self.grad_norm_rows = []
        self._last_step_grad_shared = None
        # P0-1 epoch-level state (filled by instrumented-loop branch in fit()).
        # Default-initialised so --run_dir without --role_setting does not crash.
        self.epoch_log = []
        self.best_epoch = None
        self.stopped_epoch = None
        self.best_val_logloss_list = None
        self.best_val_auc_list = None
        # pass-through observer for the per-step loss stream (used by the
        # all-off equivalence test); never alters values
        self._obs_orig = self.ctr.criterion.forward
        self._obs_recording = False

        def _wrapped(*a, **k):
            v = self._obs_orig(*a, **k)
            if self._obs_recording:
                self.loss_stream.append(float(v.detach()))
            return v

        self.ctr.criterion.forward = _wrapped
        # P0-5: when comp1 + non-equal weighting is active, print a one-line
        # visibility header so smoke runs can confirm gamma_d and loss_w
        # landed as expected without re-reading the script.
        if (args.momentum_mode == "per_domain_only" and self.momentum is not None
                and self.loss_w is not None):
            w_per_d = self.loss_w.detach().cpu().numpy()[:domain_num]
            gd = np.round(self.momentum.gamma_d, 4).tolist()
            print(f"[p0_5] loss_w=[{', '.join(f'{x:.4f}' for x in w_per_d)}] "
                  f"gamma_d=[{', '.join(f'{x:g}' for x in gd)}]", flush=True)

    def _cache_path(self, args):
        import hashlib
        key = f"{args.dataset_path}|{args.kuairand}|{args.downsample_ratio}|{args.downsample_seed}|{args.single_domain}"
        if args.role_setting:
            # P0-1 role reversal: cache key now also depends on (setting, sizes, seed)
            key = f"{key}|{args.role_setting}|{args.role_sizes}|{args.seed}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cache",
                            f"data_{h}.pkl")

    def _load_cache(self, args):
        """Deterministic preprocessing cache. The preprocessing consumes no
        torch RNG (setup_seed is called before, model init after), so reusing
        it cannot change any training trajectory."""
        path = self._cache_path(args)
        if os.path.exists(path):
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
            print(f"[p4] loaded cached data: {path}", flush=True)
            return obj
        return None

    def _store_cache(self, args, dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va,
                     x_te, y_te, domain_num):
        import pickle
        path = self._cache_path(args)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te,
                         domain_num), f)
        print(f"[p4] cached data: {path}", flush=True)

    def _load_data(self, args):
        if args.role_setting:
            # P0-1 role reversal: load full "all/all" split (val/test globally fixed),
            # then role_reversal_sample the training rows, then (oracle only) further
            # mask to a single domain and zero the domain column so domain_id==0
            # selects the kept rows. role_setting takes precedence over
            # downsample_ratio (mutually exclusive).
            sizes = parse_role_sizes(args.role_sizes)
            dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, _ = \
                get_movielens_data_rank_multidomain("all", "all", args.dataset_path)
            x_tr, y_tr = role_reversal_sample(x_tr, y_tr, args.role_setting, args.seed, sizes)
            domain_num = 3
            if args.single_domain >= 0:
                d = args.single_domain
                tr = x_tr[:, -1].astype(int) == d
                va = x_va[:, -1].astype(int) == d
                te = x_te[:, -1].astype(int) == d
                x_tr, y_tr = x_tr[tr], y_tr[tr]
                x_va, y_va = x_va[va], y_va[va]
                x_te, y_te = x_te[te], y_te[te]
                # zero domain column so domain_id==0 selects the kept rows
                x_tr[:, -1] = 0
                x_va[:, -1] = 0
                x_te[:, -1] = 0
                domain_num = 1
            return dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num
        if args.single_domain >= 0:
            # single-domain oracle: same as-published path as run_movielens --domain d
            dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, _ = \
                get_movielens_data_rank_multidomain("all", str(args.single_domain), args.dataset_path)
            domain_num = 1
        elif args.kuairand:
            from run_kuaishou import get_kuaishou_data_rank_multidomain
            dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
                get_kuaishou_data_rank_multidomain("all", "all", args.dataset_path)
        else:
            dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
                get_movielens_data_rank_multidomain("all", "all", args.dataset_path)
        if args.downsample_ratio > 0:
            x_tr, y_tr = downsample_train(x_tr, y_tr, args.downsample_ratio, args.downsample_seed)
        return dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num

    def _obs_start(self):
        self._obs_recording = True

    def _obs_stop(self):
        self._obs_recording = False

    # ---- arch update with component 3 ----
    def _arch_step(self, _loss):
        args = self.args
        for pn, p in arch_named(self.model).items():
            # P0-3: Backbone has no canonical arch params (only _weight_dummy
            # placeholder), so `p` is None for every ARCH_PARAMS entry. Skip
            # the None cases before touching .grad (which would AttributeError
            # on a None parameter object). For MDMTRec / MDMTRecBeta the path
            # is unchanged: every pn resolves to a real Parameter, and the
            # `p.grad is None` continue was already present.
            if p is None:
                continue
            if p.grad is None:
                continue
            if args.arch_grad_ema:
                g = p.grad.detach().clone()
                if self.arch_ema[pn] is None:
                    self.arch_ema[pn] = torch.zeros_like(g)
                self.arch_ema[pn].mul_(args.gamma_a).add_(g, alpha=1.0 - args.gamma_a)
                p.grad = self.arch_ema[pn].clone()
        self.ctr.optimizer_loss_weight.step()

    def train_one_epoch(self, data_loader, domain_num, task_num, log_interval=10):
        args = self.args
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0, disable=True)
        step_in_epoch = 0
        for i, (x_dict, y) in enumerate(tk0):
            if args.arch_update_every > 0 and step_in_epoch > 0 and step_in_epoch % args.arch_update_every == 0:
                self._arch_update_at_step(data_loader)
            x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
            y_pred = self.model(x_dict)
            y_pred = y_pred.to(self.device)
            if args.momentum_mode == "none":
                # original path
                domain_mask_list = []
                domain_id = x_dict[-1, :].clone().detach()
                for d in range(domain_num):
                    domain_mask_list.append((domain_id == d))
                logloss_list = list()
                for t in range(task_num):
                    for d in range(domain_num):
                        logloss_list.append(self.ctr.criterion(
                            y_pred[domain_mask_list[d], t], y[domain_mask_list[d], t].float()))
                # P0-4: per-(d, t) reweighting when --loss_weighting is set
                # (equal / single_domain -> loss_w is None -> original path).
                # Mutex with comp1 enforced in main().
                if self.loss_w is None:
                    loss = sum(torch.stack(logloss_list))
                else:
                    loss = (torch.stack(logloss_list) * self.loss_w).sum()
                self.model.zero_grad()
                loss.backward()
                self.ctr.optimizer.step()
            else:
                # component 1: D backwards
                domain_mask_list = []
                domain_id = x_dict[-1, :].clone().detach()
                for d in range(domain_num):
                    domain_mask_list.append((domain_id == d))
                logloss_list = list()
                for t in range(task_num):
                    for d in range(domain_num):
                        logloss_list.append(self.ctr.criterion(
                            y_pred[domain_mask_list[d], t], y[domain_mask_list[d], t].float()))
                # P0-5: when --loss_weighting != equal (sample on comp1) feed
                # the per-(d, t) weight into backward_accumulate so each
                # parameter group's per-step gradient receives w exactly once
                # (see backward_accumulate docstring). equal / single_domain
                # -> loss_w is None -> terms_w=None -> legacy equal path
                # bit-identical.
                terms_w = (self.loss_w if self.loss_w is not None else None)
                self.momentum.backward_accumulate(logloss_list, task_num, terms_w=terms_w)
                self.momentum.apply_grads()
                loss = sum(torch.stack(logloss_list))  # for logging only
                self.ctr.optimizer.step()
            total_loss += float(loss.item())
            self.loss_stream.append(float(loss.item()))
            step_in_epoch += 1
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def _capture_last_step_shared_grad(self):
        """Snapshot shared-param .grad tensors from the LAST training step
        into self._last_step_grad_shared. Must be called immediately after
        train_one_epoch returns — Adam does not zero .grad, so the post-
        step tensors are still alive until the next call to model.zero_grad
        (which happens at the top of the arch_update block in fit()).
        Defensive: skips params whose .grad is None (e.g., the trainer
        stepped a parameter group with no gradient this step)."""
        model = self.model
        named = dict(model.named_parameters())
        snap = {}
        for n in shared_param_names(model):
            if n in named and named[n].grad is not None:
                snap[n] = named[n].grad.detach().clone()
        self._last_step_grad_shared = snap

    def _compute_per_epoch_grad_norms(self):
        """P1-2 per-epoch grad norm logging. Returns
            (grad_norm_shared, [gn_d0, gn_d1, gn_d2], curriculum_rho)

        (a) grad_norm_shared: L2 norm of the snapshot from
            self._last_step_grad_shared (the last training step's shared-
            param gradients; reflects the actual training-time trajectory,
            which is curriculum-affected through the forward pass).

        (b) per-domain norm: for each d in {0,1,2}, zero_grad, forward on
            the FIXED mini-batch x_tr[0:bs] (deterministic given seed +
            role_setting + role_sizes — identical across all 6 arms at the
            same (setting, seed)), then w_d * BCE_d and backward. The L2
            norm of the resulting shared-param .grad is logged.

        Cleans up with model.zero_grad(set_to_none=True) BEFORE returning
        so the subsequent arch_update backward in fit() starts from a
        clean slate (its own zero_grad call would also clear, but doing
        it here is defensive and documents the contract).

        When loss_w is None (equal-weight baseline path) we fall back to
        w_d = 1.0 so the formula matches the as-published per-domain loss.
        """
        args = self.args
        model = self.model
        ctr = self.ctr
        device = self.device
        domain_num = self.domain_num
        task_num = self.task_num

        named = dict(model.named_parameters())
        names = shared_param_names(model)

        # (a) norm from the snapshotted last training step
        snap = self._last_step_grad_shared or {}
        vecs = [snap[n].to(device).flatten() for n in names if n in snap]
        gn_shared = float(torch.norm(torch.cat(vecs)).item()) if vecs else 0.0

        # (b) per-domain fixed mini-batch
        bs = args.batch_size
        # x_tr layout: (N, F); last column is domain id. model.forward
        # expects (F, bs) — i.e. x.transpose(0,1).
        x_fixed = torch.as_tensor(self.x_tr[0:bs].T, dtype=torch.long, device=device)
        y_fixed = torch.as_tensor(self.y_tr[0:bs], dtype=torch.float, device=device)
        per_d = []
        for d in range(domain_num):
            model.zero_grad(set_to_none=True)
            y_pred = model(x_fixed).to(device)
            mask = (x_fixed[-1, :] == d)
            if not bool(mask.any().to(device)):
                # no rows for this domain in the fixed slice — record 0
                per_d.append(0.0)
                continue
            mask_dev = mask.to(device)
            y_pred_m = y_pred[mask_dev]
            y_fixed_m = y_fixed[mask_dev]
            w_d = float(self.loss_w[d].item()) if self.loss_w is not None else 1.0
            ll = ctr.criterion(y_pred_m[:, 0], y_fixed_m[:, 0])
            ll = ll + ctr.criterion(y_pred_m[:, 1], y_fixed_m[:, 1])
            ll = w_d * ll
            ll.backward()
            vecs = [named[n].grad.detach().flatten()
                    for n in names if n in named and named[n].grad is not None]
            per_d.append(float(torch.norm(torch.cat(vecs)).item()) if vecs else 0.0)

        # contract: clean up so the arch_update / val backprops start fresh
        model.zero_grad(set_to_none=True)

        # curriculum_rho: only MDMTRecBeta tracks it; otherwise rho=1.0
        # (the all-off equivalence test depends on rho=1).
        rho = float(model._curriculum_rho) if isinstance(model, MDMTRecBeta) else 1.0
        return gn_shared, per_d, rho

    def _arch_update_at_step(self, data_loader):
        """Component 3 --arch_update_every: pull one batch from the train
        iterator and update arch params mid-epoch (as-published loss form)."""
        if self.arch_iter is None:
            self.arch_iter = iter(data_loader)
        try:
            _val_x, _val_y = next(self.arch_iter)
        except StopIteration:
            self.arch_iter = iter(data_loader)
            _val_x, _val_y = next(self.arch_iter)
        _val_x = _val_x.transpose(0, 1).to(self.device)
        _val_y = _val_y.to(self.device)
        _y_pred = self.model(_val_x).to(self.device)
        if self.args.arch_loss_masked:
            domain_id = _val_x[-1, :].clone().detach()
            ll = []
            for t in range(self.task_num):
                for d in range(self.domain_num):
                    m = (domain_id == d)
                    ll.append(self.ctr.criterion(_y_pred[m, t], _val_y[m, t].float()))
            _loss = sum(torch.stack(ll))
        else:
            _loss = self.ctr.criterion(_y_pred, _val_y.float())
        self.model.zero_grad()
        _loss.backward()
        self._arch_step(_loss)

    def fit(self):
        args = self.args
        self.device = args.device
        ctr = self.ctr
        model = self.model
        domain_num, task_num = self.domain_num, self.task_num
        components_on = (args.momentum_mode != "none" or args.curriculum_mode != "none"
                         or args.arch_grad_ema or args.arch_loss_masked or args.arch_update_every
                         or bool(args.role_setting))
        if not components_on:
            # pure original path (only the pass-through observer is attached)
            orig_toe = ctr.train_one_epoch

            def toe_recording(dl, domain_num, task_num, log_interval=10):
                self._obs_start()
                orig_toe(dl, domain_num=domain_num, task_num=task_num, log_interval=log_interval)
                self._obs_stop()

            ctr.train_one_epoch = toe_recording
            ctr.fit(self.train_dl, val_dataloader=self.val_dl, exp_d=1, exp_t=1,
                    bal_d=1, bal_t=1, domain_num=domain_num, task_num=task_num)
            return
        # instrumented-but-equivalent fit with component hooks.
        # When role_setting is set, all P0-1 runs (incl. baseline, oracle) go
        # through this branch; with momentum_mode=="none" and the arch-default
        # switches off, the per-step math is identical to ctr.fit() (same
        # train_one_epoch body, same _loss = ctr.criterion(...) form, no
        # scheduler step in either path).
        valid_data_iter = iter(self.train_dl)
        total_epochs = args.epoch
        self.epoch_log = []
        best_auc_seen = -1.0
        self.best_epoch = None
        self.best_val_logloss_list = None
        self.best_val_auc_list = None
        self.stopped_epoch = total_epochs - 1  # default: completed all epochs without early stop
        for epoch_i in range(total_epochs):
            if isinstance(model, MDMTRecBeta):
                model.set_epoch(epoch_i, total_epochs)
            epoch_loss_start = len(self.loss_stream)
            self.train_one_epoch(self.train_dl, domain_num=domain_num, task_num=task_num)
            # P1-2: per-epoch grad norm logging (off by default). Snapshot
            # the last training step's shared grads BEFORE arch_update's
            # zero_grad clobbers them, then run the per-domain backprops
            # on a fixed mini-batch. Both must happen before the arch
            # update block to avoid double-counting arch backprops.
            if args.log_grad_norm:
                self._capture_last_step_shared_grad()
                gn_shared, gn_per_d, rho = self._compute_per_epoch_grad_norms()
                self.grad_norm_rows.append({
                    "epoch": epoch_i,
                    "grad_norm_shared": gn_shared,
                    "grad_norm_d0": gn_per_d[0],
                    "grad_norm_d1": gn_per_d[1],
                    "grad_norm_d2": gn_per_d[2],
                    "curriculum_rho": rho,
                })
            # P0-1: train_loss_mean = mean of new entries added this epoch.
            # In the instrumented loop the observer is OFF, so loss_stream
            # grows by exactly 1 per step (the per-step summed loss appended
            # inside train_one_epoch). When observer IS active, it adds D*T
            # extra entries per step and the per-step summed loss lives at
            # offset D*T within each (D*T + 1) group.
            new_loss = self.loss_stream[epoch_loss_start:]
            if self._obs_recording:
                per_step_appends = domain_num * task_num + 1
                observer_appends = domain_num * task_num
                n_steps = len(new_loss) // per_step_appends
                step_losses = [new_loss[k * per_step_appends + observer_appends]
                               for k in range(n_steps)]
            else:
                step_losses = list(new_loss)
            train_loss_mean = float(np.mean(step_losses)) if step_losses else 0.0
            # arch update (component 3 configures cadence; default once/epoch)
            try:
                _val_x, _val_y = next(valid_data_iter)
            except StopIteration:
                del valid_data_iter
                import gc
                gc.collect()
                valid_data_iter = iter(self.train_dl)
                _val_x, _val_y = next(valid_data_iter)
            _val_x = _val_x.transpose(0, 1).to(self.device)
            _val_y = _val_y.to(self.device)
            _y_pred = model(_val_x).to(self.device)
            if args.arch_loss_masked:
                domain_id = _val_x[-1, :].clone().detach()
                ll = []
                for t in range(task_num):
                    for d in range(domain_num):
                        m = (domain_id == d)
                        ll.append(ctr.criterion(_y_pred[m, t], _val_y[m, t].float()))
                _loss = sum(torch.stack(ll))
            else:
                _loss = ctr.criterion(_y_pred, _val_y.float())
            model.zero_grad()
            _loss.backward()
            self._arch_step(_loss)

            logloss_list, auc_list, _, _ = ctr.evaluate_multi_domain(
                model, self.val_dl, domain_num=domain_num, task_num=task_num)
            auc = sum(auc_list) / (domain_num * task_num)
            rng = torch.get_rng_state()
            _, test_auc_list, _, _ = ctr.evaluate_multi_domain(
                model, self.test_dl_eval, domain_num=domain_num, task_num=task_num)
            torch.set_rng_state(rng)
            print("epoch:", epoch_i, "validation: auc:", auc, flush=True)
            # P0-1 epoch_log + best_epoch tracking (strict >, first-argmax semantics
            # consistent with EarlyStopper.stop_training)
            self.epoch_log.append({
                "epoch_i": epoch_i,
                "train_loss_mean": train_loss_mean,
                "val_logloss_list": list(logloss_list),
                "val_auc_list": list(auc_list),
                "test_auc_list": list(test_auc_list),
            })
            if auc > best_auc_seen:
                best_auc_seen = auc
                self.best_epoch = epoch_i
                self.best_val_logloss_list = list(logloss_list)
                self.best_val_auc_list = list(auc_list)
            if ctr.early_stopper.stop_training(auc, model.state_dict()):
                print(f"validation: best auc: {ctr.early_stopper.best_auc}", flush=True)
                model.load_state_dict(ctr.early_stopper.best_weights)
                self.stopped_epoch = epoch_i
                break
        torch.save(model.state_dict(), ctr.model_path + "model.pth")


def write_p0_1_metrics(args, r, ll, aucs):
    """P0-1: emit metrics_epoch.csv (long) and metrics_summary.csv (one row)
    under --run_dir. method is derived from --momentum_mode and --single_domain."""
    if args.single_domain >= 0:
        method = "oracle"
        domain_field = int(args.single_domain)
    elif args.momentum_mode == "per_domain_only":
        method = "comp1"
        domain_field = "all"
    else:
        method = "baseline"
        domain_field = "all"
    # P0-3: model label is backbone-aware so the report can group MDMTRec
    # and MMoE rows by backbone without parsing the run_dir path.
    model_label = "MMoE" if args.backbone == "mmoe" else "MDMTRec"
    sizes = parse_role_sizes(args.role_sizes)
    setting = args.role_setting or "none"
    role_assignment = ROLE_MAPS.get(args.role_setting, {})
    ratio_parts = [f"{r}{sizes[r]}" for r in ("L", "M", "S")]
    domain_ratio = ":".join(ratio_parts)
    if role_assignment:
        domain_ratio += ";d0=" + role_assignment[0] + ",d1=" + role_assignment[1] + ",d2=" + role_assignment[2]
    counts = r.train_counts
    total_train_samples = int(sum(counts.values()))
    if method == "oracle":
        # After single_domain mask + zero-domain-col, all training rows carry
        # domain=0; the row's "domain" column is the original single_domain id.
        domain_train_samples = f"{int(args.single_domain)}={total_train_samples}"
    else:
        domain_train_samples = ",".join(f"{d}={counts[d]}" for d in sorted(counts))
    best_epoch = r.best_epoch if r.best_epoch is not None else 0
    stopped_epoch = r.stopped_epoch if r.stopped_epoch is not None else 0
    if r.epoch_log and 0 <= best_epoch < len(r.epoch_log):
        train_loss_at_best = r.epoch_log[best_epoch]["train_loss_mean"]
    else:
        train_loss_at_best = ""
    final_test_auc = list(aucs)
    final_test_logloss = list(ll)
    if r.best_val_auc_list is not None:
        best_val_auc = list(r.best_val_auc_list)
        best_val_logloss = list(r.best_val_logloss_list)
    else:
        best_val_auc = [None] * r.domain_num * r.task_num
        best_val_logloss = [None] * r.domain_num * r.task_num
    if all(x is not None for x in best_val_auc):
        best_val_auc_mean = float(np.mean(best_val_auc))
    else:
        best_val_auc_mean = ""
    # ---- metrics_epoch.csv (long) ----
    ep_fields = ["experiment_stage", "experiment_name", "model", "method", "setting", "seed",
                 "epoch", "domain", "task", "train_loss", "val_loss", "val_auc", "test_auc"]
    ep_rows = []
    for entry in r.epoch_log:
        ei = entry["epoch_i"]
        tl = entry["train_loss_mean"]
        # P0-4: test_auc_list may be missing for legacy epoch_log rows
        # (older runs that pre-date the instrumentation); fall back to "".
        ta_list = entry.get("test_auc_list", [None] * (r.domain_num * r.task_num))
        for t in range(r.task_num):
            for d in range(r.domain_num):
                idx = d + t * r.domain_num
                ta = ta_list[idx] if idx < len(ta_list) else None
                ep_rows.append({
                    "experiment_stage": args.stage_label,
                    "experiment_name": args.experiment_label,
                    "model": model_label,
                    "method": method,
                    "setting": setting,
                    "seed": args.seed,
                    "epoch": ei,
                    "domain": (int(args.single_domain) if method == "oracle" else d),
                    "task": t,
                    "train_loss": f"{tl:.6f}",
                    "val_loss": f"{entry['val_logloss_list'][idx]:.6f}",
                    "val_auc": f"{entry['val_auc_list'][idx]:.6f}",
                    "test_auc": (f"{ta:.6f}" if ta is not None else ""),
                })
    ep_path = os.path.join(args.run_dir, "metrics_epoch.csv")
    with open(ep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields)
        w.writeheader()
        for r0 in ep_rows:
            w.writerow(r0)
    # ---- metrics_summary.csv (one row) ----
    sum_fields = ["experiment_stage", "experiment_name", "model", "method", "dataset", "setting",
                  "domain", "domain_ratio", "total_train_samples", "domain_train_samples", "seed",
                  "best_epoch", "stopped_epoch", "best_val_auc", "train_loss"]
    # The per-cell column layout always spans the global 3 domains, so column
    # `*_d{D}t{T}` always means domain D / task T. Oracle runs train with
    # domain_num=1 and their eval lists hold only [t0, t1] for the oracle
    # domain; only that domain's columns are filled, the rest stay blank.
    n_dom_cols = 3 if method == "oracle" else r.domain_num
    for d in range(n_dom_cols):
        for t in range(r.task_num):
            sum_fields.append(f"val_auc_d{d}t{t}")
            sum_fields.append(f"val_log_d{d}t{t}")
            sum_fields.append(f"auc_d{d}t{t}")
            sum_fields.append(f"log_d{d}t{t}")
    sum_row = {
        "experiment_stage": args.stage_label,
        "experiment_name": args.experiment_label,
        "model": model_label,
        "method": method,
        "dataset": "ml-1m",
        "setting": setting,
        "domain": domain_field,
        "domain_ratio": domain_ratio,
        "total_train_samples": total_train_samples,
        "domain_train_samples": domain_train_samples,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_auc": f"{best_val_auc_mean:.6f}" if best_val_auc_mean != "" else "",
        "train_loss": f"{train_loss_at_best:.6f}" if train_loss_at_best != "" else "",
    }
    for d in range(n_dom_cols):
        for t in range(r.task_num):
            if method == "oracle":
                if d != int(args.single_domain):
                    sum_row[f"val_auc_d{d}t{t}"] = ""
                    sum_row[f"val_log_d{d}t{t}"] = ""
                    sum_row[f"auc_d{d}t{t}"] = ""
                    sum_row[f"log_d{d}t{t}"] = ""
                    continue
                idx = t  # oracle eval lists are per-task for the single domain
            else:
                idx = d + t * r.domain_num
            vb = best_val_auc[idx] if idx < len(best_val_auc) else None
            vlb = best_val_logloss[idx] if idx < len(best_val_logloss) else None
            ftb = final_test_auc[idx] if idx < len(final_test_auc) else None
            ftlb = final_test_logloss[idx] if idx < len(final_test_logloss) else None
            sum_row[f"val_auc_d{d}t{t}"] = f"{vb:.6f}" if vb is not None else ""
            sum_row[f"val_log_d{d}t{t}"] = f"{vlb:.6f}" if vlb is not None else ""
            sum_row[f"auc_d{d}t{t}"] = f"{ftb:.6f}" if ftb is not None else ""
            sum_row[f"log_d{d}t{t}"] = f"{ftlb:.6f}" if ftlb is not None else ""
    sm_path = os.path.join(args.run_dir, "metrics_summary.csv")
    with open(sm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerow(sum_row)
    print(f"[p4] P0-1 metrics written to {args.run_dir}", flush=True)
    print(f"[p4] model_path: {r.ctr.model_path}model.pth", flush=True)


def write_grad_norm_csv(args, r):
    """P1-2: emit grad_norm_epoch.csv under --run_dir. Long format, one row
    per (epoch, domain-label) where domain-label in {shared, d0, d1, d2}.
    Columns: epoch, domain, grad_norm, curriculum_rho.

    Avoids polluting metrics_epoch.csv so p0_4_report.py can keep its
    long-format parser (which is column-position sensitive — see
    p0_4_report.py:load_metrics_epoch).
    """
    if not args.log_grad_norm:
        return
    if not getattr(r, "grad_norm_rows", None):
        print(f"[p4] --log_grad_norm=1 but no grad_norm_rows collected "
              f"(likely ran the all-off path); skipping grad_norm_epoch.csv",
              flush=True)
        return
    rows = []
    for entry in r.grad_norm_rows:
        ep = entry["epoch"]
        rho = entry["curriculum_rho"]
        rows.append({"epoch": ep, "domain": "shared",
                     "grad_norm": f"{entry['grad_norm_shared']:.6e}",
                     "curriculum_rho": f"{rho:.6f}"})
        for k in ("d0", "d1", "d2"):
            rows.append({"epoch": ep, "domain": k,
                         "grad_norm": f"{entry[f'grad_norm_{k}']:.6e}",
                         "curriculum_rho": f"{rho:.6f}"})
    path = os.path.join(args.run_dir, "grad_norm_epoch.csv")
    fields = ["epoch", "domain", "grad_norm", "curriculum_rho"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r0 in rows:
            w.writerow(r0)
    print(f"[p4] grad_norm_epoch.csv written to {args.run_dir} "
          f"({len(rows)} rows)", flush=True)


def main():
    import time
    t0 = time.time()
    args = build_argparser().parse_args()
    # P0-4 mutex: --loss_weighting is incompatible with --single_domain
    # (oracle runs only have 2 loss terms, the per-domain weight is
    # degenerate). The 6-term equal-weight path is left intact.
    # P0-5 narrow-exception: --loss_weighting in {equal, sample} is allowed
    # alongside --momentum_mode per_domain_only, since comp1's D backwards
    # are linear in L_d and accept a per-term weight w_d without changing
    # the EMA structure. sqrt weighting stays mutex with comp1 (sqrt
    # coupling would not commute with the EMA decay schedule); all other
    # --momentum_mode values stay mutex with any non-equal weighting.
    if args.loss_weighting not in ("equal", "sample", "sqrt"):
        # argparse already constrains choices, but defensive check.
        print(f"[p4][ERROR] unknown --loss_weighting={args.loss_weighting}", flush=True)
        sys.exit(2)
    if args.loss_weighting != "equal":
        if args.single_domain >= 0:
            print(f"[p4][ERROR] --loss_weighting={args.loss_weighting} is "
                  f"incompatible with --single_domain={args.single_domain} "
                  f"(oracle runs have no per-domain weighting).", flush=True)
            sys.exit(2)
        if args.momentum_mode not in ("none", "per_domain_only"):
            print(f"[p4][ERROR] --loss_weighting={args.loss_weighting} is "
                  f"incompatible with --momentum_mode={args.momentum_mode} "
                  f"(must be 'none' or 'per_domain_only').", flush=True)
            sys.exit(2)
        if args.momentum_mode == "per_domain_only" and args.loss_weighting == "sqrt":
            print(f"[p4][ERROR] --loss_weighting=sqrt is incompatible with "
                  f"--momentum_mode=per_domain_only (sqrt coupling would not "
                  f"commute with the EMA decay schedule).", flush=True)
            sys.exit(2)
    if args.role_setting:
        # P0-1 mutex: forbid non-default switches per spec section 3. All P0-1
        # arms run with --fix_optim_grouping 1 and the arch-default switches
        # off; anything else would silently change a controlled variable.
        # P1-2 carve-out: the curriculum life-or-death matrix pairs role_setting
        # with --curriculum_mode!=none (the whole point of the sweep). The
        # --stage_label flag is the contract that identifies a P1-2 arm.
        forbidden = []
        if args.downsample_ratio > 0:
            forbidden.append("--downsample_ratio>0")
        if args.curriculum_mode != "none" and args.stage_label != "P1-2":
            forbidden.append("--curriculum_mode!=none")
        # P1-3 C-arm carve-out: gamma_source=uniform + gamma_uniform drives
        # the matched-momentum hypothesis; allowed only when --stage_label
        # is P1-3 (the four P1-3 arms are the only legitimate callers).
        p1_3_c_arm = (args.stage_label == "P1-3"
                      and args.gamma_source == "uniform"
                      and args.gamma_uniform is not None)
        # P1-3 D-arm carve-out: beta1_shared is only meaningful under
        # momentum_mode=none (per spec §1c); the P1-3 stage label is the
        # contract that identifies a P1-3 caller.
        p1_3_d_arm = (args.stage_label == "P1-3"
                       and args.beta1_shared is not None)
        if args.gamma_source != "formula" and not p1_3_c_arm:
            forbidden.append("--gamma_source!=formula")
        if args.gamma_uniform is not None and not p1_3_c_arm:
            forbidden.append("--gamma_uniform is set")
        if args.beta1_shared is not None and not p1_3_d_arm:
            forbidden.append("--beta1_shared is set")
        if args.fix_arch > 0:
            forbidden.append("--fix_arch>0")
        if args.arch_grad_ema:
            forbidden.append("--arch_grad_ema!=0")
        if args.arch_loss_masked:
            forbidden.append("--arch_loss_masked!=0")
        if args.arch_update_every:
            forbidden.append("--arch_update_every!=0")
        if args.kuairand:
            forbidden.append("--kuairand!=0 (P0-1 is ml-1m only)")
        if args.momentum_mode not in ("none", "per_domain_only"):
            forbidden.append("--momentum_mode must be none|per_domain_only for P0-1 arms")
        if not args.fix_optim_grouping:
            forbidden.append("--fix_optim_grouping must be 1 for all P0-1 arms")
        assert not forbidden, ("--role_setting is incompatible with: " + ", ".join(forbidden)
                               + " (P0-1 spec section 3)")
    # P1-3 mutex: narrow the allowed (gamma_source, momentum_mode, beta1_shared,
    # loss_weighting) combinations to the four P1-3 arms. Any mismatch is a
    # misuse of the new flags and should fail loudly.
    # C arm contract: gamma_source=uniform requires per_domain_only (the EMA
    # structure that uniform collapses is meaningless under momentum_mode=none).
    # D arm contract: beta1_shared requires momentum_mode=none + loss_weighting
    # in {equal, sample} (per spec §1c). gamma_source=uniform without
    # gamma_uniform is rejected inside PerDomainMomentum.__init__.
    if args.gamma_source == "uniform" and args.momentum_mode != "per_domain_only":
        print(f"[p4][ERROR] --gamma_source=uniform requires "
              f"--momentum_mode=per_domain_only (C arm contract); got "
              f"{args.momentum_mode}", flush=True)
        sys.exit(2)
    if args.beta1_shared is not None:
        if args.momentum_mode != "none":
            print(f"[p4][ERROR] --beta1_shared requires "
                  f"--momentum_mode=none (D arm contract); got "
                  f"{args.momentum_mode}", flush=True)
            sys.exit(2)
        if args.loss_weighting not in ("equal", "sample"):
            print(f"[p4][ERROR] --beta1_shared requires "
                  f"--loss_weighting in {{equal, sample}} (D arm contract); "
                  f"got {args.loss_weighting}", flush=True)
            sys.exit(2)
    r = P4Runner(args)
    r.fit()
    fit_sec = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    print(f"[p4] fit wall time: {fit_sec:.1f}s | peak cuda mem: {peak_mb:.0f} MiB", flush=True)
    ll, aucs, tot_ll, tot_auc = r.ctr.evaluate_multi_domain(r.ctr.model, r.test_dl,
                                                            r.domain_num, task_num=r.task_num)
    os.makedirs(args.outdir, exist_ok=True)
    if not args.role_setting:
        # P0-1 runs report via --run_dir only; do not append to the P4-era
        # P4_results.csv (oracle rows would also have a different cell schema).
        row = {"variant": args.tag, "seed": args.seed}
        for d in range(r.domain_num):
            for t in range(r.task_num):
                row[f"auc_d{d}t{t}"] = aucs[d + t * r.domain_num]
                row[f"log_d{d}t{t}"] = ll[d + t * r.domain_num]
        row["auc_mean"] = float(np.mean(aucs))
        named = dict(r.model.named_parameters())

        def arch_sigmoid(pn):
            p = named.get(pn)
            if p is None:
                p = named[f"{pn}.deep_weights"]
            sig = torch.sigmoid(p.detach()).flatten()
            return ",".join(f"{v:.4f}" for v in sig.tolist())

        row["final_arch"] = "|".join(f"{pn}:{arch_sigmoid(pn)}" for pn in ARCH_PARAMS)
        out_csv = os.path.join(args.outdir, "P4_results.csv")
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
    if args.loss_stream:
        vals = r.loss_stream
        components_on = (args.momentum_mode != "none" or args.curriculum_mode != "none"
                         or args.arch_grad_ema or args.arch_loss_masked or args.arch_update_every
                         or bool(args.role_setting))
        if not components_on:
            # observer records 6 criterion calls per step -> group into per-step sums
            vals = [sum(vals[i:i + 6]) for i in range(0, len(vals) - len(vals) % 6, 6)]
        with open(args.loss_stream, "w") as f:
            for v in vals:
                f.write(f"{v!r}\n")
    print("[p4] FINAL per-(d,t) test auc:", [round(a, 6) for a in aucs], flush=True)
    print("[p4] FINAL test auc:", tot_auc, flush=True)
    if args.run_dir:
        os.makedirs(args.run_dir, exist_ok=True)
        write_p0_1_metrics(args, r, ll, aucs)
        # P1-2: gradient norm logging side-channel (no-op when
        # --log_grad_norm=0). Kept separate from metrics_epoch.csv to
        # avoid breaking the P0-4 report's column-position parser.
        write_grad_norm_csv(args, r)


if __name__ == "__main__":
    main()
