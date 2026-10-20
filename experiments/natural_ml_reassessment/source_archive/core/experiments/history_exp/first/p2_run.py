"""P2 runner: controlled variants of M3oE on MovieLens-1M.

Variants (selected by flags, default = as-published):
  --fix_arch 0.7311 : freeze the 4 architecture params at sigmoid value 0.7311
                      (raw = logit(0.7311) = 1.0), requires_grad=False, so no
                      optimizer can update them (grad stays None; Adam skips).
  --fix_arch 0.5    : same but at 0.5 (paper's "w/o AutoML" ablation), raw=0.0.
  (no --fix_arch)   : architecture params trainable as published.
  --fix_optim_grouping : repair the B10 bug — main optimizer gets only non-arch
                      params, darts optimizer only the 4 arch params (proper
                      boolean filter). Implemented by REBUILDING both optimizers
                      after CTRTrainer init; default off = as-published.

All other hyperparameters mirror examples/run_movielens.py defaults.
D0.2-style arch trajectory is recorded for every variant (cheap); D0.4 is off
by default (--d04 1 to enable).
"""
import argparse
import math
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe", "examples"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe", "examples"))

import torch

from MDMTRec.trainers import CTRTrainer
from MDMTRec.models.multi_domain import MDMTRec
from MDMTRec.utils.data import DataGenerator
from run_movielens import get_movielens_data_rank_multidomain
from p1_run import setup_seed, Instrumented, ARCH_PARAMS


def apply_fix_arch(model, target_sigmoid):
    raw = math.log(target_sigmoid / (1 - target_sigmoid))
    for pn in ARCH_PARAMS:
        w = getattr(model, pn).deep_weights
        w.data.fill_(raw)
        w.requires_grad_(False)
    print(f"[p2] arch params frozen: raw={raw:.6f} sigmoid={target_sigmoid} "
          f"requires_grad=False", flush=True)
    return raw


def rebuild_optimizers(ctr, lr_main, lr_darts, wd):
    """Fix B10: proper boolean filter instead of ~."""
    flag = lambda name: name.startswith("_weight_")
    main_params = [p for n, p in ctr.model.named_parameters() if not flag(n)]
    darts_params = [p for n, p in ctr.model.named_parameters() if flag(n)]
    ctr.optimizer = torch.optim.Adam(main_params, lr=lr_main, weight_decay=wd)
    ctr.optimizer_loss_weight = torch.optim.Adam(darts_params, lr=lr_darts, weight_decay=wd)
    print(f"[p2] optimizers rebuilt: main={len(main_params)} tensors, "
          f"darts={len(darts_params)} tensors", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--variant", choices=["orig", "fixed073", "fixed05", "fixedbug"], default="orig")
    ap.add_argument("--fix_optim_grouping", type=int, default=0,
                    help="repair B10 optimizer grouping (independent switch)")
    ap.add_argument("--batch_size", type=int, default=40960)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=1e-2)
    ap.add_argument("--learning_rate_darts", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--expert_num", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset_path", default="D:/CODE/phase0-3/m3oe/examples/data/ml-1m")
    ap.add_argument("--tag", default="p2")
    ap.add_argument("--outdir", default="D:/CODE/phase0-3/experiments/results/p2")
    ap.add_argument("--d04", type=int, default=0)
    args = ap.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
        get_movielens_data_rank_multidomain("all", "all", args.dataset_path)
    task_num = 2
    dg = DataGenerator(x_tr, y_tr)
    train_dl, val_dl, test_dl = dg.generate_dataloader(x_val=x_va, y_val=y_va, x_test=x_te,
                                                       y_test=y_te, batch_size=args.batch_size)
    model = MDMTRec(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num,
                    fcn_dims=[512, 256, 256, 64], expert_num=args.expert_num,
                    exp_d=1, exp_t=1, bal_d=1, bal_t=1)

    if args.variant == "fixed073":
        apply_fix_arch(model, 0.7311)
    elif args.variant == "fixed05":
        apply_fix_arch(model, 0.5)

    ctr = CTRTrainer(model, optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
                     optimizer_params_darts={"lr": args.learning_rate_darts, "weight_decay": args.weight_decay},
                     n_epoch=args.epoch, earlystop_patience=10, device=args.device,
                     model_path="./model_para/" + str(uuid.uuid4()),
                     scheduler_params={"step_size": 4, "gamma": 0.85})
    if args.fix_optim_grouping:
        rebuild_optimizers(ctr, args.learning_rate, args.learning_rate_darts, args.weight_decay)

    # arch trajectory instrumentation (D0.2 only unless --d04 1)
    ins = Instrumented(ctr, model, domain_num, task_num, k=10, device=args.device,
                       outdir=args.outdir,
                       tag=f"{args.tag}_{args.variant}_fixgroup{args.fix_optim_grouping}_s{args.seed}",
                       batch_size=args.batch_size, loss_stream_path=None,
                       collect_d04=bool(args.d04))
    ins.fit(train_dl, val_dl, test_dl, args.epoch)
    ins.dump(args.seed)

    ll, aucs, tot_ll, tot_auc = ctr.evaluate_multi_domain(ctr.model, test_dl, domain_num, task_num=task_num)
    print(f"[p2] variant={args.variant} fixgroup={args.fix_optim_grouping} seed={args.seed}")
    print("[p2] FINAL per-(d,t) test auc:", [round(a, 6) for a in aucs], flush=True)
    print("[p2] FINAL test auc:", tot_auc, "| test logloss:", tot_ll, flush=True)
    # final arch sigmoid values (should equal target when frozen)
    vals = {pn: round(float(torch.sigmoid(getattr(model, pn).deep_weights).item()), 4)
            for pn in ARCH_PARAMS}
    print("[p2] final arch sigmoid values:", vals, flush=True)


if __name__ == "__main__":
    main()
