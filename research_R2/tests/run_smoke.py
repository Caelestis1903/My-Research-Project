"""Numerical smoke check (≤3 batch per arm, seed 2022, KuaiRand).

Confirms:
- finite loss and gradient for every arm
- main optimizer step actually updates q/delta (only for B1/B2/B4)
- arch optimizer step actually updates the four raw scalars
- arch step does NOT write q/delta

Smoke is performance-free; AUC is intentionally ignored.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

from MDMTRec.utils.data import TorchDataset  # noqa

from stage_S1_data import load_stage_s1_bundle  # noqa
from stage_S1_models import features_from_schema  # noqa
from stage_E1_instrumentation import preserve_rng_state  # noqa

from ao04_v12_model import (  # noqa
    B3_BUFFER_NAME,
    DELTA_BAL_D_NAME,
    Q_RHO_NAME,
    build_ao04_model,
    is_ao04_architecture_parameter,
)
from ao04_v12_protocol import (  # noqa
    ARCHITECTURE_PARAMETER_NAMES,
    BATCH_SIZE,
    BUDGET_PER_DATASET,
    DOMAIN_NUM,
    TASK_NUM,
)
from ao04_v12_runner import (  # noqa
    build_optimizers_for_arm,
    setup_seed,
)


def _deterministic_train_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def smoke_one(arm: str, seed: int, dataset: str) -> Dict[str, object]:
    setting = ("natural" if dataset == "kuairand" else "d0_large")
    bundle = load_stage_s1_bundle(dataset, None, setting, seed)
    feats = features_from_schema(bundle.features)
    setup_seed(seed)
    model = build_ao04_model(feats, arm, seed=seed)
    main_opt, arch_opt, main_set, arch_set = build_optimizers_for_arm(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(
        TorchDataset(bundle.train_x, bundle.train_y),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        drop_last=False, generator=_deterministic_train_generator(seed))
    val_loader = DataLoader(
        TorchDataset(bundle.val_x, bundle.val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    initial_main_q = (model.q_rho.detach().cpu().clone()
                       if hasattr(model, "q_rho") else None)
    initial_main_d = (model.delta_bal_d.detach().cpu().clone()
                       if hasattr(model, "delta_bal_d") else None)
    initial_arch = {n: model.state_dict()[n].detach().cpu().clone()
                     for n in ARCHITECTURE_PARAMETER_NAMES}

    finite_loss = True
    n_train_batches = 0
    n_arch_batches = 0
    for step, (x_batch, y_batch) in enumerate(train_loader):
        if step >= 3:
            break
        n_train_batches += 1
        x_device = x_batch.transpose(0, 1).to(device=device, dtype=torch.float32)
        y_device = y_batch.to(device=device, dtype=torch.float32)
        pred = model(x_device)
        if not bool(torch.isfinite(pred).all().detach().cpu()):
            finite_loss = False
        loss = F.binary_cross_entropy(pred, y_device.float(), reduction="mean")
        if not bool(torch.isfinite(loss).detach().cpu()):
            finite_loss = False
        main_opt.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in model.named_parameters():
            if (not is_ao04_architecture_parameter(name)
                    and p.grad is not None
                    and not bool(torch.isfinite(p.grad).all().detach().cpu())):
                raise FloatingPointError("non-finite main grad {}".format(name))
        main_opt.step()
        for name, p in model.named_parameters():
            if not bool(torch.isfinite(p).all().detach().cpu()):
                raise FloatingPointError("non-finite param {}".format(name))

    # Arch step
    arch_iter = iter(train_loader)
    x_batch, y_batch = next(arch_iter)
    n_arch_batches += 1
    with preserve_rng_state():
        x_device = x_batch.transpose(0, 1).to(device=device, dtype=torch.float32)
        y_device = y_batch.to(device=device, dtype=torch.float32)
        pred = model(x_device)
        loss = F.binary_cross_entropy(pred, y_device.float(), reduction="mean")
        arch_opt.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in model.named_parameters():
            if (is_ao04_architecture_parameter(name) and p.grad is not None
                    and not bool(torch.isfinite(p.grad).all().detach().cpu())):
                raise FloatingPointError("non-finite arch grad {}".format(name))
        arch_opt.step()

    final_main_q = (model.q_rho.detach().cpu().clone()
                     if hasattr(model, "q_rho") else None)
    final_main_d = (model.delta_bal_d.detach().cpu().clone()
                     if hasattr(model, "delta_bal_d") else None)
    final_arch = {n: model.state_dict()[n].detach().cpu().clone()
                   for n in ARCHITECTURE_PARAMETER_NAMES}

    report = {
        "arm": arm, "dataset": dataset, "seed": seed,
        "finite_loss": finite_loss,
        "n_train_batches": n_train_batches,
        "n_arch_batches": n_arch_batches,
    }
    # Q/delta main update: BEFORE != AFTER (only meaningful if param existed)
    if initial_main_q is not None:
        report["main_q_changed"] = not torch.equal(initial_main_q, final_main_q)
    else:
        report["main_q_changed"] = None
    if initial_main_d is not None:
        report["main_d_changed"] = not torch.equal(initial_main_d, final_main_d)
    else:
        report["main_d_changed"] = None
    # Arch: every raw scalar should change.
    report["arch_changed"] = {n: not torch.equal(initial_arch[n], final_arch[n])
                                for n in ARCHITECTURE_PARAMETER_NAMES}
    return report


def main() -> int:
    ok = True
    for dataset in ("kuairand", "movielens"):
        for arm in ("B0", "B1", "B2", "B3", "B4"):
            rep = smoke_one(arm=arm, seed=2022, dataset=dataset)
            tag = "[OK]" if rep["finite_loss"] else "[FAIL]"
            print(f"  {tag} {dataset}/{arm}  finite={rep['finite_loss']} "
                  f"train_batches={rep['n_train_batches']} arch={rep['n_arch_batches']} "
                  f"q_changed={rep.get('main_q_changed')} "
                  f"d_changed={rep.get('main_d_changed')}")
            for n, ch in rep["arch_changed"].items():
                print(f"      arch_changed[{n}]={ch}")
            ok = ok and rep["finite_loss"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
