"""P0 Task 3: runtime verification of code assertions that cannot be settled by
reading source alone (optimizer param groups, dims, init values).

Runs on CPU only; does not train. Mirrors the construction performed by
examples/run_movielens.py with default hyperparameters.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "m3oe"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "m3oe", "examples"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", "m3oe", "examples"))

import torch

from MDMTRec.basic.features import DenseFeature, SparseFeature
from MDMTRec.models.multi_domain.model import MDMTRec
from MDMTRec.trainers.ctr_trainer import CTRTrainer
from run_movielens import get_movielens_data_rank_multidomain

dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
    get_movielens_data_rank_multidomain('all', 'all', "D:/CODE/phase0-3/m3oe/examples/data/ml-1m")

print("=== A2 input_dim ===")
model = MDMTRec(dense_feas + sparse_feas, domain_num=3, task_num=2,
                fcn_dims=[512, 256, 256, 64], expert_num=4,
                exp_d=1, exp_t=1, bal_d=1, bal_t=1)
print("input_dim:", model.input_dim)          # expect 113
print("star_dim:", model.star_dim)            # expect [113, 512, 256]
print("fcn_dim after slicing:", model.fcn_dim)  # expect [256, 64]

print("\n=== A5 expert structure ===")
print("expert[0]:", model.expert[0])
print("tower[0]:", model.tower[0])

print("\n=== A4 skip_conn ===")
print("skip_conn:", model.skip_conn)

print("\n=== B3/B4/B5 arch params ===")
for n in ["_weight_exp_d", "_weight_exp_t", "_weight_bal_d", "_weight_bal_t"]:
    w = getattr(model, n)
    print(n, "raw:", w.deep_weights.detach().numpy(), "sigmoid:", w().detach().numpy(),
          "softmax_type:", w.softmax_type, "tau:", w.tau, "tau_step:", w.tau_step)
print("deep_weights requires_grad:", model._weight_exp_d.deep_weights.requires_grad)

print("\n=== B1 gating ===")
print("num gates:", len(model.gate), "gate[0]:", model.gate[0])

print("\n=== B10 optimizer param groups ===")
ctr = CTRTrainer(model,
                 optimizer_params={"lr": 1e-2, "weight_decay": 1e-5},
                 optimizer_params_darts={"lr": 1e-3, "weight_decay": 1e-5},
                 n_epoch=50, earlystop_patience=10, device="cpu",
                 model_path="./model_para/verify_")
darts_flag = lambda name: name.startswith('_weight_')
main_names = [n for n, _ in model.named_parameters() if darts_flag(n)]
all_names = [n for n, _ in model.named_parameters()]
main_group = ctr.optimizer.param_groups[0]['params']
darts_group = ctr.optimizer_loss_weight.param_groups[0]['params']
main_ids = {id(p) for p in main_group}
darts_ids = {id(p) for p in darts_group}
arch_ids = {id(p) for n, p in model.named_parameters() if darts_flag(n)}
print("total named params:", len(all_names))
print("params starting with _weight_:", main_names)
print("main optimizer param tensors:", len(main_group), "of", len(all_names),
      "-> contains arch params:", bool(main_ids & arch_ids))
print("darts optimizer param tensors:", len(darts_group),
      "-> contains arch params:", bool(darts_ids & arch_ids))
print("darts group names:", [n for n, p in model.named_parameters() if id(p) in darts_ids])
# emulate the ~ bug
print("~True =", ~True, "(truthy:", bool(~True), "); ~False =", ~False, "(truthy:", bool(~False), ")")

print("\n=== B9 arch-loss vs train-loss target (shape/mask check) ===")
print("train loss: 6 BCE terms on per-domain masks (ctr_trainer.py:107-114)")
print("arch loss : single BCELoss on full batch y_pred [bs,2] vs y [bs,2] (ctr_trainer.py:143)")

print("\n=== C5/C4 defaults echo ===")
print("CTRTrainer default optimizer_params:", {"lr": 1e-3, "weight_decay": 1e-5},
      "(overridden by run_movielens.py to lr=1e-2)")
