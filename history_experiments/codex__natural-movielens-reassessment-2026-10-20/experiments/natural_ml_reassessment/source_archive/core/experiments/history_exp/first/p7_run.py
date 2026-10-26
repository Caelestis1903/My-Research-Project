"""P7 runner: backbone generalization + D/T extension.

Backbones (source noted in p7_models.py): mmoe / ple / star / sharebottom,
all consuming the same data pipeline and as-published trainer (D=3, T=2).
Arms:
  --backbone X                  : backbone validation (vs M3oE paper Table 2)
  --backbone X --momentum_mode per_domain_only --fix_optim_grouping 1
                                : component-1 port (optimizer-level, backbone-agnostic)
  --backbone mdmtoe             : the MDMTRec wrapper itself (reference)
  --domain_version five         : D=5 MovieLens split (age bands {1},{18},{25},
                                  {35},{45,50,56}); component-2 N/A (no fusion
                                  weight on these backbones - reported as such)

Component 2 (fusion-weight curriculum) requires a per-domain fusion scalar,
which none of the third-party backbones expose -> honestly reported N/A.
"""
import argparse
import csv
import os
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
from MDMTRec.utils.data import DataGenerator, TorchDataset
from run_movielens import get_movielens_data_rank_multidomain
from p1_run import setup_seed
from p4_run import PerDomainMomentum
from p7_models import Backbone

AGE5_GROUPS = [{1}, {18}, {25}, {35}, {45, 50, 56}]


def load_data_ml5(path):
    """MovieLens with 5 age-band domains (P7 task 2: D=5)."""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    data = pd.read_csv(path + "/ml-1m.csv")
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]
    x_used_cols = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id',
                   'domain_indicator']

    def map5(age):
        for i, g in enumerate(AGE5_GROUPS):
            if age in g:
                return i
        return 0
    data["domain_indicator"] = data["age"].apply(map5)
    dense_features = ['age']
    sparse_features = ['user_id', 'movie_id', 'gender', 'occupation', 'zip', "cate_id",
                       "domain_indicator"]
    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: int(x))
    sca = MinMaxScaler()
    data[dense_features] = sca.fit_transform(data[dense_features])
    del data["title"]
    del data["timestamp"]
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
    data['click'] = data['rating'].apply(lambda x: int(x > 3))
    data['like'] = data['rating'].apply(lambda x: int(x > 4))
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    dense_feas = [__import__("MDMTRec.basic.features", fromlist=["DenseFeature"]).DenseFeature(
        x_used_cols.index(f)) for f in dense_features]
    from MDMTRec.basic.features import DenseFeature, SparseFeature
    dense_feas = [DenseFeature(x_used_cols.index(f)) for f in dense_features]
    sparse_feas = [SparseFeature(x_used_cols.index(f), vocab_size=data[f].nunique(),
                                 embed_dim=16) for f in sparse_features]
    y = data[['click', 'like']]
    del data['click']
    del data['like']
    del data['rating']
    x, y = data[x_used_cols[:-1] + ['domain_indicator']].values, y.values
    tr, va = int(len(x) * 0.8), int(len(x) * 0.9)
    return dense_feas, sparse_feas, x[:tr], y[:tr], x[tr:va], y[tr:va], x[va:], y[va:], 5


class GeneralizedTrainer(CTRTrainer):
    """Always takes the masked per-(d,t) path for ANY (domain_num, task_num).
    Removes the C3 hardcode in OUR runner (repo file untouched).
    Optional component-1 port: momentum -> D backwards per step, shared-param
    grads replaced by sum_d m_d (PerDomainMomentum), private params keep the
    accumulated gradient."""

    momentum = None

    def train_one_epoch(self, data_loader, domain_num, task_num, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0, disable=True)
        targets_list = [[[] for _ in range(domain_num)] for _ in range(task_num)]
        predict_list = [[[] for _ in range(domain_num)] for _ in range(task_num)]
        for i, (x_dict, y) in enumerate(tk0):
            x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
            y_pred = self.model(x_dict)
            y_pred = y_pred.to(self.device)
            domain_mask_list = [(x_dict[-1, :].clone().detach() == d) for d in range(domain_num)]
            logloss_list = []
            for t in range(task_num):
                for d in range(domain_num):
                    targets_list[t][d] = y[domain_mask_list[d], t]
                    predict_list[t][d] = y_pred[domain_mask_list[d], t]
                    logloss_list.append(self.criterion(predict_list[t][d],
                                                       targets_list[t][d].float()))
            loss = sum(torch.stack(logloss_list))
            if self.momentum is not None:
                self.momentum.backward_accumulate(logloss_list, task_num)
                self.momentum.apply_grads()
            else:
                self.model.zero_grad()
                loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--backbone", choices=["mdmtoe", "mmoe", "ple", "star", "sharebottom"],
                    default="mmoe")
    ap.add_argument("--momentum_mode", choices=["none", "per_domain_only"], default="none")
    ap.add_argument("--fix_optim_grouping", type=int, default=1)
    ap.add_argument("--domain_version", choices=["three", "five"], default="three")
    ap.add_argument("--batch_size", type=int, default=40960)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset_path", default="D:/CODE/phase0-3/m3oe/examples/data/ml-1m")
    ap.add_argument("--tag", default="p7")
    ap.add_argument("--outdir", default="D:/CODE/phase0-3/experiments/results/p7")
    args = ap.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    if args.domain_version == "five":
        dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
            load_data_ml5(args.dataset_path)
    else:
        dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
            get_movielens_data_rank_multidomain("all", "all", args.dataset_path)
    task_num = 2
    dg = DataGenerator(x_tr, y_tr)
    train_dl, val_dl, test_dl = dg.generate_dataloader(x_val=x_va, y_val=y_va, x_test=x_te,
                                                       y_test=y_te, batch_size=args.batch_size)
    test_dl_eval = DataLoader(TorchDataset(x_te, y_te), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    if args.backbone == "mdmtoe":
        model = MDMTRec(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num,
                        fcn_dims=[512, 256, 256, 64], expert_num=4,
                        exp_d=1, exp_t=1, bal_d=1, bal_t=1)
        shared_names = None  # use p1_run default definition
    else:
        model = Backbone(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num,
                         backbone=args.backbone, expert_num=4)
        shared_names = model.shared_param_names()

    ctr = CTRTrainer(model, optimizer_params={"lr": 1e-2, "weight_decay": 1e-5},
                     optimizer_params_darts={"lr": 1e-3, "weight_decay": 1e-5},
                     n_epoch=args.epoch, earlystop_patience=10, device=args.device,
                     model_path="./model_para/" + str(uuid.uuid4()),
                     scheduler_params={"step_size": 4, "gamma": 0.85})

    momentum = None
    if args.momentum_mode == "per_domain_only":
        from run_movielens import get_movielens_data_rank_multidomain as _gl  # noqa
        dom = x_tr[:, -1].astype(int)
        counts = {int(d): int((dom == d).sum()) for d in np.unique(dom)}
        tot = sum(counts.values())
        p = [counts.get(d, 0) / tot for d in range(domain_num)]
        if shared_names is None:
            from p1_run import shared_param_names as spn
            shared_names = spn(model)
        momentum = PerDomainMomentum(model, domain_num, gamma0=0.9, gamma_max=0.99,
                                     domain_p=p, device=args.device)
        momentum.names = [n for n in shared_names if n in dict(model.named_parameters())]
        momentum.named = dict(model.named_parameters())
        momentum.m = {d: {n: torch.zeros_like(momentum.named[n]) for n in momentum.names}
                      for d in range(domain_num)}
        # beta1=0 for shared params
        shared_set = set(momentum.names)
        shared_params = [q for n, q in model.named_parameters() if n in shared_set]
        private_params = [q for n, q in model.named_parameters()
                          if n not in shared_set and not n.startswith("_weight_")]
        ctr.optimizer = torch.optim.Adam(
            [{"params": shared_params, "betas": (0.0, 0.999)},
             {"params": private_params, "betas": (0.9, 0.999)}], lr=1e-2, weight_decay=1e-5)

    # generalized masked path for any (D, T); for (3,2) identical math to the
    # original hardcoded branch
    gen = GeneralizedTrainer.__new__(GeneralizedTrainer)
    gen.__dict__.update(ctr.__dict__)
    gen.momentum = momentum

    # arch update shim: backbones have no arch params
    class NoOpOpt:
        def step(self):
            pass
    if not any(n.startswith("_weight_") for n, _ in model.named_parameters()):
        gen.optimizer_loss_weight = NoOpOpt()

    # fit: original code path (valid_data_iter etc.) but with generalized
    # train_one_epoch; evaluate_multi_domain from the base class is already
    # generic over (D, T)
    valid_data_iter = iter(train_dl)
    domain_num_, task_num_ = domain_num, task_num
    for epoch_i in range(args.epoch):
        gen.train_one_epoch(train_dl, domain_num=domain_num_, task_num=task_num_)
        try:
            _val_x, _val_y = next(valid_data_iter)
        except StopIteration:
            del valid_data_iter
            import gc
            gc.collect()
            valid_data_iter = iter(train_dl)
            _val_x, _val_y = next(valid_data_iter)
        _val_x = _val_x.transpose(0, 1).to(args.device)
        _val_y = _val_y.to(args.device)
        _y_pred = model(_val_x).to(args.device)
        if domain_num_ == 3 and task_num_ == 2:
            _loss = ctr.criterion(_y_pred, _val_y.float())  # as-published arch loss
            model.zero_grad()
            _loss.backward()
            gen.optimizer_loss_weight.step()
        logloss_list, auc_list, _, _ = ctr.evaluate_multi_domain(
            model, val_dl, domain_num=domain_num_, task_num=task_num_)
        auc = sum(auc_list) / (domain_num_ * task_num_)
        rng = torch.get_rng_state()
        ctr.evaluate_multi_domain(model, test_dl_eval, domain_num=domain_num_,
                                  task_num=task_num_)
        torch.set_rng_state(rng)
        print("epoch:", epoch_i, "validation: auc:", auc, flush=True)
        if ctr.early_stopper.stop_training(auc, model.state_dict()):
            print(f"validation: best auc: {ctr.early_stopper.best_auc}", flush=True)
            model.load_state_dict(ctr.early_stopper.best_weights)
            break
    torch.save(model.state_dict(), ctr.model_path + "model.pth")

    ll, aucs, tot_ll, tot_auc = ctr.evaluate_multi_domain(model, test_dl,
                                                          domain_num=domain_num_,
                                                          task_num=task_num)
    row = {"backbone": args.backbone, "domain_version": args.domain_version,
           "momentum": args.momentum_mode, "seed": args.seed,
           "test_auc_mean": round(float(tot_auc), 6)}
    for d in range(domain_num_):
        for t in range(task_num_):
            row[f"auc_d{d}t{t}"] = round(float(aucs[d + t * domain_num_]), 6)
    out_csv = os.path.join(args.outdir, "P7_backbones.csv")
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print("[p7]", row, flush=True)


if __name__ == "__main__":
    main()
