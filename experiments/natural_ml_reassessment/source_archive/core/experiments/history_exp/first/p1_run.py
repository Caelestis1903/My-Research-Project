"""P1 instrumented runner for M3oE on MovieLens-1M.

Bypass-only instrumentation. With --instrument 0 the ORIGINAL CTRTrainer code
paths run unmodified (a pass-through loss observer may wrap the criterion /
train_one_epoch externally without touching computation), so the trajectory is
identical to examples/run_movielens.py. With --instrument 1 an overridden
train_one_epoch/fit adds:

  D0.2  per-epoch snapshot of the 4 architecture params (raw + sigmoid);
        epoch=-1 is the pre-training init snapshot
  D0.4  every K steps: per-domain loss L_d = BCE(t0|d) + BCE(t1|d) (the two
        masked-mean BCE terms of that domain, i.e. its contribution to the
        training loss), backward per domain with retain_graph=True, flatten
        shared-param grads, then zero_grad. The real step then runs exactly as
        the original (zero_grad -> loss.backward -> optimizer.step).
  D0.6  per-epoch per-(d,t) train loss (sample-weighted mean of masked BCE)
        and per-(d,t) validation AUC
  D0.7  per-epoch per-(d,t) TEST AUC (eval-only bypass) for post-hoc
        early-stop fairness analysis

"Shared parameters" (declared once here; P1 D0.4 definition):
  INCLUDED: all 7 embedding tables; shared_weight/shared_bias (star W_sh/b_sh);
            skip_conn; star_mlp; expert x4 (shared experts); domain_expert x3;
            task_expert x2.
  EXCLUDED: slot_weight/slot_bias (per-domain private W_d/b_d of the star
            fusion), tower (per-(d,t)), gate (per-(d,t)), and the 4 _weight_*
            architecture scalars (optimization variables, outside the P1
            include list).
"""
import argparse
import csv
import os
import random
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe", "examples"))

import numpy as np
import torch
import tqdm

from torch.utils.data import DataLoader

from MDMTRec.trainers import CTRTrainer
from MDMTRec.models.multi_domain import MDMTRec
from MDMTRec.utils.data import DataGenerator, TorchDataset
from run_movielens import get_movielens_data_rank_multidomain

ARCH_PARAMS = ["_weight_exp_d", "_weight_exp_t", "_weight_bal_d", "_weight_bal_t"]
PAIRS = ["0-1", "0-2", "1-2"]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def shared_param_names(model):
    excl = ("tower.", "gate.", "slot_weight.", "slot_bias.", "_weight_")
    return [n for n, _ in model.named_parameters() if not n.startswith(excl)]


class LossObserver:
    """Pass-through: records criterion outputs; never alters values."""

    def __init__(self, trainer):
        self.calls = []
        self.recording = False
        self._orig = trainer.criterion.forward
        trainer.criterion.forward = self._wrapped

    def _wrapped(self, *a, **k):
        v = self._orig(*a, **k)
        if self.recording:
            self.calls.append(float(v.detach()))
        return v


class Instrumented:
    def __init__(self, trainer, model, domain_num, task_num, k, device, outdir, tag,
                 batch_size, loss_stream_path, collect_d04=True):
        self.tr = trainer
        self.model = model
        self.domain_num = domain_num
        self.task_num = task_num
        self.k = k
        self.collect_d04 = collect_d04
        self.device = device
        self.outdir = outdir
        self.tag = tag
        self.batch_size = batch_size
        self.loss_stream_path = loss_stream_path
        self.shared_names = shared_param_names(model)
        self.named = dict(model.named_parameters())
        self.cur_epoch = -1
        self.d02_rows = []          # epoch, param_name, raw, sigmoid
        self.d04_by_epoch = {}      # epoch -> list of stat rows per domain
        self.d04_cos_by_epoch = {}  # epoch -> list of {domain_pair, cos_mean, cos_std}
        self.d06_rows = []          # epoch, domain, task, train_loss, val_auc
        self.d06_test_rows = []     # epoch, domain, task, test_auc
        self.loss_stream = []       # per-step summed BCE (6 terms)
        self.epoch_val_mean = []

    # ---- D0.2 ----
    def snap_arch(self, epoch):
        for pn in ARCH_PARAMS:
            w = getattr(self.model, pn).deep_weights.detach()
            self.d02_rows.append({"epoch": epoch, "param_name": pn,
                                  "raw_value": float(w.item()),
                                  "sigmoid_value": float(torch.sigmoid(w).item())})

    # ---- D0.4 ----
    def collect_grads(self, logloss_terms):
        gvecs = []
        for d in range(self.domain_num):
            self.model.zero_grad()
            L_d = logloss_terms[d] + logloss_terms[self.domain_num + d]
            L_d.backward(retain_graph=True)
            gvecs.append(torch.cat([self.named[n].grad.detach().reshape(-1)
                                    if self.named[n].grad is not None
                                    else torch.zeros(self.named[n].numel(), device=self.device)
                                    for n in self.shared_names]))
        self.model.zero_grad()
        return gvecs

    def flush_d04(self, g_buf, ns_buf, cos_buf):
        for d in range(self.domain_num):
            G = torch.stack(g_buf[d])                       # [K', P]
            mean_norm = float(G.mean(dim=0).norm())
            sq = float((G.norm(dim=1) ** 2).mean())
            self.d04_by_epoch.setdefault(self.cur_epoch, []).append({
                "domain": d,
                "n_samples_avg": float(np.mean(ns_buf[d])),
                "mean_grad_norm": float(G.norm(dim=1).mean()),
                "norm_of_mean_grad": mean_norm,
                "snr": mean_norm ** 2 / sq})
        for pair in PAIRS:
            vals = cos_buf[pair]
            if vals:
                self.d04_cos_by_epoch.setdefault(self.cur_epoch, []).append({
                    "domain_pair": pair, "cos_mean": float(np.mean(vals)),
                    "cos_std": float(np.std(vals))})

    # ---- instrumented train_one_epoch: copy of ctr_trainer.py:68-122 with taps ----
    def train_one_epoch(self, data_loader, domain_num, task_num, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0, disable=True)
        targets_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        predict_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        s_loss = [[0.0] * domain_num for _ in range(task_num)]
        s_cnt = [[0] * domain_num for _ in range(task_num)]
        g_buf = [[] for _ in range(self.domain_num)]
        ns_buf = [[] for _ in range(self.domain_num)]
        cos_buf = {p: [] for p in PAIRS}

        for i, (x_dict, y) in enumerate(tk0):
            x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
            y_pred = self.model(x_dict)
            y_pred = y_pred.to(self.device)

            if not (domain_num == 3 and task_num == 2):
                loss = self.tr.criterion(y_pred, y.float())
            else:
                domain_mask_list = []
                domain_id = x_dict[-1, :].clone().detach()
                for d in range(domain_num):
                    domain_mask_list.append((domain_id == d))
                for t in range(task_num):
                    for d in range(domain_num):
                        targets_list[t][d] = y[domain_mask_list[d], t]
                        predict_list[t][d] = y_pred[domain_mask_list[d], t]
                logloss_list = list()
                for t in range(task_num):
                    for d in range(domain_num):
                        term = self.tr.criterion(predict_list[t][d], targets_list[t][d].float())
                        logloss_list.append(term)
                        n_d = int(domain_mask_list[d].sum())
                        s_loss[t][d] += float(term.detach()) * n_d
                        s_cnt[t][d] += n_d
                if self.collect_d04 and (i + 1) % self.k == 0:  # D0.4 bypass collection
                    gvecs = self.collect_grads(logloss_list)
                    for d in range(self.domain_num):
                        g_buf[d].append(gvecs[d])
                        ns_buf[d].append(int(domain_mask_list[d].sum()))
                    for a in range(self.domain_num):
                        for b in range(a + 1, self.domain_num):
                            cos_buf[f"{a}-{b}"].append(
                                float(torch.cosine_similarity(gvecs[a], gvecs[b], dim=0)))
                loss = sum(torch.stack(logloss_list))

            self.model.zero_grad()
            loss.backward()
            self.tr.optimizer.step()
            total_loss += loss.item()
            self.loss_stream.append(float(loss.item()))
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

        if g_buf[0]:
            self.flush_d04(g_buf, ns_buf, cos_buf)
        self.last_train_loss = {(t, d): (s_loss[t][d] / s_cnt[t][d] if s_cnt[t][d] else float("nan"))
                                for t in range(task_num) for d in range(domain_num)}

    # ---- instrumented fit: copy of ctr_trainer.py:124-159 with taps ----
    def fit(self, train_dataloader, val_dataloader, test_dataloader, n_epoch):
        valid_data_iter = iter(train_dataloader)
        self.snap_arch(-1)
        for epoch_i in range(n_epoch):
            self.cur_epoch = epoch_i
            self.train_one_epoch(train_dataloader, domain_num=self.domain_num, task_num=self.task_num)
            if self.tr.scheduler is not None:
                if epoch_i % self.tr.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.tr.optimizer.state_dict()['param_groups'][0]['lr']))
                self.tr.scheduler.step()
            try:
                _val_x, _val_y = next(valid_data_iter)
            except StopIteration:
                del valid_data_iter
                import gc
                gc.collect()
                valid_data_iter = iter(train_dataloader)
                _val_x, _val_y = next(valid_data_iter)
            _val_x, _val_y = _val_x.transpose(0, 1).to(self.device), _val_y.to(self.device)
            _y_pred = self.model(_val_x).to(self.device)
            _loss = self.tr.criterion(_y_pred, _val_y.float())
            self.model.zero_grad()
            _loss.backward()
            self.tr.optimizer_loss_weight.step()

            logloss_list, auc_list, _, _ = self.tr.evaluate_multi_domain(
                self.model, val_dataloader, domain_num=self.domain_num, task_num=self.task_num)
            auc = sum(auc_list) / (self.domain_num * self.task_num)
            # D0.7 test eval (bypass). Creating a DataLoader iterator draws
            # _base_seed from the GLOBAL RNG (dataloader.py:697, even with
            # num_workers=0), which would shift the training shuffle stream
            # from the next epoch on. Snapshot and restore the global RNG
            # around this eval-only call so it is trajectory-neutral.
            rng_state = torch.get_rng_state()
            tll_list, test_auc_list, _, _ = self.tr.evaluate_multi_domain(
                self.model, test_dataloader, domain_num=self.domain_num, task_num=self.task_num)
            torch.set_rng_state(rng_state)
            for t in range(self.task_num):
                for d in range(self.domain_num):
                    self.d06_rows.append({"epoch": epoch_i, "domain": d, "task": t,
                                          "train_loss": self.last_train_loss[(t, d)],
                                          "val_auc": auc_list[t * self.domain_num + d]})
                    self.d06_test_rows.append({"epoch": epoch_i, "domain": d, "task": t,
                                               "test_auc": test_auc_list[t * self.domain_num + d],
                                               "test_logloss": tll_list[t * self.domain_num + d]})
            self.epoch_val_mean.append(auc)
            self.snap_arch(epoch_i)
            print('epoch:', epoch_i, 'validation: auc:', auc, flush=True)
            if self.tr.early_stopper.stop_training(auc, self.model.state_dict()):
                print(f'validation: best auc: {self.tr.early_stopper.best_auc}', flush=True)
                self.model.load_state_dict(self.tr.early_stopper.best_weights)
                break
        torch.save(self.model.state_dict(), self.tr.model_path + "model.pth")

    def dump(self, seed):
        def w(path, header, rows):
            with open(path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=header)
                wr.writeheader()
                wr.writerows(rows)

        w(os.path.join(self.outdir, f"D02_arch_traj_{self.tag}.csv"),
          ["seed", "epoch", "param_name", "raw_value", "sigmoid_value"],
          [{"seed": seed, **r} for r in self.d02_rows])
        w(os.path.join(self.outdir, f"D04_grad_stats_{self.tag}.csv"),
          ["seed", "batch_size", "epoch", "domain", "n_samples_avg",
           "mean_grad_norm", "norm_of_mean_grad", "snr"],
          [{"seed": seed, "batch_size": self.batch_size, "epoch": e, **r}
           for e, rows in self.d04_by_epoch.items() for r in rows])
        w(os.path.join(self.outdir, f"D04_grad_cosine_{self.tag}.csv"),
          ["seed", "batch_size", "epoch", "domain_pair", "cos_mean", "cos_std"],
          [{"seed": seed, "batch_size": self.batch_size, "epoch": e, **r}
           for e, rows in self.d04_cos_by_epoch.items() for r in rows])
        w(os.path.join(self.outdir, f"D06_curves_{self.tag}.csv"),
          ["seed", "epoch", "domain", "task", "train_loss", "val_auc"],
          [{"seed": seed, **r} for r in self.d06_rows])
        w(os.path.join(self.outdir, f"D06_test_{self.tag}.csv"),
          ["seed", "epoch", "domain", "task", "test_auc", "test_logloss"],
          [{"seed": seed, **r} for r in self.d06_test_rows])
        if self.loss_stream_path:
            with open(self.loss_stream_path, "w") as f:
                for v in self.loss_stream:
                    f.write(f"{v!r}\n")


def dump_observer_stream(trainer, obs, path):
    """Group criterion calls (6 per training step) into per-step sums."""
    if not path or not obs.calls:
        return
    with open(path, "w") as f:
        for i in range(0, len(obs.calls) - len(obs.calls) % 6, 6):
            f.write(f"{sum(obs.calls[i:i+6])!r}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--batch_size", type=int, default=40960)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--instrument", type=int, default=0)
    ap.add_argument("--d04", type=int, default=1, help="with --instrument 1: also collect D0.4 gradient stats")
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=1e-2)
    ap.add_argument("--learning_rate_darts", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--expert_num", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset_path", default="D:/CODE/phase0-3/m3oe/examples/data/ml-1m")
    ap.add_argument("--tag", default="s")
    ap.add_argument("--outdir", default="D:/CODE/phase0-3/experiments/results/p1")
    ap.add_argument("--loss_stream", default=None)
    args = ap.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    dense_feas, sparse_feas, x_tr, y_tr, x_va, y_va, x_te, y_te, domain_num = \
        get_movielens_data_rank_multidomain("all", "all", args.dataset_path)
    task_num = 2
    dg = DataGenerator(x_tr, y_tr)
    train_dl, val_dl, test_dl = dg.generate_dataloader(x_val=x_va, y_val=y_va, x_test=x_te,
                                                       y_test=y_te, batch_size=args.batch_size)
    # RNG-neutral test loader for the per-epoch D0.7 test eval: creating an
    # iterator of a num_workers>0 loader draws base_seed from the GLOBAL RNG,
    # which would shift the training batch order from epoch 1 on (this exact
    # bug was caught by the equivalence check). num_workers=0 + shuffle=False
    # consumes no global RNG, so the instrumented fit keeps the as-published
    # trajectory. Batches are identical to test_dl (same dataset/order).
    test_dl_eval = DataLoader(TorchDataset(x_te, y_te), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    model = MDMTRec(dense_feas + sparse_feas, domain_num=domain_num, task_num=task_num,
                    fcn_dims=[512, 256, 256, 64], expert_num=args.expert_num,
                    exp_d=1, exp_t=1, bal_d=1, bal_t=1)
    ctr = CTRTrainer(model, optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
                     optimizer_params_darts={"lr": args.learning_rate_darts, "weight_decay": args.weight_decay},
                     n_epoch=args.epoch, earlystop_patience=10, device=args.device,
                     model_path="./model_para/" + str(uuid.uuid4()),
                     scheduler_params={"step_size": 4, "gamma": 0.85})
    obs = LossObserver(ctr)

    if args.instrument:
        ins = Instrumented(ctr, model, domain_num, task_num, args.k, args.device,
                           args.outdir, args.tag, args.batch_size, args.loss_stream,
                           collect_d04=bool(args.d04))
        ins.fit(train_dl, val_dl, test_dl_eval, args.epoch)
        ins.dump(args.seed)
    else:
        orig_toe = ctr.train_one_epoch

        def toe_recording(dl, domain_num, task_num, log_interval=10):
            obs.recording = True
            orig_toe(dl, domain_num=domain_num, task_num=task_num, log_interval=log_interval)
            obs.recording = False

        ctr.train_one_epoch = toe_recording
        ctr.fit(train_dl, val_dataloader=val_dl, exp_d=1, exp_t=1, bal_d=1, bal_t=1,
                domain_num=domain_num, task_num=task_num)
        dump_observer_stream(ctr, obs, args.loss_stream)

    ll, aucs, tot_ll, tot_auc = ctr.evaluate_multi_domain(ctr.model, test_dl, domain_num, task_num=task_num)
    print("FINAL per-(d,t) test auc:", [round(a, 6) for a in aucs], flush=True)
    print("FINAL test auc:", tot_auc, "| test logloss:", tot_ll, flush=True)


if __name__ == "__main__":
    main()
