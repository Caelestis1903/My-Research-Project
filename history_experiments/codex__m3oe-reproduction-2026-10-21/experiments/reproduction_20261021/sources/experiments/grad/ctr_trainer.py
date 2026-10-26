import os
import numpy as np
import pandas as pd
import torch
import tqdm, gc
from sklearn.metrics import roc_auc_score, log_loss
from ..basic.callback import EarlyStopper
import torch.nn.functional as F


class CTRTrainer(object):
    def __init__(
            self,
            model,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=None,
            optimizer_params_darts=None,
            scheduler_fn=None,
            scheduler_params=None,
            n_epoch=10,
            earlystop_patience=10,
            device="cpu",
            gpus=None,
            model_path="./",
            scarce_domain=None,
            cda_weight=0.0,
            grad_tau=0.5,  # tau超参数，控制平衡强度 (0.0 - 1.0)
            use_grad_norm_balance=True  # 开关
    ):
        self.scarce_domain = scarce_domain
        self.cda_weight = cda_weight
        self.grad_tau = grad_tau
        self.use_grad_norm_balance = use_grad_norm_balance

        self.model = model
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)
        self.model.to(self.device)

        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}

        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)

        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()
        self.evaluate_fn = roc_auc_score
        self.evaluate_fn_logloss = log_loss
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, domain_num, task_num, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)

        # DBO 超参数
        lambda1, alpha1 = 0.1, 0.5
        lambda2, alpha2 = 0.1, 0.5
        norm_l, norm_u = 0.1, 1.0

        for i, (x_dict, y) in enumerate(tk0):
            # 清空梯度
            self.optimizer.zero_grad()

            x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)

            domain_id = x_dict[-1, :]
            scarce_mask = (domain_id == self.scarce_domain)

            if self.scarce_domain is not None and scarce_mask.sum() > 0 and self.model.training:
                x_aug = x_dict[:, scarce_mask].clone()
                y_aug = y[scarce_mask].clone()
                original_bs = x_dict.shape[1]
                aug_bs = x_aug.shape[1]
                x_input = torch.cat([x_dict, x_aug], dim=1)
                y_input = torch.cat([y, y_aug], dim=0)
                aug_mask = torch.zeros(original_bs + aug_bs, dtype=torch.bool, device=self.device)
                aug_mask[original_bs:] = True
            else:
                x_input = x_dict
                y_input = y
                aug_mask = None
                original_bs = x_dict.shape[1]

            y_pred, complexity, specificity = self.model(x_input, aug_mask=aug_mask, noise_std=0.1)
            y_pred = y_pred.to(self.device)
            current_domain_id = x_input[-1, :].clone().detach()

            # --- DBO 权重计算 (保持原有逻辑) ---
            comp_raw = complexity[:original_bs]
            spec_raw = specificity[:original_bs]
            c_mean = comp_raw.mean()
            c_std = comp_raw.std() + 1e-9
            c_norm = (comp_raw - c_mean) / c_std + 0.5
            w_spf_raw = lambda2 + alpha2 * spec_raw
            weights_raw = w_spf_raw

            if y_pred.shape[0] > original_bs:
                aug_len = y_pred.shape[0] - original_bs
                avg_raw_weight = weights_raw.mean().detach()
                weights_fake = torch.full((aug_len,), avg_raw_weight, device=self.device)
                sample_weights = torch.cat([weights_raw, weights_fake], dim=0)
            else:
                sample_weights = weights_raw

            #先把每个任务的loss独立存储
            task_losses = []

            if not (domain_num == 3 and task_num == 2):
                # 单任务情况，直接当作一个任务处理
                loss_per_sample = F.binary_cross_entropy(y_pred, y_input.float(), reduction='none')
                loss_weighted = loss_per_sample.mean(dim=-1) * sample_weights
                task_losses.append(loss_weighted.mean())
            else:
                # 多任务多领域情况
                domain_mask_list = []
                for d in range(domain_num):
                    domain_mask_list.append(current_domain_id == d)

                for t in range(task_num):
                    for d in range(domain_num):
                        mask = domain_mask_list[d]
                        if mask.sum() == 0:
                            # 放入一个带梯度的 0 tensor，保证 graph 完整性
                            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                            logloss_list.append(zero_loss)
                            continue  # 跳过没有样本的领域-任务组合

                        target_sub = y_input[mask, t]
                        pred_sub = y_pred[mask, t]
                        weight_sub = sample_weights[mask]

                        per_sample_loss = F.binary_cross_entropy(pred_sub, target_sub.float(), reduction='none')
                        weighted_sample_loss = per_sample_loss * weight_sub

                        # 保存该子任务的 loss
                        task_losses.append(weighted_sample_loss.mean())

            #GradCraft相关
            if self.use_grad_norm_balance and len(task_losses) > 1:
                #收集每个任务的原始梯度
                task_grads = []
                for loss in task_losses:
                    self.optimizer.zero_grad()  # 清除上一次的梯度
                    loss.backward(retain_graph=True)  # 保留图，因为后面还要用到网络计算图

                    # 提取所有可训练参数的梯度，展平或保持列表
                    grads = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grads.append(param.grad.clone())
                        else:
                            grads.append(torch.zeros_like(param))  # 某些参数可能在这个任务中没有梯度
                    task_grads.append(grads)

                #计算梯度范数
                # Norm 计算公式: sqrt(sum(g_i^2))
                grad_norms = []
                for grads in task_grads:
                    sum_sq = torch.sum(torch.stack([torch.sum(g ** 2) for g in grads]))
                    grad_norms.append(torch.sqrt(sum_sq))
                grad_norms = torch.stack(grad_norms)
                max_norm = torch.max(grad_norms)

                #GradCraft的梯度修正公式。由最大梯度与当前梯度得到比值，然后用超参数权衡一下比值和原始数据，最后得到修正梯度
                # 修正梯度g = tau * (max_norm / norm_i) * g_i + (1-tau) * g_i
                final_accumulated_grads = [torch.zeros_like(p) for p in self.model.parameters()]

                for i, grads in enumerate(task_grads):
                    norm_i = grad_norms[i]
                    # 防止除以0
                    if norm_i == 0:
                        scale = 1.0
                    else:
                        # GradCraft Magnitude Adjustment Formula
                        scale = self.grad_tau * (max_norm / norm_i) + (1 - self.grad_tau)

                    # 累加: Final_Grad += scale * Task_Grad_i
                    for p_idx, g in enumerate(grads):
                        final_accumulated_grads[p_idx] += g * scale

                #将计算好的最终梯度赋值回 optimizer
                self.optimizer.zero_grad()
                for p_idx, param in enumerate(self.model.parameters()):
                    if param.requires_grad:
                        param.grad = final_accumulated_grads[p_idx]

                loss_val = sum([l.item() for l in task_losses])

            else:
                # 如果不使用 GradCraft 或者只有一个任务，退回标准反向传播
                final_loss = sum(task_losses)
                self.optimizer.zero_grad()
                final_loss.backward()
                loss_val = final_loss.item()

            self.optimizer.step()

            total_loss += loss_val
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, exp_d, exp_t, bal_d, bal_t, domain_num, task_num, val_dataloader=None):
        valid_data_iter = iter(train_dataloader)
        for epoch_i in range(self.n_epoch):
            self.train_one_epoch(train_dataloader, domain_num=domain_num, task_num=task_num)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler
            # update loss_weight with a batch of validation data
            # print(f'loss weight: {_loss}, {[round(i.item(), 4) for i in self.model.loss_weight()]}')

            logloss_list, auc_list, _, _ = self.evaluate_multi_domain(self.model, val_dataloader, domain_num=domain_num,
                                                                      task_num=task_num)
            auc = sum(auc_list) / (domain_num * task_num)
            log_loss_val = sum(logloss_list) / (domain_num * task_num)
            # auc, log_loss = self.evaluate(self.model, val_dataloader, domain_num=domain_num, task_num=task_num)
            print('epoch:', epoch_i, 'validation: auc:', auc, 'log loss:', log_loss_val)
            if self.early_stopper.stop_training(auc, self.model.state_dict()):
                print(f'validation: best auc: {self.early_stopper.best_auc}')
                self.model.load_state_dict(self.early_stopper.best_weights)
                break

        torch.save(self.model.state_dict(), self.model_path + "model.pth")  # save best auc model

    def evaluate(self, model, data_loader, domain_num, task_num):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
                y_pred = model(x_dict)
                if y.shape[-1] != 1:
                    targets.extend(y.tolist())
                    predicts.extend(y_pred.tolist())
                else:
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(y_pred.squeeze(1).tolist())
        return self.evaluate_fn(targets, predicts), log_loss(targets, predicts)

    def evaluate_multi_domain(self, model, data_loader, domain_num, task_num):
        model.eval()
        targets, predicts = list(), list()
        # target_list: [task_num, domain_num, [list of ground truth]]
        # predict_list: [task_num, domain_num, [list of prediction]]
        targets_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        predict_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        # [[[d0t0], [d1t0], [d2t0]], [[d0t1], [d1t1], [d2t1]]] T*D
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            # gate_values = list()
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []

                x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
                domain_id = x_dict[-1, :].clone().detach()

                y_pred = model(x_dict)
                # y_pred, _gate = model(x_dict, test_flag=True)
                y_pred = y_pred.to(self.device)
                # gate_values.append(_gate)

                for d in range(domain_num):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                for t in range(task_num):
                    for d in range(domain_num):
                        y_ = y[domain_mask_list[d], t].tolist()
                        y_pred_ = y_pred[domain_mask_list[d], t].tolist()
                        targets_list[t][d].extend(y_)
                        predict_list[t][d].extend(y_pred_)

                if y.shape[-1] != 1:
                    targets.extend(y.tolist())
                    predicts.extend(y_pred.tolist())
                else:
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(y_pred.squeeze(1).tolist())
                    # gate_values = pd.concat(gate_values)
            # gate_values.to_csv('gate.csv', index=False)
            # print(f'saved to gate.csv with shape of {gate_values.shape}.')
        logloss_list, auc_list = list(), list()
        for t in range(task_num):
            for d in range(domain_num):
                logloss_list.append(log_loss(targets_list[t][d], predict_list[t][d]))
                auc_list.append(self.evaluate_fn(targets_list[t][d], predict_list[t][d]))

        total_logloss = log_loss(targets, predicts) if predicts else None

        total_auc = self.evaluate_fn(targets, predicts) if predicts else None
        return logloss_list, auc_list, total_logloss, total_auc

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts