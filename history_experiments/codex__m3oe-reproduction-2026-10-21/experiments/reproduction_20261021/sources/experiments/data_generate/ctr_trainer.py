import os
import numpy as np
import pandas as pd
import torch
import tqdm, gc
from sklearn.metrics import roc_auc_score, log_loss
from ..basic.callback import EarlyStopper
import torch.nn.functional as F


class CTRTrainer(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

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
            cda_weight=0.0
    ):
        self.scarce_domain = scarce_domain
        self.cda_weight = cda_weight

        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        # darts_para_flag = lambda name: name.startswith('loss')
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)

        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.evaluate_fn_logloss = log_loss
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, domain_num, task_num, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)

        # 定义 DBO 超参数 (参考论文实验部分)
        lambda1, alpha1 = 0.1, 0.5  # For Complexity
        lambda2, alpha2 = 0.1, 0.5  # For Specificity
        norm_l, norm_u = 0.1, 1.0  #

        targets, predicts = list(), list()
        # target_list: [task_num, domain_num, [list of ground truth]]
        # predict_list: [task_num, domain_num, [list of prediction]]
        targets_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        predict_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        # [[[d0t0], [d1t0], [d2t0]], [[d0t1], [d1t1], [d2t1]]] T*D

        for i, (x_dict, y) in enumerate(tk0):
            logloss_list = []
            x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
            # 1. 识别稀缺样本
            domain_id = x_dict[-1, :]  # 获取当前 Batch 所有样本的领域 ID
            scarce_mask = (domain_id == self.scarce_domain)
            # 2. 只有当确实存在稀缺样本，且在训练模式下，才进行扩充
            if self.scarce_domain is not None and scarce_mask.sum() > 0 and self.model.training:
                # 复制稀缺样本的输入 x 和标签 y
                x_aug = x_dict[:, scarce_mask].clone()
                y_aug = y[scarce_mask].clone()
                # 记录原始 Batch Size
                original_bs = x_dict.shape[1]
                aug_bs = x_aug.shape[1]
                #加入生成样本
                x_input = torch.cat([x_dict, x_aug], dim=1)
                y_input = torch.cat([y, y_aug], dim=0)
                #aug_mask标记是否是生成样本
                aug_mask = torch.zeros(original_bs + aug_bs, dtype=torch.bool, device=self.device)
                aug_mask[original_bs:] = True

            else:
                x_input = x_dict
                y_input = y
                aug_mask = None
            # ===========================================
            y_pred, complexity, specificity = self.model(x_input, aug_mask=aug_mask, noise_std=0.1)
            y_pred = y_pred.to(self.device)
            current_domain_id = x_input[-1, :].clone().detach()

            # 复用之前记录的原始Batch Size。生成的样本是纯噪声随机，在添加一定的因果限制之前，DBO模块最好只学习原始的样本防止学习到负面噪声效果影响
            original_bs = x_dict.shape[1]
            comp_raw = complexity[:original_bs]
            spec_raw = specificity[:original_bs]

            # 权重计算，根据实验结果注释掉了复杂性权重
            c_mean = comp_raw.mean()
            c_std = comp_raw.std() + 1e-9
            c_norm = (comp_raw - c_mean) / c_std + 0.5
            c_prime = torch.clamp(c_norm, min=norm_l, max=norm_u)
            w_cpl_raw = lambda1 + alpha1 * c_prime
            w_spf_raw = lambda2 + alpha2 * spec_raw
            weights_raw =  w_spf_raw  # Shape: [original_bs]
            #weights_raw = w_cpl_raw * w_spf_raw

            #处理增强数据的权重
            # 判断是否有增强数据
            if y_pred.shape[0] > original_bs:
                aug_len = y_pred.shape[0] - original_bs

                # DBO不使用生成的数据，并且对应的权重使用原始权重的均值
                avg_raw_weight = weights_raw.mean().detach()
                weights_fake = torch.full((aug_len,), avg_raw_weight, device=self.device)

                # 6. 拼接：组合成完整的 sample_weights
                # Shape: [Batch_Total]
                sample_weights = torch.cat([weights_raw, weights_fake], dim=0)
            else:
                sample_weights = weights_raw

            # =======================================================
            # sample_weights = torch.ones_like(sample_weights) #权重置1，用于排除DBO模块的影响

            if not (domain_num == 3 and task_num == 2):
                # 单任务/普通多任务情况
                # 使用 reduction='none' 获取每个样本的 loss
                loss_per_sample = F.binary_cross_entropy(y_pred, y_input.float(), reduction='none')
                # 对 loss 进行加权
                loss_weighted = loss_per_sample.mean(dim=-1) * sample_weights
                loss_original = loss_weighted.mean()
            else:
                # === MTMD setting (修改部分) ===
                domain_mask_list = []
                domain_id = x_dict[-1, :].clone().detach()

                for d in range(domain_num):
                    domain_mask = (current_domain_id == d)
                    domain_mask_list.append(domain_mask)

                for t in range(task_num):
                    for d in range(domain_num):
                        mask = domain_mask_list[d]

                        # 如果当前 batch 中该领域没有样本，加入 0 loss 避免报错
                        if mask.sum() == 0:
                            # 放入一个带梯度的 0 tensor，保证 graph 完整性
                            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                            logloss_list.append(zero_loss)
                            continue

                        # 获取当前领域和任务的数据
                        target_sub = y_input[mask, t]
                        pred_sub = y_pred[mask, t]

                        weight_sub = sample_weights[mask]

                        per_sample_loss = F.binary_cross_entropy(
                            pred_sub,
                            target_sub.float(),
                            reduction='none'
                        )

                        weighted_sample_loss = per_sample_loss * weight_sub
                        logloss_list.append(weighted_sample_loss.mean())

                if len(logloss_list) > 0:
                    loss_original = sum(torch.stack(logloss_list))
                else:
                    loss_original = torch.tensor(0.0, device=self.device, requires_grad=True)

            loss = loss_original

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
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