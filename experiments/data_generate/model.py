import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from ...basic.layers import EmbeddingLayer
from ...basic.activation import activation_layer
from ...basic.layers import EmbeddingLayer, DBO_Complexity_Layer, DBO_Specificity_Layer
        
class MLP_N(nn.Module):
    '''
    fcn_dim: list of dimensions of mlp layers, e.g., [1024, 512, 512, 256, 256, 64] 
    return [linear, bn1d, relu]*n
    '''
    def __init__(self, fcn_dim):
        super().__init__()
        self.fcn_dim = fcn_dim
        self.n = len(fcn_dim)
        
        self.domain_specific = nn.ModuleList()
        for (i) in range(self.n-1):#尾部依次加入线性层、归一层、激活层
            self.domain_specific.append(nn.Linear(self.fcn_dim[i], self.fcn_dim[i+1]))
            self.domain_specific.append(nn.LayerNorm(self.fcn_dim[i+1]))
            self.domain_specific.append(nn.ReLU())            
        # self.domain_specific.append(nn.Linear(self.fcn_dim[-1], 1))
        
    def forward(self, x):
        output = x
        for f in self.domain_specific:
            output = f(output)
        return output


class MDMTRec(nn.Module):
    # add domain-specific expert based on MTMD
    # gate(shared expert) + task-specific + domain-specifics
    def __init__(self, features, domain_num, task_num, fcn_dims, expert_num, exp_d, exp_t, bal_d, bal_t, tau=1, tau_step=0.00005, softmax_type=3):
        super().__init__()#定义构造函数，接收特征配置、领域数量、任务数量、MLP维度、专家数量、四个初始权重参数以及温度相关参数
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1 
        self.fcn_dim = [self.input_dim] + fcn_dims 
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)
        self.exp_d = exp_d
        self.exp_t = exp_t
        self.bal_d = bal_d
        self.bal_t = bal_t
        #领域专家、任务专家、领域平衡和任务平衡的影响程度，我是要平衡模块，所以重点关注后面两个

        # === [新增] DBO 模块初始化 ===
        # 计算 embedding 之后的特征维度
        self.num_features = len(features)
        # 假设所有特征维度一致，或者你需要根据实际 features 计算 flat 维度
        total_input_dim = sum([fea.embed_dim for fea in features])

        sparse_dims = []
        dense_dims = []
        for f in features:
            # 使用类名判断，避免额外导入
            if 'Dense' in type(f).__name__:
                dense_dims.append(f.embed_dim)
            else:
                sparse_dims.append(f.embed_dim)

        self.sorted_dims = sparse_dims + dense_dims  # [16, 16, ..., 16, 1]
        self.max_dim = max(self.sorted_dims) if self.sorted_dims else 1

        # 初始化 DBO 辅助层
        self.dbo_complexity = DBO_Complexity_Layer(input_dim=self.max_dim)
        self.dbo_specificity = DBO_Specificity_Layer(input_dim=self.input_dim)
        # ==========================

        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])
        #每个领域分配独立的权重和偏置
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe，依次是创建共享专家、领域专家、任务专家，
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim))
            
        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(MLP_N(self.fcn_dim))
            
        self.task_expert = nn.ModuleList()
        for d in range(task_num):
            self.task_expert.append(MLP_N(self.fcn_dim))
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(domain_num*task_num)])
        
        self.tower = nn.ModuleList()
        for d in range(domain_num*task_num):#预测塔包含了线性层、归一层、激活层、线性层四个部分，为每个领域×任务组合都分配了一个预测塔
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                # nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.LayerNorm(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)
        
    def forward(self, x, test_flag=False, aug_mask=None, noise_std=0.1):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        mask = []
        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
#mask计算每个领域对应的掩码
        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # === 噪声===
        if self.training and aug_mask is not None:
            # 生成噪声并加入噪声，用aug_mask索引到生成的样本
            noise = torch.randn_like(input_emb[aug_mask]) * noise_std
            input_emb[aug_mask] = input_emb[aug_mask] + noise
        # =================================

        # 拆分各个特征的维度，并对其到最大的维度数
        feature_embs = torch.split(input_emb, self.sorted_dims, dim=1)
        padded_features = []
        for i, emb in enumerate(feature_embs):
            curr_dim = self.sorted_dims[i]
            if curr_dim < self.max_dim:
                # F.pad 参数顺序: (左边填充数, 右边填充数)
                pad_len = self.max_dim - curr_dim
                emb = torch.nn.functional.pad(emb, (0, pad_len))
            padded_features.append(emb.unsqueeze(1))

        #重新拼接然后进行DBO权重计算
        input_emb_seq = torch.cat(padded_features, dim=1)
        complexity = self.dbo_complexity(input_emb_seq)  # [batch_size]
        specificity = self.dbo_specificity(input_emb)  # [batch_size]
        # =========================================================

        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.task_num*self.domain_num)] # [domain_num*task_num, batch_size, 1, expert_num]
        

        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            domain_input = self.expert[i](emb)
            out.append(domain_input)       
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)  
        task_exp_out = []
        for i in range(self.task_num):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)       
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)], dim = 1) # batch_size, domain_num, 1
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)], dim = 1) # batch_size, task_num, 1
#分别计算共享专家、领域专家、任务专家的输出，然后完成拼接，这里似乎是说加了一个维度然后在二维上拼接成一个张量
        weighted_domain_fea = []
        for i in range(self.domain_num):
            temp_ = self.bal_d * domain_fea[:, i, :]
            for j in range(self.domain_num):
                if j != i:
                    temp_ += (1-self.bal_d)/(self.domain_num-1) * domain_fea[:, j, :]
            weighted_domain_fea.append(temp_)
        weighted_task_fea = []
        for i in range(self.task_num):
            temp_ = self.bal_t * task_fea[:, i, :]
            for j in range(self.task_num):
                if j != i:
                    temp_ += (1-self.bal_t)/(self.task_num-1) * task_fea[:, j, :]
            weighted_task_fea.append(temp_)
        #然后是计算领域、任务的平衡特征。对每个领域或任务，将其自身特征用对应权重加权，其他领域或任务的特征用剩余权重平均加权，生成平衡后的特征列表。
        fused_fea = [torch.bmm(gate_value[i], fea).squeeze(1) + 
                     self.exp_d * weighted_domain_fea[i%self.domain_num] +
                     self.exp_t * weighted_task_fea[i//self.domain_num]
                     for i in range(self.task_num*self.domain_num)]
        # 接着是计算融合特征，先是用门控与共享专家结合，然后对任务、领域进行加权
        results = [torch.sigmoid(self.tower[i](fused_fea[i]).squeeze(1)) for i in range(self.task_num*self.domain_num)]
#预测塔进行激活预测
        # save output in result [bs, task_num], all domains are integrated with bs length
        output = torch.zeros((len(results[0]), len(results))).to(_device)
        for i in range(len(results)):
            output[:, i] = results[i]

        result = torch.zeros((len(results[0]), self.task_num)).to(_device)
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(mask[d], output[:, t*self.domain_num+d], result[:, t])

        if self.training:
            return result, complexity, specificity
        else:
            return result
#在输出上用零张量存储所有结果，mask指代领域，按任务返回

# PLE
class PLE(nn.Module):
    def __init__(self, features, task_num, domain_num, fcn_dims, expert_num):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.embedding = EmbeddingLayer(features)

        self.task_num = task_num
        self.domain_num = domain_num
        self.total_tasks = task_num * domain_num  # 在你的框架中，每个domain*task被视为一个子任务

        # 简化版 PLE：单层 Extraction Layer
        # 1. 任务独有专家 (Task-specific experts)
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([MLP_N([self.input_dim] + fcn_dims) for _ in range(expert_num)])
            for _ in range(self.total_tasks)
        ])

        # 2. 共享专家 (Shared experts)
        self.shared_experts = nn.ModuleList([
            MLP_N([self.input_dim] + fcn_dims) for _ in range(expert_num)
        ])

        # 3. 门控机制 (Gates)
        # 为每个任务定义一个 Gate，它能看到该任务的独有专家和共享专家
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_dim, expert_num * 2), nn.Softmax(dim=1))
            for _ in range(self.total_tasks)
        ])

        # 4. 预测塔 (Towers)
        self.tower = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fcn_dims[-1], fcn_dims[-1]),
                nn.LayerNorm(fcn_dims[-1]),
                nn.ReLU(),
                nn.Linear(fcn_dims[-1], 1)
            ) for _ in range(self.total_tasks)
        ])

    def forward(self, x, test_flag=False, aug_mask=None, noise_std=0.1):
        _device = x.device
        input_emb = self.embedding(x, self.features, squeeze_dim=True)

        # === 优化 1: 预先计算共享专家（全 Batch 只算一次！） ===
        # 这样不管有多少个任务，共享部分都不再重复计算
        shared_outs = [exp(input_emb).unsqueeze(1) for exp in self.shared_experts]

        task_outputs = []
        for i in range(self.total_tasks):
            # 1. 计算当前任务独有专家
            spec_outs = [exp(input_emb).unsqueeze(1) for exp in self.specific_experts[i]]

            # 2. 拼接（直接使用预写好的 shared_outs）
            all_experts_out = torch.cat(spec_outs + shared_outs, dim=1)

            # 3. 门控与预测塔
            gate_score = self.gates[i](input_emb).unsqueeze(1)
            fused_fea = torch.bmm(gate_score, all_experts_out).squeeze(1)
            res = torch.sigmoid(self.tower[i](fused_fea).squeeze(1))
            task_outputs.append(res)

        # === 以下逻辑保持不变 ===
        domain_id = x[-1, :].long()
        result = torch.zeros((input_emb.shape[0], self.task_num)).to(_device)
        for t in range(self.task_num):
            for d in range(self.domain_num):
                mask = (domain_id == d)
                idx = t * self.domain_num + d
                result[:, t] = torch.where(mask, task_outputs[idx], result[:, t])

        if self.training:
            dummy_dbo = torch.zeros(input_emb.shape[0]).to(_device)
            return result, dummy_dbo, dummy_dbo
        else:
            return result