"""P7 backbone library.

Sources:
- MMoE_Layer / CGC_Layer (PLE): copied verbatim from FuxiCTR v2.3.9
  (model_zoo/multitask/{MMoE,PLE}/src, Apache-2.0), with the FuxiCTR BaseModel
  plumbing replaced by our own embedding/tower wrappers so they consume the
  same input tensors as MDMTRec. The layer math is unchanged.
- ShareBottom_Layer: the FuxiCTR v2.3.9 original is a *single* MLP_Block
  shared bottom feeding num_tasks tower heads (no num_experts, no averaging).
  Our `num_experts`-parallel `MLPBlock` + mean-over-experts aggregation is a
  *repo-local adaptation* added in the P7 fix to plug a `Backbone` wrapper
  that hands `num_tasks = D*T` into the layer; it is NOT a verbatim copy of
  the FuxiCTR layer math, and the parameter count of the bottom is roughly
  `num_experts`× the original single MLP. Semantics (single shared bottom
  representation broadcast to all task towers) match; layer math does not.
- STAR: our implementation following L. Sheng et al., "One Model to Serve
  All: Star Topology Adaptive Recommender", RecSys 2021 (per-domain weight
  W_p element-wise multiplied with shared weight W_0, plus per-domain bias).
  Cross-checked structurally against M3oE's own star fusion (model.py:151-153)
  which is a single-layer instance of the same topology.
"""
import torch
from torch import nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3oe"))
from MDMTRec.basic.layers import EmbeddingLayer  # noqa: E402


class MLPBlock(nn.Module):
    """Minimal MLP: [Linear+LayerNorm+ReLU]*n (+ optional output Linear)."""

    def __init__(self, input_dim, hidden_units, output_dim=None):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_units:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        if output_dim is not None:
            layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------- FuxiCTR v2.3.9 MMoE_Layer (verbatim math) ----------------
class MMoE_Layer(nn.Module):
    """From FuxiCTR v2.3.9 model_zoo/multitask/MMoE/src/MMoE.py (Apache-2.0)."""

    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units,
                 gate_hidden_units):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLPBlock(input_dim, expert_hidden_units)
                                      for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([MLPBlock(input_dim, gate_hidden_units, output_dim=num_experts)
                                   for _ in range(self.num_tasks)])
        self.gate_activation = nn.Softmax(dim=1)

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate_activation(self.gate[i](x))
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1),
                                                        experts_output), dim=1))
        return mmoe_output


# ---------------- FuxiCTR v2.3.9 CGC_Layer (PLE, verbatim math) ----------------
class CGC_Layer(nn.Module):
    """From FuxiCTR v2.3.9 model_zoo/multitask/PLE/src/PLE.py (Apache-2.0).
    x must be a list of length num_tasks+1; x[-1] feeds the shared experts."""

    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim,
                 expert_hidden_units, gate_hidden_units):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.shared_experts = nn.ModuleList([MLPBlock(input_dim, expert_hidden_units)
                                             for _ in range(self.num_shared_experts)])
        self.specific_experts = nn.ModuleList(
            [nn.ModuleList([MLPBlock(input_dim, expert_hidden_units)
                            for _ in range(self.num_specific_experts)])
             for _ in range(num_tasks)])
        self.gate = nn.ModuleList(
            [MLPBlock(input_dim, gate_hidden_units,
                      output_dim=num_specific_experts + num_shared_experts if i < num_tasks
                      else num_shared_experts)
             for i in range(self.num_tasks + 1)])
        self.gate_activation = nn.Softmax(dim=1)

    def forward(self, x):
        specific_expert_outputs = [[self.specific_experts[i][j](x[i])
                                    for j in range(self.num_specific_experts)]
                                   for i in range(self.num_tasks)]
        shared_expert_outputs = [self.shared_experts[i](x[-1])
                                 for i in range(self.num_shared_experts)]
        cgc_outputs = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                gate_input = torch.stack(specific_expert_outputs[i] + shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
            else:
                gate_input = torch.stack(shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[-1]))
            cgc_outputs.append(torch.sum(gate.unsqueeze(-1) * gate_input, dim=1))
        return cgc_outputs


# ---------------- ShareBottom (repo-local adaptation; not verbatim FuxiCTR) ----------------
# The FuxiCTR v2.3.9 original is a *single* MLP_Block shared bottom feeding
# num_tasks tower heads; it has no num_experts parameter and no averaging
# across experts. To plug the original into our `Backbone` wrapper, which
# hands `num_tasks = D*T` (and historically tried to slice expert_output
# along the task axis, causing IndexError), this layer keeps the
# Backbone-expected num_experts / num_tasks signature by stacking num_experts
# parallel MLPBlocks along dim=1 and averaging them into a single shared
# bottom representation, then broadcasting `[shared] * num_tasks` to all
# tower inputs. Semantics (one shared bottom -> all task heads) match the
# FuxiCTR original; layer math (parameter count = num_experts * single MLP)
# does not. See P7_generalization.md §D.2 and the file header for the
# factual distinction used in the report.
class ShareBottom_Layer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units):
        super().__init__()
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLPBlock(input_dim, expert_hidden_units)
                                      for _ in range(num_experts)])
        self.tower_input_dim = expert_hidden_units[-1]

    def forward(self, x):
        # Mean-over-experts aggregation, then broadcast to num_tasks towers.
        expert_output = torch.stack([e(x) for e in self.experts], dim=1)
        shared = expert_output.mean(dim=1)  # (batch, tower_input_dim)
        return [shared] * self.num_tasks


# ---------------- STAR (our implementation, RecSys 2021 paper) ----------------
class STAR_Layer(nn.Module):
    """One STAR topology FCN layer: y = (W_d (elem.) W_0) x + b_d + b_0."""

    def __init__(self, input_dim, output_dim, domain_num):
        super().__init__()
        self.shared_weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.shared_bias = nn.Parameter(torch.zeros(output_dim))
        self.domain_weight = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, output_dim)) for _ in range(domain_num)])
        self.domain_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(output_dim)) for _ in range(domain_num)])
        for w in list(self.domain_weight) + [self.shared_weight]:
            nn.init.xavier_uniform_(w.data)
        self.domain_num = domain_num

    def forward(self, x, domain_id):
        out = None
        for d in range(self.domain_num):
            mask = (domain_id == d).unsqueeze(1)
            y = torch.matmul(x, torch.multiply(self.domain_weight[d], self.shared_weight)) \
                + self.domain_bias[d] + self.shared_bias
            out = y if out is None else torch.where(mask, y, out)
        return out


class Backbone(nn.Module):
    """Common wrapper: MDMTRec-style embedding -> backbone core -> D*T towers,
    output assembled per-sample domain exactly like MDMTRec.forward, so the
    as-published trainer (masked 6-term BCE, per-(d,t) eval) works unchanged."""

    def __init__(self, features, domain_num, task_num, backbone="mmoe",
                 expert_num=4, fcn_dims=(512, 256, 64)):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.domain_num = domain_num
        self.task_num = task_num
        self.backbone_name = backbone
        n_task = domain_num * task_num
        expert_hidden = [fcn_dims[1], fcn_dims[2]]  # [256, 64] like M3oE experts
        if backbone == "mmoe":
            self.embedding = EmbeddingLayer(features)
            self.core = MMoE_Layer(num_experts=expert_num, num_tasks=n_task,
                                   input_dim=self.input_dim,
                                   expert_hidden_units=expert_hidden,
                                   gate_hidden_units=[fcn_dims[2]])
            tower_in = fcn_dims[2]
        elif backbone == "ple":
            self.embedding = EmbeddingLayer(features)
            self.core = CGC_Layer(num_shared_experts=expert_num, num_specific_experts=1,
                                  num_tasks=n_task, input_dim=self.input_dim,
                                  expert_hidden_units=expert_hidden,
                                  gate_hidden_units=[fcn_dims[2]])
            tower_in = fcn_dims[2]
        elif backbone == "sharebottom":
            self.embedding = EmbeddingLayer(features)
            self.core = ShareBottom_Layer(num_experts=expert_num, num_tasks=n_task,
                                          input_dim=self.input_dim,
                                          expert_hidden_units=expert_hidden)
            tower_in = fcn_dims[2]
        elif backbone == "star":
            self.embedding = EmbeddingLayer(features)
            self.star1 = STAR_Layer(self.input_dim, fcn_dims[0], domain_num)
            self.star2 = STAR_Layer(fcn_dims[0], fcn_dims[1], domain_num)
            self.relu = nn.ReLU()
            self.proj = MLPBlock(fcn_dims[1], [fcn_dims[2]])
            tower_in = fcn_dims[2]
        else:
            raise ValueError(backbone)
        self.tower = nn.ModuleList([
            nn.Sequential(nn.Linear(tower_in, tower_in), nn.LayerNorm(tower_in),
                          nn.ReLU(), nn.Linear(tower_in, 1))
            for _ in range(n_task)])
        # placeholder arch-param slot so the as-published trainer's darts
        # optimizer has a (never-used-in-forward) parameter; excluded from the
        # main optimizer by the fix_optim_grouping split and from momentum by
        # the shared-name filter
        self._weight_dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, test_flag=False):
        _device = x.device
        domain_id = x[-1, :].clone().detach()
        cpu_masks = [(domain_id == d).cpu() for d in range(self.domain_num)]
        input_emb = self.embedding(x, self.features, squeeze_dim=True)
        n_task = self.task_num * self.domain_num
        if self.backbone_name in ("mmoe", "sharebottom"):
            outs = self.core(input_emb)
        elif self.backbone_name == "ple":
            # every (domain,task) sees the same input; x[-1] duplicates it for
            # the shared experts (as in the standard multi-task setting)
            outs = self.core([input_emb] * n_task + [input_emb])
        elif self.backbone_name == "star":
            h = self.relu(self.star1(input_emb, domain_id))
            outs = [self.proj(self.relu(self.star2(h, domain_id))) for _ in range(n_task)]
        results = [torch.sigmoid(self.tower[i](outs[i]).squeeze(1)) for i in range(n_task)]
        output = torch.zeros((len(results[0]), n_task))
        for i in range(n_task):
            output[:, i] = results[i]
        result = torch.zeros((len(results[0]), self.task_num))
        for t in range(self.task_num):
            for d in range(self.domain_num):
                result[:, t] = torch.where(cpu_masks[d],
                                           output[:, t * self.domain_num + d], result[:, t])
        return result

    def shared_param_names(self):
        """Parameters shared across domains/tasks (component-1 momentum target).
        Private: per-(d,t) towers and gates, PLE task-specific experts,
        STAR per-domain weights/biases."""
        excl = ("tower.",)
        if self.backbone_name == "ple":
            excl += ("core.specific_experts.",)
        if self.backbone_name == "star":
            excl += ("star1.domain_weight.", "star1.domain_bias.",
                     "star2.domain_weight.", "star2.domain_bias.")
        names = []
        for n, _ in self.named_parameters():
            if n.startswith(excl):
                continue
            if n.startswith("embedding."):
                names.append(n)
            elif n.startswith("core."):
                # FuxiCTR naming: gate.i / experts.i / shared_experts.i /
                # specific_experts.i.j — keep gates private (per task)
                if ".gate." in n:
                    continue
                if "specific_experts" in n:
                    continue
                names.append(n)
            elif n.startswith("star1.shared_") or n.startswith("star2.shared_"):
                names.append(n)
            elif n.startswith("proj."):
                names.append(n)
        return names
