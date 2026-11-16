"""Pinned numerical forward with complete input derivatives and constant parameters."""
import torch
from torch import nn
from torch.func import functional_call


def embedding(model, x):
    with torch.no_grad():
        return model.embedding(x, model.features, squeeze_dim=True).detach()


def sparse_fields(model, h):
    return torch.stack([h[:, lo:hi] for lo, hi in model.perturb_slices], dim=1)


def add_delta(model, h, delta):
    result = h.clone()
    for f, (lo, hi) in enumerate(model.perturb_slices):
        result[:, lo:hi] = h[:, lo:hi]+delta[:, f]
    return result


class _Probe(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, x, h, gains):
        m = self.model
        if gains is not None:
            h = add_delta(m, h, sparse_fields(m, h)*gains.unsqueeze(-1))
        masks = [x[-1] == d for d in range(m.domain_num)]
        skip = m.skip_conn(h)
        emb = h.new_zeros((len(h), m.star_dim[1]))
        for d, (weight, bias) in enumerate(zip(m.slot_weight, m.slot_bias)):
            out = h@(weight*m.shared_weight)+bias+m.shared_bias
            emb = torch.where(masks[d].unsqueeze(1), out, emb)
        emb = m.star_mlp(emb)+skip
        gates = [gate(emb).unsqueeze(1) for gate in m.gate]
        fea = torch.stack([expert(emb) for expert in m.expert], dim=1)
        domain = torch.stack([expert(emb) for expert in m.domain_expert], dim=1)
        task = torch.stack([expert(emb) for expert in m.task_expert], dim=1)
        domain_weighted = []
        for d in range(m.domain_num):
            value = m.bal_d*domain[:, d]
            for other in range(m.domain_num):
                if other != d: value = value+(1-m.bal_d)/(m.domain_num-1)*domain[:, other]
            domain_weighted.append(value)
        task_weighted = []
        for t in range(m.task_num):
            value = m.bal_t*task[:, t]
            for other in range(m.task_num):
                if other != t: value = value+(1-m.bal_t)/(m.task_num-1)*task[:, other]
            task_weighted.append(value)
        logits = [m.tower[i](torch.bmm(gates[i], fea).squeeze(1)
            +m.exp_d*domain_weighted[i % m.domain_num]
            +m.exp_t*task_weighted[i // m.domain_num]).squeeze(1)
            for i in range(m.task_num*m.domain_num)]
        result = h.new_zeros((len(h), m.task_num))
        for t in range(m.task_num):
            for d in range(m.domain_num):
                result[:, t] = torch.where(masks[d], logits[t*m.domain_num+d], result[:, t])
        return result


def probe_logits(model, x_target, *, embedding_leaf=None, gains=None):
    h = embedding(model, x_target) if embedding_leaf is None else embedding_leaf
    probe = _Probe(model)
    values = {'model.'+name: value.detach() for name, value in
        list(model.named_parameters())+list(model.named_buffers())}
    return functional_call(probe, values, (x_target, h, gains))


def task_gradients(model, x, y):
    h = embedding(model, x).requires_grad_(True)
    logits = probe_logits(model, x, embedding_leaf=h)
    loss = torch.nn.functional.binary_cross_entropy(logits.sigmoid(), y, reduction='none')
    gradients = [sparse_fields(model, torch.autograd.grad(loss[:, t].sum(), h,
        retain_graph=t == 0, create_graph=False)[0]).detach() for t in range(2)]
    return sparse_fields(model, h).detach(), gradients


def gain_gradients(model, x, y=None):
    a = torch.zeros((x.shape[1], len(model.perturb_slices)), device=x.device,
        dtype=next(model.parameters()).dtype, requires_grad=True)
    logits = probe_logits(model, x, gains=a)
    if y is not None:
        loss = torch.nn.functional.binary_cross_entropy(logits.sigmoid(), y, reduction='none').mean(1)
        return torch.autograd.grad(loss.sum(), a, create_graph=False)[0].detach(), logits.detach()
    jacobian = torch.stack([torch.autograd.grad(logits[:, t].sum(), a,
        retain_graph=t == 0, create_graph=False)[0] for t in range(2)], dim=1).detach()
    return jacobian, logits.detach()
