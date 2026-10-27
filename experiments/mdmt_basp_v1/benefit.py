"""Same-state action measurements and detached, versioned training teachers."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .model import BASPModel
from .relation import weighted_low_rank


def logit_bce(y, z):
    return F.softplus(z.double()) - y.double()*z.double()


def exact_benefit(y, reference, correction):
    return logit_bce(y, reference) - logit_bce(y, reference+correction)


@torch.no_grad()
def measure_benefits(model, packet, domain, y):
    """Pure measurement: no relation fitting, scale refresh or model mutation."""
    if not isinstance(model, BASPModel):
        raise TypeError('A source-free model has no reading action library')
    output = model.measure_output(packet, domain)
    maxa = max(len(actions) for actions in model.action_maps)
    gain = y.new_zeros(len(y), model.T, maxa)
    for d, part in output.details.items():
        rows = part['rows']
        b = exact_benefit(y[rows, :, None], part['reference'][..., None], part['c'])
        gain[rows, :, :b.shape[-1]] = b.to(gain.dtype)
    return dict(benefits=gain.detach(), output=output,
                prediction=output.gain_prediction, q=output.need_mass,
                actions=copy.deepcopy(model.action_maps))


def _group_eligibility(teacher, domain, normalized_gain, prediction, mass, panels):
    result = {}
    if not teacher.flags['CONTRASTIVE']:
        return result
    cfg = teacher.cfg['contrast']
    for d in range(teacher.D):
        row = domain == d
        actions = len(teacher.action_maps[d])
        local_gain = normalized_gain[row, :, :actions]
        q = mass[row, :len(teacher.needs[d].leaves)]
        hard = q.argmax(-1)
        means = []
        supports = []
        for panel in (0, 1):
            pm = panels[row] == panel
            panelmeans, ns = [], []
            for k in range(q.shape[1]):
                selected = pm & (hard == k)
                ns.append(int(selected.sum()))
                panelmeans.append(local_gain[selected].mean(0) if selected.any()
                                  else normalized_gain.new_zeros(teacher.T, actions))
            means.append(panelmeans)
            supports.append(ns)
        qualified = []
        weights = teacher.action_weight(d)
        for k in range(q.shape[1]):
            for j in range(k+1, q.shape[1]):
                if any(min(supports[p][k], supports[p][j]) < cfg['half_panel_min_support'] for p in (0, 1)):
                    continue
                distances = [float(((means[p][k]-means[p][j]).square()*weights).sum().div(teacher.T).sqrt())
                             for p in (0, 1)]
                if min(distances) < cfg['negative_group_distance_min']:
                    continue
                different = True
                for p in (0, 1):
                    task_difference = []
                    for t in range(teacher.T):
                        first = torch.cat((means[p][k][t], means[p][k].new_zeros(1))).argmax()
                        second = torch.cat((means[p][j][t], means[p][j].new_zeros(1))).argmax()
                        task_difference.append(first != second)
                    different = different and any(task_difference)
                if cfg['requires_different_best_actions'] and not different:
                    continue
                margin = min(cfg['negative_margin_fraction']*min(distances), cfg['negative_margin_max'])
                qualified.append((k, j, margin))
        result[d] = qualified
    return result


@dataclass
class TeacherCache:
    teacher: BASPModel
    packet: dict
    domain: torch.Tensor
    y: torch.Tensor
    benefits: torch.Tensor
    need_mass: torch.Tensor
    prediction: torch.Tensor | None
    row_ids: np.ndarray
    panels: torch.Tensor
    version: object
    pairs: dict

    def to(self, device):
        self.teacher.to(device)
        self.packet = {key: value.to(device) for key, value in self.packet.items()}
        for key in ('domain', 'y', 'benefits', 'need_mass', 'prediction', 'panels'):
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, value.to(device))
        return self

    def subset(self, indices):
        index = torch.as_tensor(indices, device=self.domain.device, dtype=torch.long)
        return dict(packet={key: value[index] for key, value in self.packet.items()},
                    domain=self.domain[index], y=self.y[index],
                    target=self.benefits[index]/self.teacher.benefit_scale,
                    teacher_prediction=None if self.prediction is None else self.prediction[index],
                    teacher_q=self.need_mass[index], pairs=self.pairs,
                    panels=self.panels[index], row_ids=self.row_ids[np.asarray(indices)],
                    version=self.version)


@torch.no_grad()
def align_cache(cache, model):
    """Only duplicate inherited action coordinates during a frozen transaction."""
    if cache is None:
        return None
    oldmaps = getattr(cache, 'action_maps', cache.teacher.action_maps)
    maxa = max(len(actions) for actions in model.action_maps)
    gain = cache.benefits.new_zeros(len(cache.domain), model.T, maxa)
    prediction = torch.zeros_like(gain) if cache.prediction is not None else None
    for d in range(model.D):
        rows = cache.domain == d
        for a, (source, leaf) in enumerate(model.action_maps[d]):
            ancestor = [(len(oldleaf), i) for i, (s, oldleaf) in enumerate(oldmaps[d])
                        if s == source and (leaf == oldleaf or leaf.startswith(oldleaf))]
            if not ancestor:
                raise ValueError('Missing inherited action in frozen teacher')
            old = max(ancestor)[1]
            gain[rows, :, a] = cache.benefits[rows, :, old]
            if prediction is not None:
                prediction[rows, :, a] = cache.prediction[rows, :, old]
    cache.benefits, cache.prediction = gain, prediction
    cache.action_maps = copy.deepcopy(model.action_maps)
    return cache


@torch.no_grad()
def refresh_teacher(model, probe_packet, domain, y, row_ids, panels=None,
                    version=None, freeze_scale=False):
    if not isinstance(model, BASPModel):
        return None
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    measured = measure_benefits(teacher, probe_packet, domain, y)
    gain, q = measured['benefits'], measured['q']
    if freeze_scale or not model.scale_frozen:
        scale = gain.double().square().mean().sqrt().clamp_min(model.cfg['benefit']['scale_floor'])
        model.benefit_scale.copy_(scale.to(model.benefit_scale))
        model.scale_frozen = True
        teacher.benefit_scale.copy_(model.benefit_scale)
        teacher.scale_frozen = True
    rows_np = np.asarray(row_ids, dtype=np.int64)
    for d in range(model.D):
        rows = domain == d
        na, nk = len(model.action_maps[d]), len(model.needs[d].leaves)
        model.tables[d].measure(gain[rows, :, :na], q[rows, :nk], rows_np[rows.cpu().numpy()],
                                minimum_ess=model.cfg['benefit']['minimum_observed_ess'],
                                low_rank=False, version=version)
    if model.flags['LOW_RANK']:
        # One directed matrix per task, including all domain pairs and masks.
        need_offsets = np.cumsum([0]+[len(tree.leaves) for tree in model.needs])
        source_offsets = np.cumsum([0]+[len(tree.leaves) for tree in model.sources])
        for task in range(model.T):
            matrix = gain.new_zeros(int(need_offsets[-1]), int(source_offsets[-1]))
            weights = torch.zeros_like(matrix)
            for d in range(model.D):
                lo, hi = int(need_offsets[d]), int(need_offsets[d+1])
                for a, (s, leaf) in enumerate(model.action_maps[d]):
                    j = int(source_offsets[s]) + model.sources[s].leaves.index(leaf)
                    matrix[lo:hi, j] = model.tables[d].observed_gain[task, :, a]
                    weights[lo:hi, j] = model.tables[d].ess * model.tables[d].observed[task, :, a]
            lowrank = weighted_low_rank(matrix, weights, model.cfg['low_rank_relation']['rank'],
                                        model.cfg['low_rank_relation']['als_iterations'],
                                        model.cfg['low_rank_relation']['ridge'])
            for d in range(model.D):
                lo, hi = int(need_offsets[d]), int(need_offsets[d+1])
                for a, (s, leaf) in enumerate(model.action_maps[d]):
                    j = int(source_offsets[s]) + model.sources[s].leaves.index(leaf)
                    valid = model.tables[d].observed[task, :, a]
                    model.tables[d].policy_gain[task, :, a] = torch.where(valid, lowrank[lo:hi, j],
                                                                        model.tables[d].policy_gain[task, :, a])
    panel = torch.as_tensor(np.arange(len(domain)) % 2 if panels is None else panels,
                            device=domain.device, dtype=torch.long)
    pairs = _group_eligibility(teacher, domain, gain/teacher.benefit_scale,
                               measured['prediction'], q, panel)
    packet = {key: value.detach().clone() for key, value in probe_packet.items()}
    return TeacherCache(teacher, packet, domain.detach().clone(), y.detach().clone(),
                        gain, q.detach(), None if measured['prediction'] is None else measured['prediction'].detach(),
                        rows_np.copy(), panel, version, pairs)
