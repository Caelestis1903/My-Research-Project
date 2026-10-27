"""Label-free BASP-A prediction, executable source trees and need trees."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from .relation import GainTable, reading_policy


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def residual_head(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), zero_output(nn.Linear(width, outputs)))


def packet_values(packet, domain=None):
    if not isinstance(packet, dict) or not {'h', 'z', 'phi'}.issubset(packet):
        raise TypeError('Packet requires h, z and phi')
    if set(packet) - {'h', 'z', 'phi', 'domain', 'h0', 'z0'}:
        raise TypeError('Prediction packet may contain only deployed features')
    return packet['h'], packet['z'], packet['phi'], packet.get('domain') if domain is None else domain


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    reference_logits: torch.Tensor
    need_mass: torch.Tensor
    source_values: torch.Tensor | None
    read_mass: torch.Tensor | None
    null_mass: torch.Tensor | None
    gain_prediction: torch.Tensor | None
    details: dict | None = None


class Centers(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.base = nn.Parameter(values.clone())
        self.extra = nn.ParameterList()

    def forward(self):
        return torch.cat([self.base] + list(self.extra), -1)

    def append_coordinate(self, parent):
        self.extra.append(nn.Parameter(self()[..., parent:parent+1].detach().clone()))


class GainHead(nn.Module):
    def __init__(self, width, tasks, actions):
        super().__init__()
        self.tasks, self.actions = tasks, actions
        self.base = zero_output(nn.Linear(width, tasks*actions))
        self.extra = nn.ModuleList()

    def forward(self, r):
        return torch.cat([self.base(r).reshape(len(r), self.tasks, self.actions)] +
                         [head(r).unsqueeze(-1) for head in self.extra], -1)

    def append_coordinate(self, parent):
        head = nn.Linear(self.base.in_features, self.tasks).to(self.base.weight)
        with torch.no_grad():
            if parent < self.actions:
                head.weight.copy_(self.base.weight.reshape(self.tasks, self.actions, -1)[:, parent])
                head.bias.copy_(self.base.bias.reshape(self.tasks, self.actions)[:, parent])
            else:
                old = self.extra[parent-self.actions]
                head.weight.copy_(old.weight)
                head.bias.copy_(old.bias)
        self.extra.append(head)


class NeedTree(nn.Module):
    def __init__(self, width, tasks, actions, leaves, device=None):
        super().__init__()
        self.width, self.tasks = width, tasks
        self.centers = Centers(torch.zeros(leaves, tasks, actions, device=device))
        self.locals = nn.ModuleDict({f'n{i}': zero_output(nn.Linear(width, tasks)) for i in range(leaves)})
        self.split_centers = nn.ModuleDict()
        self.splits = {}
        self.roots = list(self.locals)
        self.leaves = self.roots.copy()
        self.paths = {name: [name] for name in self.roots}
        self.initialized = False
        self.fixed = False
        self.register_buffer('fixed_centers', torch.zeros(leaves, width))

    def all_center_modules(self):
        return [self.centers] + list(self.split_centers.values())

    def route(self, profile, fixed_feature, action_weight, temperature):
        def distance(p, c):
            return ((p[:, None]-c[None]).square()*action_weight[None, None, None]).sum((-1, -2))/self.tasks
        if self.fixed:
            root = torch.softmax(-torch.cdist(fixed_feature, self.fixed_centers).square()/temperature, -1)
        elif self.initialized:
            root = torch.softmax(-distance(profile, self.centers())/temperature, -1)
        else:
            root = profile.new_full((len(profile), len(self.roots)), 1/len(self.roots))
        mass = {name: root[:, i] for i, name in enumerate(self.roots)}
        for parent, (left, right) in self.splits.items():
            if self.fixed:
                split = torch.softmax(-torch.cdist(fixed_feature, self.split_centers[parent]()).square()/temperature, -1)
            else:
                split = torch.softmax(-distance(profile, self.split_centers[parent]())/temperature, -1)
            mass[left], mass[right] = mass[parent]*split[:, 0], mass[parent]*split[:, 1]
        return torch.stack([mass[leaf] for leaf in self.leaves], -1), mass

    def local_values(self, r):
        values = {name: module(r) for name, module in self.locals.items()}
        return torch.stack([sum((values[n] for n in self.paths[leaf]), r.new_zeros(len(r), self.tasks))
                            for leaf in self.leaves], 1)

    def split(self, parent_index, centers=None):
        parent = self.leaves[parent_index]
        left, right = parent+'a', parent+'b'
        values = (self.fixed_centers if self.fixed else self.centers()).detach()[:2].clone() if centers is None else centers.detach().clone()
        if values.shape[0] != 2:
            raise ValueError('A split needs two deployable benefit centers')
        self.split_centers[parent] = Centers(values).to(self.centers.base)
        if self.fixed:
            self.split_centers[parent].requires_grad_(False)
        for child in (left, right):
            self.locals[child] = zero_output(nn.Linear(self.width, self.tasks)).to(self.centers.base)
            self.paths[child] = self.paths[parent] + [child]
        self.splits[parent] = (left, right)
        self.leaves[parent_index] = left
        self.leaves.append(right)
        return left, right


class SourceResidual(nn.Module):
    def __init__(self, features, rank):
        super().__init__()
        self.V = nn.Parameter(torch.randn(rank, features)*.02)
        self.U = nn.Parameter(torch.zeros(features, rank))

    def forward(self, phi):
        return F.linear(F.gelu(F.linear(phi, self.V)), self.U)


class SourceTree(nn.Module):
    def __init__(self, features, rank, leaves, input_dim):
        super().__init__()
        self.features, self.rank, self.input_dim = features, rank, input_dim
        self.blocks = nn.ModuleDict({f's{i}': SourceResidual(features, rank) for i in range(leaves)})
        self.owner_root = zero_output(nn.Linear(features, leaves))
        self.owner_branches = nn.ModuleDict()
        self.roots = list(self.blocks)
        self.leaves = self.roots.copy()
        self.paths = {name: [name] for name in self.roots}
        self.splits = {}
        self.foreign_weights = {}
        self.foreign_biases = {}

    def values(self, phi):
        residuals = {key: block(phi) for key, block in self.blocks.items()}
        return torch.stack([phi+sum((residuals[n] for n in self.paths[leaf]), torch.zeros_like(phi))
                            for leaf in self.leaves], 1)

    def responsibilities(self, phi, features=None, owner=False):
        root = self.owner_root(phi).softmax(-1) if owner else phi.new_full((len(phi), len(self.roots)), 1/len(self.roots))
        mass = {name: root[:, i] for i, name in enumerate(self.roots)}
        for parent, (left, right) in self.splits.items():
            if owner:
                r = self.owner_branches[parent](phi).squeeze(-1).sigmoid()
            else:
                r = F.linear(features, self.foreign_weights[parent], self.foreign_biases[parent]).squeeze(-1).sigmoid()
            mass[left], mass[right] = mass[parent]*r, mass[parent]*(1-r)
        return torch.stack([mass[leaf] for leaf in self.leaves], -1)

    def split(self, parent_index, weight=None, bias=None):
        parent = self.leaves[parent_index]
        left, right = parent+'a', parent+'b'
        ref = self.owner_root.weight
        for child in (left, right):
            self.blocks[child] = SourceResidual(self.features, self.rank).to(ref)
            self.paths[child] = self.paths[parent] + [child]
        self.owner_branches[parent] = zero_output(nn.Linear(self.features, 1)).to(ref)
        weight = ref.new_zeros(1, self.input_dim) if weight is None else weight.to(ref).reshape(1, self.input_dim)
        bias = ref.new_zeros(1) if bias is None else bias.to(ref).reshape(1)
        # Fixed deployed partition; buffers follow device/dtype and RAM snapshots.
        self.register_buffer('foreign_weight_'+parent, weight.clone())
        self.register_buffer('foreign_bias_'+parent, bias.clone())
        self.foreign_weights[parent] = getattr(self, 'foreign_weight_'+parent)
        self.foreign_biases[parent] = getattr(self, 'foreign_bias_'+parent)
        self.splits[parent] = (left, right)
        self.leaves[parent_index] = left
        self.leaves.append(right)
        return left, right

    def _apply(self, fn):
        super()._apply(fn)
        for parent in self.splits:
            self.foreign_weights[parent] = getattr(self, 'foreign_weight_'+parent)
            self.foreign_biases[parent] = getattr(self, 'foreign_bias_'+parent)
        return self


class TargetModel(nn.Module):
    def __init__(self, cfg, arm=None, teacher_w=None, teacher_b=None, seed=2022):
        super().__init__()
        self.cfg, self.arm = copy.deepcopy(cfg), copy.deepcopy(arm or {})
        self.D, self.T = cfg['data']['domains'], cfg['data']['tasks']
        self.H = cfg['model']['native_dim']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.base_heads = nn.ModuleList([residual_head(self.H+self.T, cfg['model']['target_match_width'], 1)
                                            for _ in range(self.D*self.T)])
        self.epoch = 1
        self.seed = int(seed)

    def forward(self, packet, domain=None):
        h, z, phi, domain = packet_values(packet, domain)
        x = torch.cat((h, z), -1)
        out = z.clone()
        for d in range(self.D):
            rows = domain == d
            if rows.any():
                for t in range(self.T):
                    out[rows, t] += self.base_heads[d*self.T+t](x[rows]).squeeze(-1)
        return ForwardOutput(out, out, out.new_ones(len(out), 1), None, None, None, None, {})

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def parameter_categories(self):
        return {name: 'native' if name.startswith('current_native.') else 'target'
                for name, _ in self.named_parameters()}

    @torch.no_grad()
    def project_readers(self):
        pass


class BASPModel(nn.Module):
    def __init__(self, cfg, arm=None, teacher_w=None, teacher_b=None, seed=2022):
        super().__init__()
        self.cfg, self.arm = copy.deepcopy(cfg), copy.deepcopy(arm or {})
        self.flags = copy.deepcopy(cfg['switches'])
        self.flags.update((arm or {}).get('switches', {}))
        self.fallback = (arm or {}).get('relation_fallback', self.flags.get('relation_fallback', 'NONE'))
        self.aux_matched = (arm or {}).get('aux_matched', self.flags.get('aux_matched', False))
        if self.flags['CONTRASTIVE'] and not self.flags['GAIN_CLUSTER'] and not self.aux_matched:
            raise ValueError('CL without gain routing requires aux_matched=true')
        self.D, self.T = cfg['data']['domains'], cfg['data']['tasks']
        self.H, self.R = cfg['model']['native_dim'], cfg['model']['need_hidden']
        self.P, self.J = cfg['prefix']['source_feature_dim'], cfg['model']['source_prototypes_initial']
        self.K = cfg['model']['need_prototypes_initial']
        self.has_gain = self.flags['GAIN_CLUSTER'] or self.aux_matched
        self.foreign = self.flags['GAIN_RELATION'] or self.fallback == 'FIXED'
        self.epoch = 1
        self.seed = int(seed)
        self.version = 0
        self.action_maps = [[(s, f's{j}') for s in range(self.D) if s != d for j in range(self.J)] for d in range(self.D)]
        self.stable_node_ids = {'need': [[f'd{d}/n{k}' for k in range(self.K)] for d in range(self.D)],
                                'source': [[f'd{s}/s{j}' for j in range(self.J)] for s in range(self.D)]}
        actions = (self.D-1)*self.J
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.base_heads = nn.ModuleList([residual_head(self.H+self.T, cfg['model']['base_head_width'], 1)
                                            for _ in range(self.D*self.T)])
            self.trunk = nn.Linear(self.H+self.T, self.R)
            nn.init.xavier_uniform_(self.trunk.weight)
            nn.init.zeros_(self.trunk.bias)
            self.gain_heads = nn.ModuleList([GainHead(self.R, self.T, actions) for _ in range(self.D)]) if self.has_gain else nn.ModuleList()
            self.needs = nn.ModuleList([NeedTree(self.R, self.T, actions, self.K) for _ in range(self.D)])
            if not self.flags['GAIN_CLUSTER']:
                for tree in self.needs:
                    tree.centers.base.requires_grad_(False)
                    tree.fixed = True
            self.sources = nn.ModuleList([SourceTree(self.P, cfg['model']['source_rank'], self.J, self.H+self.T)
                                          for _ in range(self.D)])
            self.readers = nn.ParameterDict({f'd{d}s{s}': nn.Parameter(torch.zeros(self.T, self.P))
                                             for d in range(self.D) for s in range(self.D) if s != d})
            fixed = torch.randn(self.H, self.R) / self.H**.5
        self.register_buffer('fixed_projection', fixed)
        self.tables = nn.ModuleList([GainTable(self.T, self.K, actions) for _ in range(self.D)])
        for d in range(self.D):
            self.register_buffer(f'action_weight_{d}', torch.full((actions,), 1/actions))
        self.register_buffer('benefit_scale', torch.tensor(.001))
        self.scale_frozen = False
        self.register_buffer('teacher_w', torch.zeros(self.D, self.T, self.P) if teacher_w is None else torch.as_tensor(teacher_w).detach().clone())
        self.register_buffer('teacher_b', torch.zeros(self.D, self.T) if teacher_b is None else torch.as_tensor(teacher_b).detach().clone())
        self.register_buffer('constraint_streak', torch.zeros(self.D, dtype=torch.long))
        self.register_buffer('growth_blocked', torch.zeros(self.D, dtype=torch.bool))
        self.extra_readers = nn.ModuleDict()
        self.extra_wide = nn.ModuleDict()
        self.extra_metadata = {}
        self.partition_status = ['UNINITIALIZED'] * self.D

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def action_weight(self, domain):
        return getattr(self, f'action_weight_{domain}')

    def source_values(self, phi, foreign=False):
        values = [tree.values(phi[:, source]) for source, tree in enumerate(self.sources)]
        if foreign and (not self.flags['SOURCE_WRITE'] or self.epoch < 3):
            values = [value.detach() for value in values]
        return values

    def predict_components(self, packet, domain=None, auxiliary=False, instrument=False):
        h, z, phi, domain = packet_values(packet, domain)
        if domain is None or len(domain) != len(h):
            raise ValueError('One domain per input row is required')
        if h.shape != (len(h), self.H) or z.shape != (len(h), self.T) or phi.shape != (len(h), self.D, self.P):
            raise ValueError('Invalid BASP packet axes')
        if auxiliary:
            h, z, phi = h.detach(), z.detach(), phi.detach()
        x = torch.cat((h, z), -1)
        r = F.gelu(self.trunk(x))
        # A fixed reference coordinate is retained when the native is trainable.
        h0 = packet.get('h0', h).detach()
        z0 = packet.get('z0', z).detach()
        fixed = h0 @ self.fixed_projection
        execute_sources = self.foreign or instrument
        foreign_values = self.source_values(phi.detach(), foreign=True) if execute_sources else []
        pieces = {}
        for d in range(self.D):
            rows = domain == d
            rd = r[rows]
            actions = self.action_maps[d]
            mu = self.gain_heads[d](rd) if self.has_gain else rd.new_zeros(len(rd), self.T, len(actions))
            q, node_mass = self.needs[d].route(mu, fixed[rows], self.action_weight(d), self.cfg['model']['center_temperature'])
            reference = z[rows] + torch.cat([self.base_heads[d*self.T+t](x[rows]) for t in range(self.T)], -1)
            reference = reference + (q[..., None]*self.needs[d].local_values(rd)).sum(1)
            for key, module in self.extra_readers.items():
                meta = self.extra_metadata[key]
                if meta['domain'] == d:
                    value = module(rd).squeeze(-1)
                    mass = node_mass.get(meta.get('parent'), torch.ones_like(value))
                    reference[:, meta['task']] += value*mass
            for key, module in self.extra_wide.items():
                if self.extra_metadata[key]['domain'] == d:
                    reference = reference + module(rd)
            c = rd.new_zeros(len(rd), self.T, len(actions))
            prior = rd.new_zeros(len(rd), len(actions))
            features0 = torch.cat((h0[rows], z0[rows]), -1)
            if execute_sources:
                source_mass = {s: self.sources[s].responsibilities(phi[rows, s].detach(), features0, owner=False)
                               for s in range(self.D) if s != d}
                for a, (s, leaf) in enumerate(actions):
                    index = self.sources[s].leaves.index(leaf)
                    c[:, :, a] = F.linear(foreign_values[s][rows, index], self.readers[f'd{d}s{s}'])
                    prior[:, a] = source_mass[s][:, index] / (self.D-1)
            alpha, null = reading_policy(self.tables[d].policy_gain, prior, self.benefit_scale,
                                         uniform=self.epoch < 3 or not self.flags['GAIN_RELATION'])
            if not self.foreign:
                alpha = torch.zeros_like(alpha)
                null = torch.ones_like(null)
            logits = reference + torch.einsum('bk,btka,bta->bt', q, alpha, c) if self.foreign and bool(alpha.any()) else reference
            pieces[d] = dict(rows=rows, r=rd, mu=mu, q=q, node_mass=node_mass, reference=reference,
                             logits=logits, c=c, alpha=alpha, null=null, prior=prior)
        return pieces, foreign_values

    def forward(self, packet, domain=None):
        return self._output(packet, domain, instrument=False)

    def measure_output(self, packet, domain=None):
        return self._output(packet, domain, instrument=True)

    def _output(self, packet, domain=None, instrument=False):
        h, z, phi, domain = packet_values(packet, domain)
        pieces, source = self.predict_components(packet, domain, instrument=instrument)
        maxk = max(len(tree.leaves) for tree in self.needs)
        maxj = max(len(tree.leaves) for tree in self.sources)
        maxa = max(len(actions) for actions in self.action_maps)
        logits, reference = z.new_zeros(len(z), self.T), z.new_zeros(len(z), self.T)
        q = z.new_zeros(len(z), maxk)
        gain = z.new_zeros(len(z), self.T, maxa) if self.has_gain else None
        reads = z.new_zeros(len(z), self.T, maxk, self.D, maxj)
        null = z.new_ones(len(z), self.T, maxk)
        values = z.new_zeros(len(z), self.D, maxj, self.P)
        for s, value in enumerate(source):
            values[:, s, :value.shape[1]] = value
        for d, part in pieces.items():
            rows, kn = part['rows'], part['q'].shape[1]
            logits[rows], reference[rows] = part['logits'], part['reference']
            q[rows, :kn] = part['q']
            null[rows, :, :kn] = part['null']
            if gain is not None:
                gain[rows, :, :part['mu'].shape[-1]] = part['mu']
            for a, (s, leaf) in enumerate(self.action_maps[d]):
                j = self.sources[s].leaves.index(leaf)
                reads[rows, :, :kn, s, j] = part['alpha'][..., a]
        return ForwardOutput(logits, reference, q, values if source else None,
                             reads if self.foreign else None, null, gain, pieces)

    def owner_logits(self, packet, source, return_values=False):
        h, z, phi, domain = packet_values(packet)
        p = phi[:, source].detach()
        values = self.sources[source].values(p)
        u = self.sources[source].responsibilities(p, owner=True)
        mixed = p+(u[..., None]*(values-p[:, None])).sum(1)
        logits = z.detach() + F.linear(mixed, self.teacher_w[source], self.teacher_b[source])
        return (logits, u, values) if return_values else logits

    @torch.no_grad()
    def initialize_partition(self, packet, domain=None, seed=2022):
        h, z, phi, domain = packet_values(packet, domain)
        r = F.gelu(self.trunk(torch.cat((h, z), -1)))
        for d in range(self.D):
            rows = domain == d
            if not rows.any():
                continue
            tree = self.needs[d]
            profile = self.gain_heads[d](r[rows]) if self.flags['GAIN_CLUSTER'] else (h[rows] @ self.fixed_projection)
            flat = profile.reshape(len(profile), -1).cpu().numpy()
            if len(flat) < self.K or np.max(np.ptp(flat, axis=0), initial=0) <= 1e-8:
                self.partition_status[d] = 'NO_PREDICTABLE_PARTITION'
                if tree.fixed:
                    tree.fixed_centers.zero_()
                continue
            scale = np.tile(np.sqrt(self.action_weight(d).cpu().numpy()/self.T), self.T) if self.flags['GAIN_CLUSTER'] else np.ones(flat.shape[1])
            kmeans = KMeans(n_clusters=self.K, init='k-means++', n_init=8, max_iter=100, tol=1e-4, random_state=int(seed)).fit(flat*scale)
            centers = kmeans.cluster_centers_/scale
            if np.max(np.linalg.norm(centers[:, None]-centers[None], axis=-1)) <= 1e-8:
                self.partition_status[d] = 'NO_PREDICTABLE_PARTITION'
                continue
            if self.flags['GAIN_CLUSTER']:
                tree.centers.base.copy_(torch.as_tensor(centers.reshape(self.K, self.T, -1), device=h.device, dtype=h.dtype))
            else:
                tree.fixed_centers.copy_(torch.as_tensor(centers, device=h.device, dtype=h.dtype))
            tree.initialized = True
            self.partition_status[d] = 'INITIALIZED_PREDICTED_GAIN' if self.flags['GAIN_CLUSTER'] else 'INITIALIZED_UNLABELED_FEATURES'
        return self.partition_status

    @torch.no_grad()
    def project_readers(self):
        radius = self.cfg['model']['w_l2_radius']
        for reader in self.readers.values():
            reader.mul_((radius/reader.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp_max(1))

    def parameter_categories(self):
        result = {}
        for name, _ in self.named_parameters():
            if name.startswith('current_native.'):
                category = 'native'
            elif name.startswith('base_heads.') or name.startswith(('extra_readers.', 'extra_wide.')) or '.locals.' in name:
                category = 'target'
            elif name.startswith('trunk.'):
                category = 'trunk'
            elif name.startswith('gain_heads.'):
                category = 'gain'
            elif name.startswith('needs.'):
                category = 'center'
            elif name.startswith('readers.'):
                category = 'reader'
            elif '.owner_' in name:
                category = 'owner'
            elif name.startswith('sources.'):
                category = 'source'
            else:
                raise ValueError('Unclassified parameter: '+name)
            result[name] = category
        return result

    def structure_metadata(self):
        return dict(need_leaves=[tree.leaves.copy() for tree in self.needs],
                    source_leaves=[tree.leaves.copy() for tree in self.sources],
                    action_maps=copy.deepcopy(self.action_maps), version=self.version)

    def add_structure(self, variant, proposal):
        # Candidate initialization has its own stream and cannot consume the main stream.
        with torch.random.fork_rng(devices=[]):
            event_seed = int(np.random.SeedSequence([self.seed, 97, self.version,
                              int(proposal['domain']), int(proposal['parent'])]).generate_state(1)[0])
            torch.manual_seed(event_seed)
            return self._add_structure(variant, proposal)

    def _add_structure(self, variant, proposal):
        """Mutate only a fork. Existing Parameter objects and Adam state survive."""
        before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        kind, d, parent = proposal['kind'], int(proposal['domain']), int(proposal['parent'])
        if variant == 'KEEP':
            return dict(added_parameters=0, added_macs=0, capacity_match=True)
        if variant == 'SPLIT':
            if kind == 'NEED_SPLIT':
                self.needs[d].split(parent, proposal.get('centers'))
                self.tables[d].split_need(parent)
            elif kind == 'SOURCE_SPLIT':
                if not self.foreign:
                    raise ValueError('SOURCE_SPLIT requires foreign prediction')
                old = self.sources[d].leaves[parent]
                left, right = self.sources[d].split(parent, proposal.get('weight'), proposal.get('bias'))
                for target in range(self.D):
                    if target == d:
                        continue
                    a = self.action_maps[target].index((d, old))
                    self.action_maps[target][a] = (d, left)
                    self.action_maps[target].append((d, right))
                    if self.has_gain:
                        self.gain_heads[target].append_coordinate(a)
                    if not self.needs[target].fixed:
                        for centers in self.needs[target].all_center_modules():
                            centers.append_coordinate(a)
                    w = self.action_weight(target)
                    new = torch.cat((w, w[a:a+1]/2))
                    new[a] /= 2
                    setattr(self, f'action_weight_{target}', new)
                    self.tables[target].split_source(a)
            else:
                raise ValueError('Unknown structure event')
        elif variant == 'READER':
            targets = [d] if kind == 'NEED_SPLIT' else [t for t in range(self.D) if t != d]
            for target in targets:
                for task in range(self.T):
                    key = f'v{self.version}d{target}t{task}r{len(self.extra_readers)}'
                    self.extra_readers[key] = residual_head(self.R, 2, 1).to(self.trunk.weight)
                    self.extra_metadata[key] = dict(domain=target, task=task,
                                                    parent=self.needs[d].leaves[parent] if kind == 'NEED_SPLIT' else None)
        elif variant == 'WIDE':
            split_copy = copy.deepcopy(self)
            expected = split_copy.add_structure('SPLIT', proposal)
            wanted = expected['added_parameters']
            if kind == 'NEED_SPLIT':
                candidates = [(abs(((self.R+1)*width+(width+1)*self.T)-wanted), width)
                              for width in range(1, 257)]
                width = min(candidates)[1]
                key = f'v{self.version}d{d}w{len(self.extra_wide)}'
                self.extra_wide[key] = residual_head(self.R, width, self.T).to(self.trunk.weight)
                self.extra_metadata[key] = dict(domain=d)
            else:
                rank = min(range(4, 17), key=lambda rank: abs(2*self.P*rank-wanted))
                tree = self.sources[d]
                leaf = tree.leaves[parent]
                key = f'{leaf}wide{len(tree.blocks)}'
                tree.blocks[key] = SourceResidual(self.P, rank).to(self.trunk.weight)
                tree.paths[leaf] = tree.paths[leaf]+[key]
        else:
            raise ValueError('Unknown transaction variant')
        self.version += 1
        after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # MACs use actual multiplications in newly executing dense maps.
        added_macs = 0
        if variant == 'SPLIT':
            if kind == 'NEED_SPLIT':
                center_dims = self.R if self.needs[d].fixed else self.T*len(self.action_maps[d])
                added_macs = 2*self.R*self.T + 4*center_dims
            else:
                added_macs = 4*self.P*self.cfg['model']['source_rank'] + self.P + self.H+self.T
                if self.has_gain:
                    added_macs += (self.D-1)*self.R*self.T
                added_macs += sum(2*self.T*sum(center.base.shape[0] for center in self.needs[target].all_center_modules())
                                  for target in range(self.D) if target != d and not self.needs[target].fixed)
        elif variant == 'READER':
            added_macs = len(targets)*self.T*(self.R*2+2)
        elif kind == 'NEED_SPLIT':
            added_macs = self.R*width+width*self.T
        else:
            added_macs = 2*self.P*rank
        result = dict(added_parameters=after-before, added_macs=added_macs, capacity_match=True)
        if variant == 'WIDE':
            result['parameter_mismatch'] = abs(result['added_parameters']-wanted)/max(wanted, 1)
            result['mac_mismatch'] = abs(added_macs-expected['added_macs'])/max(expected['added_macs'], 1)
            result['capacity_match'] = result['parameter_mismatch'] <= .05 and result['mac_mismatch'] <= .20
        return result


def make_model(cfg, arm, teacher_w=None, teacher_b=None, seed=2022):
    cls = TargetModel if arm.get('kind') == 'TARGET_MATCH' else BASPModel
    return cls(cfg, arm, teacher_w, teacher_b, seed)
