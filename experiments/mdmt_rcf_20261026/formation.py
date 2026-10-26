"""RCF-v1 finite-budget target-supervised delta formation; RAM states only."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit

from .protocol import FormationResult, FeatureBank, derive_rng, record_access
from .vendor_snapshot.mdmt_dfd_20261026.state import digest, array_hash, state_hash
from .memory_data import persist,release_data_pages


@dataclass
class DeltaFunction:
    module: nn.Module
    scale: float
    source: int
    task: int
    slot: int
    function_id: str
    target: int | None = None
    metadata: dict = field(default_factory=dict)
    training_state: dict = field(default_factory=dict, repr=False)
    outputs: dict = field(default_factory=dict, repr=False)
    anchor_pilots: object = field(default=None, repr=False)

    def values(self, h, chunk=16384, *, unscaled=False):
        device = next(self.module.parameters()).device
        result = []
        with torch.no_grad():
            for i in range(0, len(h), chunk):
                x = torch.as_tensor(np.array(h[i:i+chunk]), dtype=torch.float32, device=device)
                y = self.module(x).reshape(-1).cpu().numpy()
                result.append(y if unscaled else self.scale*y)
        return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


class DeltaMLP(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.hidden = nn.Linear(weight.shape[1], weight.shape[0])
        self.output = nn.Linear(weight.shape[0], 1)
        with torch.no_grad():
            self.hidden.weight.copy_(weight)
            self.hidden.bias.copy_(bias)
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, x):
        return self.output(F.relu(self.hidden(x))).squeeze(-1)


def initialize_delta(bundle, source, task, slot, config, *, local_replica=None,
                     wide=False, random_hidden=False):
    if local_replica is None:
        weight, bias = bundle.pilots.clone_hidden(source, task)
    else:
        weight, bias = bundle.pilots.local_clone_hidden(source, task, local_replica)
    weight, bias = weight.detach().cpu().clone(), bias.detach().cpu().clone()
    if random_hidden:
        rng = derive_rng(bundle.seed, bundle.dataset, 'formation_init', source, task,
                         unit=1000+slot)
        limit = np.sqrt(6.0/(weight.shape[0]+weight.shape[1]))
        weight = torch.from_numpy(rng.uniform(-limit, limit, size=tuple(weight.shape)).astype(np.float32))
        bias.zero_()
    pieces = []
    for unit in (range(2) if wide else (slot,)):
        rng = derive_rng(bundle.seed, bundle.dataset, 'formation_init', source, task, unit=unit)
        std = config['formation']['hidden_noise_std']
        pieces.append((weight+torch.from_numpy(rng.normal(0, std, size=tuple(weight.shape)).astype(np.float32)),
                       bias+torch.from_numpy(rng.normal(0, std, size=tuple(bias.shape)).astype(np.float32))))
    w = torch.cat([p[0] for p in pieces]) if wide else pieces[0][0]
    b = torch.cat([p[1] for p in pieces]) if wide else pieces[0][1]
    # Linear constructors consume RNG; isolate those unused draws from all streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return DeltaMLP(w, b)


class RowCycle:
    """Independent fixed auxiliary permutations, retaining cursor and wrap count."""
    def __init__(self, positions, rng):
        self.positions = np.asarray(positions, np.int64)
        self.rng = rng
        self.order = np.empty(0, np.int64)
        self.cursor = 0
        self.cycles = 0
        self.accessed = 0

    def take(self, n):
        if not len(self.positions):
            return np.empty(0, np.int64)
        pieces, left = [], int(n)
        while left:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.positions)
                self.cursor = 0
                self.cycles += 1
            size = min(left, len(self.order)-self.cursor)
            pieces.append(self.order[self.cursor:self.cursor+size])
            self.cursor += size
            left -= size
        self.accessed += n
        return np.concatenate(pieces)

    def state(self):
        return dict(order=self.order.copy(), cursor=self.cursor, cycles=self.cycles,
                    accessed=self.accessed, rng=copy.deepcopy(self.rng.bit_generator.state))


def bernoulli_kl(base, delta):
    """KL(Ber(sigmoid(base)) || Ber(sigmoid(base+delta))), no probability clip."""
    return F.softplus(base+delta)-F.softplus(base)-torch.sigmoid(base)*delta


def constraint_values(source_base, source_y, source_delta, foreign_deltas,
                      *, local=False, scale=1.0):
    if local:
        s = None
    else:
        a, y = np.asarray(source_base, np.float64), np.asarray(source_y, np.float64)
        d = scale*np.asarray(source_delta, np.float64)
        s = float(np.mean(np.logaddexp(0, a+d)-np.logaddexp(0, a)-y*d))
    r = float(np.mean(np.asarray(foreign_deltas[0], np.float64)**2)) if local else \
        float(0.5*sum(np.mean(np.asarray(v, np.float64)**2) for v in foreign_deltas))
    return s, scale*scale*r


def feasible_scale(source_base, source_y, source_delta, foreign_deltas, config,
                   *, source_constraint=True, local=False):
    fc = config['formation']
    def valid(c):
        s, r = constraint_values(source_base, source_y, source_delta, foreign_deltas,
                                 local=local, scale=c)
        return (not source_constraint or s <= fc['source_ll_budget']) and \
            r <= fc['foreign_delta_mean_square_budget']
    if valid(1.0):
        return 1.0
    low, high = 0.0, 1.0
    if not valid(low):
        raise FloatingPointError('Zero delta must be exactly feasible')
    for _ in range(fc['projection_bisection_steps']):
        middle = (low+high)/2
        if valid(middle):
            low = middle
        else:
            high = middle
    return 0.0 if low < fc['zero_scale_below'] else low


def _query_request(bundle, unit, slot, *, shuffle=False, wide=False):
    view = bundle.views['P_form']
    row_mass = np.zeros(len(view.row_id), np.float64)
    assigned = np.zeros(len(view.row_id), np.float64)
    for i, group in enumerate(unit['groups']):
        pos = group['positions']
        row_mass[pos] = group['pi']/group['n']
        assigned[pos] = float(np.any(unit['assignment'][i] > 0)) if wide else unit['assignment'][i, slot]
    if shuffle:
        # Permute slot membership as a ROW-PAIR vector; same permutation for each slot.
        for target in range(3):
            pos = np.flatnonzero((view.domain == target) & (row_mass > 0))
            rng = derive_rng(bundle.seed, bundle.dataset, 'formation_shuffle',
                             unit['source'], unit['task'], unit=target)
            assigned[pos] = assigned[rng.permutation(pos)]
    mass = row_mass*assigned
    positions = np.flatnonzero(mass > 0)
    z = float(mass.sum())
    omega = len(positions)*mass[positions]/z if z > 0 else np.empty(0, np.float64)
    return positions, omega, dict(Z=z, row_mass_hash=array_hash(row_mass),
                                 assignment_hash=array_hash(assigned))


def _pooled_request(bundle, source, task, config):
    view = bundle.views['P_form']
    gc = config['groups']
    eligible = []
    for target in range(3):
        if target == source:
            continue
        pos = np.flatnonzero(view.domain == target)
        positive = int(view.y[pos, task].sum())
        record_access(bundle, 'POOLED_TARGET_SUPPORT', 'P_form', view.row_id[pos], task,
                      purpose='formation', source=source)
        if len(pos) >= gc['min_rows'] and positive >= gc['min_positive'] and len(pos)-positive >= gc['min_negative']:
            eligible.append(target)
    pos = np.flatnonzero(np.isin(view.domain, eligible))
    if not len(pos):
        return pos, np.empty(0), dict(status='LOW_SUPPORT_POOLED', eligible_targets=[])
    counts = {t: int(np.sum(view.domain == t)) for t in eligible}
    omega = np.asarray([len(pos)/(len(eligible)*counts[int(view.domain[i])]) for i in pos])
    return pos, omega, dict(status='ELIGIBLE', eligible_targets=eligible,
                            nominal_k=2, independent_of_rcf_request=True)


class _TensorStore:
    def __init__(self, bundle, device):
        self.bundle = bundle
        self.device = device
        self.h = {r: torch.as_tensor(np.array(bundle.views[r].h), device=device,
                                    dtype=torch.float32) for r in ('fit', 'P_form')}
        self.y = torch.as_tensor(np.array(bundle.views['P_form'].y), device=device,
                                 dtype=torch.float32)
        self.b = torch.as_tensor(np.array(bundle.base_oof), device=device, dtype=torch.float32)
        self.source_base = {}

    def base(self, role, source, task):
        key = (role, source, task)
        if key not in self.source_base:
            self.source_base[key] = torch.as_tensor(
                np.array(self.bundle.pilots.logits(role, source, task)),
                device=self.device, dtype=torch.float32)
        return self.source_base[key]


def _train_slot(bundle, function, query_pos, omega, config, store,
                *, source_constraint=True, kl=True, local=False):
    fc = config['formation']
    source, task, slot = function.source, function.task, function.slot
    pf, fit = bundle.views['P_form'], bundle.views['fit']
    model = function.module.to(store.device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=fc['lr'], betas=tuple(fc['betas']),
                           eps=fc['eps'], weight_decay=fc['weight_decay'])
    source_fit = np.flatnonzero(fit.domain == source)
    source_form = np.flatnonzero(pf.domain == source)
    foreign = [np.flatnonzero(pf.domain == t) for t in range(3) if t != source]
    r_pools = [source_form] if local else foreign
    pools = [source_fit, source_form]+r_pools
    cycles = [RowCycle(pos, derive_rng(bundle.seed, bundle.dataset, 'formation_aux',
               source, task, unit=100*slot+i)) for i, pos in enumerate(pools)]
    weights = np.zeros(len(pf.row_id), np.float64)
    weights[query_pos] = omega
    omega_tensor = torch.as_tensor(weights, device=store.device, dtype=torch.float32)
    nu = float(fc['dual_init']) if source_constraint else 0.0
    xi = float(fc['dual_init'])
    total_steps, records = 0, []
    started = time.monotonic()
    # Baselines and fixed inputs are detached; only this delta is in opt.
    def full_values():
        d_source = function.values(pf.h[source_form], unscaled=True)
        d_foreign = [function.values(pf.h[pos], unscaled=True) for pos in r_pools]
        a_source = bundle.pilots.logits('P_form', source, task)[source_form]
        source_y = pf.y[source_form, task]
        if not local:
            record_access(bundle, function.metadata['method'], 'P_form', pf.row_id[source_form], task,
                          purpose='formation', function_id=function.function_id, term='S_full_audit')
        s, r = constraint_values(a_source, source_y, d_source, d_foreign, local=local)
        d_query = function.values(pf.h[query_pos], unscaled=True).astype(np.float64)
        bq = np.asarray(bundle.base_oof[query_pos, task], np.float64)
        tq = float(np.mean(omega*(np.logaddexp(0, bq+d_query)-pf.y[query_pos, task]*(bq+d_query))))
        record_access(bundle, function.metadata['method'], 'P_form', pf.row_id[query_pos], task,
                      purpose='formation', function_id=function.function_id, term='T_full_audit')
        return s, r, tq, a_source, source_y, d_source, d_foreign
    for pass_id in range(1, fc['passes']+1):
        order = derive_rng(bundle.seed, bundle.dataset, 'formation_query', source, task,
                           unit=slot, pass_id=pass_id).permutation(query_pos)
        loss_sum, query_count = 0.0, 0
        for start in range(0, len(order), fc['query_batch']):
            idx = order[start:start+fc['query_batch']]
            opt.zero_grad(set_to_none=True)
            pred_delta = model(store.h['P_form'][idx])
            target_loss = (omega_tensor[idx]*F.binary_cross_entropy_with_logits(
                store.b[idx, task]+pred_delta, store.y[idx, task], reduction='none')).mean()
            objective = target_loss
            record_access(bundle, function.metadata['method'], 'P_form', pf.row_id[idx], task,
                          purpose='formation', function_id=function.function_id, term='T', pass_id=pass_id)
            if kl:
                fi = cycles[0].take(fc['kl_source_fit_batch'])
                d = model(store.h['fit'][fi])
                objective = objective+fc['kl_weight']*bernoulli_kl(store.base('fit', source, task)[fi], d).mean()
                record_access(bundle, function.metadata['method'], 'fit', fit.row_id[fi], task,
                              purpose='source_kl', function_id=function.function_id, term='K', pass_id=pass_id,
                              labels=False)
            if source_constraint:
                si = cycles[1].take(fc['source_constraint_form_batch'])
                d = model(store.h['P_form'][si])
                base = store.base('P_form', source, task)[si]
                s_loss = (F.softplus(base+d)-F.softplus(base)-store.y[si, task]*d).mean()
                objective = objective+nu*(s_loss-fc['source_ll_budget'])
                record_access(bundle, function.metadata['method'], 'P_form', pf.row_id[si], task,
                              purpose='formation', function_id=function.function_id, term='S', pass_id=pass_id)
            r_loss = torch.zeros((), device=store.device)
            for j, pool in enumerate(r_pools):
                ri = cycles[2+j].take(512 if local else fc['foreign_rms_batch_per_target'])
                d = model(store.h['P_form'][ri])
                r_loss = r_loss+d.square().mean()/(1 if local else 2)
                record_access(bundle, function.metadata['method'], 'P_form', pf.row_id[ri], task,
                              purpose='formation', function_id=function.function_id, term='R_inputs_only', pass_id=pass_id,
                              labels=False)
            objective = objective+xi*(r_loss-fc['foreign_delta_mean_square_budget'])
            if not torch.isfinite(objective):
                raise FloatingPointError(f'Nonfinite formation loss {function.function_id}')
            objective.backward()
            opt.step()
            total_steps += 1
            loss_sum += float(objective.detach())*len(idx)
            query_count += len(idx)
        s, r, t, *_ = full_values()
        if source_constraint:
            nu = float(np.clip(nu+fc['dual_lr']*(s-fc['source_ll_budget']), *fc['dual_bounds']))
        xi = float(np.clip(xi+fc['dual_lr']*(r-fc['foreign_delta_mean_square_budget']), *fc['dual_bounds']))
        records.append(dict(function_id=function.function_id, method=function.metadata['method'],
            source=source, task=task, slot=slot, target=function.target, pass_id=pass_id,
            query_rows=len(query_pos), query_visits=query_count,
            query_plan_hash=array_hash(order), objective=loss_sum/query_count,
            T=t, S=s, R=r, nu=nu, xi=xi, steps=total_steps,
            source_constraint=source_constraint, kl_constraint=kl,
            elapsed_seconds=time.monotonic()-started))
    s, r, t, source_base, source_y, source_d, foreign_d = full_values()
    c = feasible_scale(source_base, source_y, source_d, foreign_d, config,
                       source_constraint=source_constraint, local=local)
    function.scale = c
    s_after, r_after = constraint_values(source_base, source_y, source_d, foreign_d,
                                        local=local, scale=c)
    if not local:
        fit_d = function.values(fit.h[source_fit], unscaled=True).astype(np.float64)
        fit_a = np.asarray(bundle.pilots.logits('fit', source, task)[source_fit], np.float64)
        def kvalue(scale):
            d = scale*fit_d
            return float(np.mean(np.logaddexp(0, fit_a+d)-np.logaddexp(0, fit_a)-expit(fit_a)*d))
        k_before, k_after = kvalue(1), kvalue(c)
    else:
        k_before = k_after = None
    dq = c*function.values(pf.h[query_pos], unscaled=True).astype(np.float64)
    bq = np.asarray(bundle.base_oof[query_pos, task], np.float64)
    t_after = float(np.mean(omega*(np.logaddexp(0, bq+dq)-pf.y[query_pos, task]*(bq+dq))))
    function.metadata.update(source_constraint=source_constraint, kl=kl, local=local,
        query_rows=len(query_pos), query_rows_hash=array_hash(pf.row_id[query_pos]),
        omega_mean=float(np.mean(omega)), omega_hash=array_hash(omega), effective_steps=total_steps,
        query_label_visits=len(query_pos)*fc['passes'], source_fit_aux_visits=cycles[0].accessed,
        source_form_aux_visits=cycles[1].accessed,
        foreign_aux_visits=[cycle.accessed for cycle in cycles[2:]],
        auxiliary_cycles=[cycle.cycles for cycle in cycles],
        pre_projection=dict(S=s, R=r, K=k_before, T=t),
        post_projection=dict(S=s_after, R=r_after, K=k_after, T=t_after),
        projection_c=c, status='COLLAPSED_BY_CONSTRAINT' if c == 0 else 'FORMED',
        parameter_count=sum(p.numel() for p in model.parameters()),
        state_hash=state_hash(model.state_dict()), elapsed_seconds=time.monotonic()-started)
    function.training_state = dict(optimizer=copy.deepcopy(opt.state_dict()), nu=nu, xi=xi,
        next_pass=fc['passes']+1, next_query=0, auxiliary=[cycle.state() for cycle in cycles],
        torch_rng=torch.get_rng_state().clone(), query_rows=query_pos.copy(), omega=omega.copy())
    function.module.eval().requires_grad_(False)
    for p in function.module.parameters():
        p.grad = None
    # Keep complete final numerical state in RAM but release CUDA optimizer allocations.
    function.module.cpu()
    for state in function.training_state['optimizer']['state'].values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cpu()
    return records


def features_from_functions(bundle, functions, method, *, device=None):
    values, columns, bounds = {}, {}, {}
    for fn in functions.values():
        if device is not None:
            fn.module.to(device)
        for role, view in bundle.views.items():
            fn.outputs[role] = persist(bundle,('function_output',fn.function_id,role),fn.values(view.h))
        fn.module.cpu()
        fn.metadata['output_hashes'] = {r: array_hash(v) for r, v in fn.outputs.items()}
    for target in range(3):
        for task in range(2):
            heads = [fn for fn in functions.values() if fn.task == task and
                     (fn.target == target if fn.target is not None else fn.source != target)]
            heads.sort(key=lambda fn: (fn.source, fn.slot, fn.function_id))
            columns[target, task] = [dict(function_id=f.function_id, source=f.source,
                task=f.task, slot=f.slot, target=f.target, scale=f.scale,
                hidden_width=f.module.hidden.out_features) for f in heads]
            bounds[target, task] = [(-2.0, 2.0) if f.metadata.get('wide') else (-1.0, 1.0) for f in heads]
            for role, view in bundle.views.items():
                pos = np.flatnonzero(view.domain == target)
                value = np.column_stack([f.outputs[role][pos] for f in heads]) \
                    if heads else np.empty((len(pos), 0), dtype=np.float32)
                values[role, target, task] = persist(bundle,('formation_features',method,role,target,task),value)
    bank = FeatureBank(values=values, columns=columns, bounds=bounds,
        identity=dict(method=method, dataset=bundle.dataset, seed=bundle.seed,
                      columns_hash=digest(columns), function_hashes={k:v.metadata.get('state_hash') for k,v in functions.items()}))
    bank.functions = functions  # RAM-only full raw-X reconstruction dependency.
    release_data_pages(bundle)
    return bank


def form(bundle, demand, method, config, *, paired=None):
    allowed = {'RCF', 'POOLED_TARGET', 'LOCAL_ACTIVE', 'ASSIGN_SHUFFLE',
               'K1_WIDE', 'ANCHOR_OFF', 'SOURCE_INIT_RANDOM', 'SOURCE_LABEL_SHUFFLE'}
    if method not in allowed:
        raise ValueError(f'Unknown formation method {method}')
    real_bundle = bundle
    if method == 'SOURCE_LABEL_SHUFFLE':
        from .prefix import rebuild_shuffled_pilots
        from .demand import compute_demand
        bundle = rebuild_shuffled_pilots(bundle, config)
        demand = compute_demand(bundle, config)
    if method == 'K1_WIDE' and paired is None:
        raise ValueError('K1_WIDE requires the paired frozen RCF result')
    device = str(next(bundle.pilots.source_bank.parameters()).device)
    # Pilot may already be CPU after prefix; use explicit campaign device when supplied.
    device = config.get('_device', device)
    store = _TensorStore(bundle, device)
    functions, records = {}, []
    if method == 'LOCAL_ACTIVE':
        unit_specs = [(s, q, k) for s in range(3) for q in range(2) for k in range(4)]
    elif method == 'POOLED_TARGET':
        unit_specs = [(s, q, k) for s in range(3) for q in range(2) for k in range(2)]
    else:
        unit_specs = []
        for (s, q), unit in demand.units.items():
            k_count = unit['effective_k']
            unit_specs.extend((s, q, k) for k in range(1 if method == 'K1_WIDE' and k_count == 2 else k_count))
            if not k_count:
                records.append(dict(method=method, source=s, task=q, status=unit['status'], effective_k=0))
    for source, task, slot in unit_specs:
        local = method == 'LOCAL_ACTIVE'
        wide = method == 'K1_WIDE' and demand.units[source, task]['effective_k'] == 2
        if method == 'K1_WIDE' and demand.units[source, task]['effective_k'] == 1:
            original = next(f for f in paired.functions.values() if f.source == source and f.task == task)
            fn = copy.copy(original)
            fn.metadata = dict(original.metadata, method=method, reused_from=original.function_id,
                               status='REUSED_ORIGINAL_K1', additional_training_steps=0,
                               additional_training_seconds=0.0)
            fn.function_id = f'{method}_s{source}_q{task}_k0'
            functions[fn.function_id] = fn
            records.append(dict(fn.metadata, function_id=fn.function_id))
            continue
        if local:
            group, replica = divmod(slot, 2)
            pos = np.flatnonzero((bundle.views['P_form'].domain == source) &
                (bundle.views['P_form'].group[:, task] == group))
            omega, query_meta = np.ones(len(pos)), dict(target=source, group=group, replica=replica)
        elif method == 'POOLED_TARGET':
            pos, omega, query_meta = _pooled_request(bundle, source, task, config)
        else:
            pos, omega, query_meta = _query_request(bundle, demand.units[source, task], slot,
                shuffle=method == 'ASSIGN_SHUFFLE', wide=wide)
        if not len(pos):
            records.append(dict(method=method, source=source, task=task, slot=slot,
                status='EMPTY_LOCAL_GROUP' if local else 'LOW_SUPPORT_POOLED', optimizer_created=False))
            continue
        module = initialize_delta(bundle, source, task, slot, config,
            local_replica=replica if local else None, wide=wide,
            random_hidden=method == 'SOURCE_INIT_RANDOM')
        fid = f'{method}_s{source}_q{task}_k{slot}'
        fn = DeltaFunction(module=module, scale=1.0, source=source, task=task, slot=slot,
            function_id=fid, target=source if local else None,
            anchor_pilots=bundle.pilots,
            metadata=dict(method=method, wide=wide, query=query_meta,
                initial_state_hash=state_hash(module.state_dict()), zero_output_exact=True))
        records.extend(_train_slot(bundle, fn, pos, omega, config, store,
            source_constraint=method not in ('ANCHOR_OFF', 'LOCAL_ACTIVE'),
            kl=method not in ('ANCHOR_OFF', 'LOCAL_ACTIVE'), local=local))
        functions[fid] = fn
        records.append(dict(fn.metadata, function_id=fid, source=source, task=task, slot=slot,
                            phase='TERMINAL_PROJECTION'))
    del store
    features = features_from_functions(real_bundle, functions, method, device=device)
    identity = dict(method=method, dataset=bundle.dataset, seed=bundle.seed, prefix=bundle.identity,
        demand=demand.identity if method != 'LOCAL_ACTIVE' else 'NOT_APPLICABLE',
        functions_hash=digest({k:v.metadata for k,v in functions.items()}), memory_only=True,
        total_parameters=sum(sum(p.numel() for p in f.module.parameters()) for f in functions.values()))
    if method == 'K1_WIDE' and all(u['effective_k'] != 2 for u in demand.units.values()):
        identity['status'] = 'NOT_APPLICABLE_MULTIFUNCTION'
    result = FormationResult(method=method, features=features, functions=functions,
        requests=demand if method not in ('LOCAL_ACTIVE', 'POOLED_TARGET') else
            dict(status='NOT_APPLICABLE', independent_request=method),
        records=records, identity=identity)
    result.request_bundle = bundle  # RAM-only isolated shuffled pilot lineage, when applicable.
    return result
