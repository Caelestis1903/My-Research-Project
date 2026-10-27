"""Selected-state interventions, source audits and measured execution costs."""
from __future__ import annotations
import time
import json
import numpy as np
import torch
from .data import tensor
from .metrics import evaluate
from .native import synchronize
from .trainer import predict
from .state import array_hash


@torch.no_grad()
def mean_relations(model, view, device):
    totals = np.zeros((3, 2, 2*model.basis_count+1), np.float64)
    counts = np.zeros(3, np.int64)
    for begin in range(0, len(view['ids']), 4096):
        sl = slice(begin, begin+4096)
        d = tensor(view['domain'][sl], device)
        hz = torch.cat((tensor(view['h'][sl], device), tensor(view['z'][sl], device)), -1)
        alpha, null = model.relation(hz, d)
        routes = torch.cat((alpha.flatten(2), null[:, :, None]), -1).cpu().numpy()
        for target in range(3):
            rows = view['domain'][sl] == target
            totals[target] += routes[rows].sum(0, dtype=np.float64)
            counts[target] += int(rows.sum())
    return torch.as_tensor((totals/counts[:, None, None]).astype(np.float32))


def metric_difference(alternative, actual):
    return dict(macro_logloss=alternative['macro_logloss']-actual['macro_logloss'],
                macro_auc=alternative['macro_auc']-actual['macro_auc'],
                natural_logloss=alternative['natural_logloss']-actual['natural_logloss'],
                cell_logloss=[a['logloss']-b['logloss'] for a, b in zip(alternative['cells'], actual['cells'])])


@torch.no_grad()
def diagnose(model, data, views, source, device):
    if not model.reads_source:
        return dict(status='NO_SOURCE_PATH', source_constraint_lambda=0.)
    started = time.monotonic()
    model.eval()
    q = views['Q']
    y, mask = data.labels('Q', 'diagnostic', method=model.method)
    full = evaluate(y, predict(model, q, device), q['domain'], mask)
    root = evaluate(y, predict(model, q, device, root=True), q['domain'], mask)
    constant = mean_relations(model, views['F'], device)
    fixed = evaluate(y, predict(model, q, device, constant_alpha=constant), q['domain'], mask)
    totals = np.zeros((3, 2, 2*model.basis_count+1), np.float64)
    square = np.zeros_like(totals)
    entropy = np.zeros((3, 2), np.float64)
    counts = np.zeros(3, np.int64)
    function_ss = np.zeros(3, np.float64)
    function_n = np.zeros(3, np.int64)
    for begin in range(0, len(q['ids']), 4096):
        sl = slice(begin, begin+4096)
        h, z, d = tensor(q['h'][sl], device), tensor(q['z'][sl], device), tensor(q['domain'][sl], device)
        alpha, null = model.relation(torch.cat((h, z), -1), d)
        routes = torch.cat((alpha.flatten(2), null[:, :, None]), -1).cpu().numpy().astype(np.float64)
        for target in range(3):
            rows = q['domain'][sl] == target
            totals[target] += routes[rows].sum(0)
            square[target] += (routes[rows]**2).sum(0)
            entropy[target] += (-(routes[rows]*np.log(np.maximum(routes[rows], 1e-300))).sum(-1)).sum(0)
            counts[target] += int(rows.sum())
        for s in range(3):
            f = model.function(tensor(q['phi'][sl, s], device), s)
            if model.basis_count > 1:
                function_ss[s] += float((f[:, 0]-f[:, 1]).double().square().sum())
                function_n[s] += len(f)*64
    mean = totals/counts[:, None, None]
    std = np.sqrt(np.maximum(square/counts[:, None, None]-mean*mean, 0))
    distances = np.sqrt(function_ss/np.maximum(function_n, 1))
    u_norms = model.U.detach().norm(dim=(-1, -2)).cpu().numpy().tolist() if model.forms_basis else []
    source_audit = []
    ss = views['S']
    ys, masks = data.labels('S', 'diagnostic', method=model.method)
    for s in range(3):
        own = np.flatnonzero(ss['domain'] == s)
        teacher_logits, replaced = [], [[] for _ in range(model.basis_count)]
        rms_ss = np.zeros(model.basis_count, np.float64)
        for begin in range(0, len(own), 4096):
            pos = own[begin:begin+4096]
            phi, z = tensor(ss['phi'][pos, s], device), tensor(ss['z'][pos], device)
            teacher_logits.append(source.teacher(phi, z, s).cpu().numpy())
            formed = model.function(phi, s)
            for j in range(model.basis_count):
                replaced[j].append(source.teacher(formed[:, j], z, s).cpu().numpy())
                rms_ss[j] += float((formed[:, j]-phi).double().square().sum())
        base_z = np.concatenate(teacher_logits)
        domain = np.full(len(own), s)
        base = evaluate(ys[own], base_z, domain, masks[own])
        for j in range(model.basis_count):
            changed = evaluate(ys[own], np.concatenate(replaced[j]), domain, masks[own])
            source_audit.append(dict(source=s, basis=j, baseline=base, replaced=changed,
                                      delta_natural_logloss=changed['natural_logloss']-base['natural_logloss'],
                                      output_deviation_rms=float(np.sqrt(rms_ss[j]/(len(own)*64)))))
    flags = []
    if np.max(distances) <= np.finfo(np.float32).eps:
        flags.append('FUNCTION_COLLAPSE')
    difference = metric_difference(root, full)
    if abs(difference['macro_logloss']) <= 1e-12 and float(std.max()) > np.finfo(np.float32).eps:
        flags.append('ROUTING_WITHOUT_FUNCTIONAL_VALUE')
    synchronize(device)
    return dict(status='COMPLETE', panel='Q_all_rows', full=full, all_root=root,
                all_root_minus_full=difference, constant_relation=fixed,
                constant_minus_full=metric_difference(fixed, full),
                constant_relation_fit_role='F_inputs_only', constant_alpha=constant.tolist(),
                route_mean=mean.tolist(), route_std=std.tolist(), entropy=(entropy/counts[:, None]).tolist(),
                route_order='foreign_source_ascending_then_basis; final_NULL',
                foreign_sources=model.relation.foreign.tolist(),
                basis_output_rms_distance=distances.tolist(), U_norm=u_norms,
                formation_evidence=bool(model.forms_basis and any(x > 0 for row in u_norms for x in row)
                                        and difference['macro_logloss'] > 1e-12),
                source_retention=source_audit, source_constraint_lambda=0.,
                flags=flags, seconds=time.monotonic()-started)


@torch.no_grad()
def raw_forward(native, source, model, x):
    packet = native.encode(x)
    domain = x[:, -1].long()
    phi = source(packet.h, domain) if model.reads_source else None
    return model(packet.h, packet.logits, domain, phi)


def compare_raw_logits(raw, cached, row_ids=None):
    """Cross-batch FP32 check; cached logits are the explicit reference."""
    atol, rtol = 1e-5, 1.3e-6
    record = dict(atol=atol, rtol=rtol, comparison_reference='cached',
                  raw_shape=list(raw.shape), cached_shape=list(cached.shape),
                  batch=len(cached), finite=False, passed=False,
                  max_abs=None, mean_abs=None, max_index=None,
                  cached_value=None, raw_value=None, allowed_at_max=None)
    if row_ids is not None:
        ids = np.asarray(row_ids, dtype=np.int64)
        record.update(row_id_hash=array_hash(ids), row_id_first=int(ids[0]),
                      row_id_last=int(ids[-1]))
    if raw.shape != cached.shape:
        record['reason'] = 'shape_mismatch'
    elif not bool(torch.isfinite(raw).all() and torch.isfinite(cached).all()):
        record['reason'] = 'nonfinite_logits'
    else:
        difference = (raw.detach().double()-cached.detach().double()).abs()
        allowance = atol+rtol*cached.detach().double().abs()
        index = tuple(int(i) for i in np.unravel_index(int(difference.argmax()), tuple(difference.shape)))
        record.update(finite=True, passed=bool((difference <= allowance).all()),
                      max_abs=float(difference[index]), mean_abs=float(difference.mean()),
                      max_index=list(index), cached_value=float(cached[index]),
                      raw_value=float(raw[index]), allowed_at_max=float(allowance[index]))
        ratio = difference/allowance
        worst = tuple(int(i) for i in np.unravel_index(int(ratio.argmax()), tuple(ratio.shape)))
        record.update(max_normalized_error=float(ratio[worst]),
                      worst_index=list(worst), worst_abs=float(difference[worst]),
                      worst_cached_value=float(cached[worst]), worst_raw_value=float(raw[worst]),
                      worst_allowed=float(allowance[worst]))
        record['reason'] = 'within_tolerance' if record['passed'] else 'logit_tolerance_exceeded'
    if not record['passed']:
        raise ValueError('Raw-X reconstruction mismatch: '+json.dumps(record, allow_nan=False))
    return record


@torch.no_grad()
def latency(native, source, model, data, view, device):
    model.eval()
    records = []
    for batch in (1, 256):
        ids = data.roles['Q'][:batch]
        x = tensor(data.x[ids], device)
        packet = native.encode(x)
        domain = x[:, -1].long()
        phi = source(packet.h, domain) if model.reads_source else None
        expected = model(tensor(view['h'][:batch], device), tensor(view['z'][:batch], device), domain,
                         tensor(view['phi'][:batch], device) if model.reads_source else None)
        actual = raw_forward(native, source, model, x)
        comparison = compare_raw_logits(actual, expected, ids)
        max_diff = comparison['max_abs']
        for kind in ('raw_X', 'cached_prefix'):
            def run():
                return raw_forward(native, source, model, x) if kind == 'raw_X' else model(packet.h, packet.logits, domain, phi)
            for _ in range(50):
                run()
            times = []
            for _ in range(200):
                synchronize(device)
                started = time.perf_counter()
                run()
                synchronize(device)
                times.append((time.perf_counter()-started)*1000)
            records.append(dict(batch=batch, kind=kind, warmup=50, measurements=200,
                                p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
                                raw_x_max_abs_diff=max_diff, raw_x_comparison=comparison))
    native_count = sum(p.numel() for p in native.parameters())
    hidden_source_count = 49344 if model.reads_source else 0
    # Linear hooks are an explicit partial MAC inventory; elementwise/star/einsum work is listed separately.
    macs = {}
    handles = []
    def hook(name):
        def record(module, inputs, output):
            macs[name] = macs.get(name, 0) + int(inputs[0].numel()*module.out_features)
        return record
    for prefix, module in [('native', native), ('source', source), ('adaptation', model)]:
        for name, layer in module.named_modules():
            if isinstance(layer, torch.nn.Linear):
                handles.append(layer.register_forward_hook(hook(prefix+'.'+name)))
    x = tensor(data.x[data.roles['Q'][:256]], device)
    raw_forward(native, source, model, x)
    for handle in handles:
        handle.remove()
    distinct_domains = np.unique(data.domain[data.roles['Q'][:256]])
    source_calls = sum(bool(np.any(data.domain[data.roles['Q'][:256]] != s)) for s in range(3)) if model.reads_source else 0
    function_calls = 2*len(distinct_domains)*model.basis_count if model.forms_basis else 0
    return dict(device=str(device), device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
                latency=records, native_parameters=native_count, frozen_source_hidden_parameters=hidden_source_count,
                audit_only_teacher_parameters=198 if model.reads_source else 0,
                adaptation_parameters=model.parameter_counts(),
                deployment_parameters=native_count+hidden_source_count+model.parameter_counts()['adaptation'],
                batch256_source_function_calls=source_calls, batch256_residual_function_calls=function_calls,
                batch256_linear_MAC=sum(macs.values()), linear_MAC_by_module=macs,
                MAC_scope='nn.Linear multiply-accumulates; excludes embeddings, LayerNorm, activations, STAR matrix products and explicit einsum relations/function residuals',
                explicit_residual_MAC_per_row=2*model.basis_count*(64*4+4*64) if model.forms_basis else 0,
                explicit_lowrank_relation_MAC_per_row=2*2*model.basis_count*2 if model.method == 'LOWRANK_RELATION' else 0,
                sparse_execution=False)

