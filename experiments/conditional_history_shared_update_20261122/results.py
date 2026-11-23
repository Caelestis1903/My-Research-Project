"""Validation-only direction A tables and finite gates. No model or data evaluation."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .protocol import DATASETS, SEEDS, digest, read, sha, write

CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
CORE = ('A00', 'A10', 'A20', 'A30')
HISTORIES = ('A22', 'A21', 'A20')  # Increasing complexity
DOSE = ('A40_HISTORY_L025', 'A41_ADAM_L025')
SCOPE = ('A50_STAR_SKIP_ONLY', 'A51_EXPERT_ONLY')
STAGES = ('A_CORE', 'A_HISTORY', 'A_DOSE', 'A_SCOPE')
PRIMARY = ('fixed_ema', 'selected')
GUARD = -.001
SIMPLE_BAND = .0001


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite endpoint')
    return value


def metrics(row):
    """Keep scoped metrics distinct. Pooled results must come from the evaluator."""
    values = {}
    counts = {cell: finite(row[cell + '_count']) for cell in CELLS}
    if any(n <= 0 or not n.is_integer() for n in counts.values()):
        raise ValueError('Invalid cell support')
    for metric in ('auc', 'logloss'):
        cells = {c: finite(row[f'{c}_{metric}']) for c in CELLS}
        if any(v < 0 or (metric == 'auc' and v > 1) for v in cells.values()):
            raise ValueError(f'Invalid {metric}')
        values.update({(c, metric): v for c, v in cells.items()})
        groups = {'macro': CELLS, **{f'domain{d}': CELLS[2*d:2*d+2] for d in range(3)},
                  **{f'task{t}': CELLS[t::2] for t in range(2)}}
        for group, members in groups.items():
            values[group, metric] = statistics.fmean(cells[c] for c in members)
        values['traffic', metric] = sum(cells[c] * counts[c] for c in CELLS) / sum(counts.values())
        # Keep the source's selected-model aggregates when provided, including its summation order.
        names = {'macro': f'macro_{metric}', 'traffic': f'sample_weighted_cell_{metric}'}
        for group, name in names.items():
            if row.get(name) not in ('', None):
                reported = finite(row[name])
                if not math.isclose(reported, values[group, metric], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f'{name} disagrees with cell metrics')
                values[group, metric] = reported
        for group in ('task0_pooled', 'task1_pooled'):
            val = row.get(f'{group}_{metric}')
            values[group, metric] = None if val in ('', None) else finite(val)
    for group, metric, name in [('pooled_task_mean','auc','pooled_task_mean_auc'),
                                ('pooled','logloss','pooled_logloss'), ('pooled_all_entry','auc','pooled_auc')]:
        val = row.get(name)
        values[group, metric] = None if val in ('', None) else finite(val)
    return values


def audited_reuse(inventory):
    """Consume only the source adapter's already audited Validation-only inventory."""
    source_hashes = {r['path']: r['sha256'] for r in inventory['sources']}
    for r in inventory['rows']:
        src = r['source_ref']
        yield {**r, 'config_hash':digest(r['config']), 'source_file_sha256':source_hashes[src]}


def completed_rows(path, result):
    receipt = read(path.with_name('COMPLETE_VERIFIED.json'))
    if (result['epochs'] != 50 or result['cost_class'] != 'NEW_SCIENTIFIC_TRAIN'
            or receipt['result_hash'] != digest(result) or receipt['owner_generation'] != result['owner_generation']):
        raise ValueError(f'Unverified science trajectory: {path}')
    for row in result['rows']:
        if row['checkpoint_rule'] == 'selected':
            kind = row['readout']
            if (row['epoch'],row['model_hash']) != (result['selected_epochs'][kind],result['selected_hashes'][kind]):
                raise ValueError(f'Selected endpoint identity mismatch: {path}')
        yield {**row, 'source_ref':str(path), 'data_hash':result['data_hash'], 'group_hash':result['group_hash'],
               'physical_attempt':result['attempt_id'], 'owner_generation':result['owner_generation']}



def index_rows(rows):
    index, trajectories, populations = {}, {}, {}
    for raw in rows:
        r = dict(raw)
        d, a = r['dataset'], r['arm']
        s, e = int(r['seed']), int(r['epoch'])
        kind, checkpoint = r['readout'], r['checkpoint_rule']
        if r['split'] != 'validation' or d not in DATASETS or s not in SEEDS:
            raise ValueError(f'Unexpected population {d}/{s}/{r["split"]}')
        if kind not in ('online', 'fixed_ema') or checkpoint not in ('selected', 'E10', 'E50'):
            raise ValueError(f'Unexpected readout/checkpoint {kind}/{checkpoint}')
        if not 1 <= e <= 50 or (checkpoint != 'selected' and e != int(checkpoint[1:])):
            raise ValueError(f'Invalid epoch {e} for {checkpoint}')
        if r['reuse_status'] not in ('NEW_TRAIN','REUSE_NUMERIC') or not r.get('source_ref'):
            raise ValueError('A science records require accepted trajectory provenance')
        if len(r['model_hash']) != 64 or any(c not in '0123456789abcdef' for c in r['model_hash']):
            raise ValueError('Invalid selected or checkpoint model SHA256')
        key = (d, a, s, kind, checkpoint)
        if key in index:
            raise ValueError(f'Duplicate endpoint, select one accepted trajectory upstream: {key}')
        t = r['trajectory_id']
        if not t or trajectories.setdefault((d,a,s), t) != t:
            raise ValueError(f'Mixed physical trajectories for {key}')
        vals = metrics(r)
        counts = tuple(int(float(r[c+'_count'])) for c in CELLS)
        if populations.setdefault(d, counts) != counts:
            raise ValueError(f'Validation cell supports changed for {key}')
        r.update(seed=s, epoch=e)
        index[key] = (r, vals)
    return index


def paired_stats(left, right):
    """Both maps must have exactly one finite observation per prescribed seed."""
    if set(left) != set(SEEDS) or set(right) != set(SEEDS):
        raise ValueError('Five paired training seeds required')
    l = [finite(left[s]) for s in SEEDS]
    r = [finite(right[s]) for s in SEEDS]
    diffs = [a-b for a,b in zip(l,r)]
    m, sd = statistics.fmean(diffs), statistics.stdev(diffs)
    radius = 2.776445 * sd / math.sqrt(5)
    return dict(n=5, mean_left=statistics.fmean(l), sd_left=statistics.stdev(l),
                mean_right=statistics.fmean(r), sd_right=statistics.stdev(r),
                mean_delta=m, sd_delta=sd, ci95_low=m-radius, ci95_high=m+radius,
                positive=sum(x>0 for x in diffs), negative=sum(x<0 for x in diffs),
                zero=diffs.count(0), **{f'delta_{s}':x for s,x in zip(SEEDS,diffs)})


def compare(index, dataset, left, right, readout, checkpoint):
    columns = [(dataset, arm, s, readout, checkpoint) for arm in (left,right) for s in SEEDS]
    if not all(key in index for key in columns):
        return [], []
    results, pairs = [], []
    shared = dict(dataset=dataset, comparison_id=left+'-'+right, left=left, right=right,
                  split='validation', readout=readout, checkpoint_rule=checkpoint)
    for cell, metric in index[columns[0]][1]:
        base = dict(shared, cell=cell, metric=metric)
        values = [{}, {}]
        for s in SEEDS:
            records = [index[dataset,a,s,readout,checkpoint] for a in (left,right)]
            lv, rv = (record[1][cell,metric] for record in records)
            if lv is None or rv is None:
                continue
            for bucket, val in zip(values, (lv,rv)):
                bucket[s] = val
            lr, rr = (record[0] for record in records)
            pairs.append(dict(base, seed=s, left_value=lv, right_value=rv, delta=lv-rv,
                              left_epoch=lr['epoch'], right_epoch=rr['epoch'],
                              left_hash=lr['model_hash'], right_hash=rr['model_hash'],
                              left_source=lr['source_ref'], right_source=rr['source_ref']))
        if len(values[0]) == len(SEEDS):
            stats = paired_stats(*values)
            guard = stats['mean_delta'] >= GUARD if metric == 'auc' and cell in CELLS else None
            results.append(dict(base, **stats, guard=guard))
    return results, pairs


def primary_effect(index, dataset, arm, control):
    delta = {}
    mean = {}
    for c in (*CELLS, 'macro'):
        for metric in ('auc', 'logloss'):
            maps = [{s:index[dataset,a,s,*PRIMARY][1][c,metric] for s in SEEDS} for a in (arm,control)]
            st = paired_stats(*maps)
            delta[c,metric], mean[c,metric] = st['mean_delta'], st['mean_left']
    by_cell = {c:delta[c,'auc'] for c in CELLS}
    worst = min(by_cell, key=by_cell.get)
    return dict(macro=mean['macro','auc'], delta_auc=delta['macro','auc'],
                delta_logloss=delta['macro','logloss'], delta_auc_cells=by_cell, worst_cell=worst,
                worst_cell_delta=by_cell[worst], guard=by_cell[worst] >= GUARD)


def preferred(options, effects, simplicity_order):
    """Prefer protected options, then the simplest within the registered macro band."""
    pool = [a for a in options if effects[a]['guard']] or list(options)
    best = max(effects[a]['macro'] for a in pool)
    return next(a for a in simplicity_order if a in pool and best-effects[a]['macro'] <= SIMPLE_BAND)


def localization_evidence(rows, dataset, arm, history, damage):
    """Same-trajectory whole-group shadow replacements on fixed Validation/EMA panels."""
    valid = {}
    for r in rows:
        if (r['dataset'],r['arm'],r['history_form']) != (dataset,arm,history):
            continue
        if (r['split'],r['readout'],r['action']) != ('validation','fixed_ema','whole_group_to_adam'):
            continue
        key = (r['group'], int(r['seed']), int(r['anchor_epoch']),r['scale'])
        if key in valid or not r.get('source_ref'):
            raise ValueError(f'Duplicate/unidentified localization observation {key}')
        valid[key] = r
    evidence = {}
    for group in ('STAR_SKIP', 'EXPERT'):
        by_scale = {}
        for scale in ('raw','norm_matched'):
            keys = [(group,s,e,scale) for s in SEEDS for e in (5,20,35,50)]
            if not all(k in valid for k in keys):
                by_scale[scale] = None
                continue
            # All registered anchors of each seed contribute once. An unmatchable norm is explicit.
            if any(valid[k].get('status','OK') != 'OK' for k in keys):
                by_scale[scale] = {'informative': False}
                continue
            by_scale[scale] = {cell:statistics.fmean(
                statistics.fmean(finite(valid[group,s,e,scale][cell+'_delta_auc']) for e in (5,20,35,50))
                for s in SEEDS) for cell in CELLS}
            by_scale[scale]['macro'] = statistics.fmean(by_scale[scale].values())
            by_scale[scale]['informative'] = True
        confirmed = [c for c in damage if all(b and b.get('informative') and b['macro']>0 and b[c]>0
                                             for b in by_scale.values())]
        evidence[group] = dict(by_scale, confirmed_damaged_cells=confirmed)
    return evidence


def stage_arms(stage, previous):
    if stage == 'A_CORE':
        return CORE
    if stage not in STAGES or not previous:
        raise ValueError(f'{stage} requires a completed preceding gate')
    if previous['stage'] != STAGES[STAGES.index(stage)-1] or previous['status'] != 'COMPLETE' or stage not in previous['next_stages']:
        raise ValueError(f'{stage} has no preceding gate authorization')
    if stage == 'A_HISTORY':
        return (*CORE, 'A21','A22')
    common = ('A00','A10',previous['history_form'],*DOSE)
    return (*common,*SCOPE) if stage == 'A_SCOPE' else common


def stage_decision(stage, index, previous=None, localization=None):
    arms = stage_arms(stage, previous)
    missing = [[d,a,s] for d in DATASETS for a in arms for s in SEEDS if (d,a,s,*PRIMARY) not in index]
    out = dict(stage=stage, status='TECHNICAL_BLOCKED' if missing else 'COMPLETE', split='validation',
               readout=PRIMARY[0], checkpoint_rule=PRIMARY[1], seeds=list(SEEDS),
               missing_primary=missing, datasets={}, next_stages=[])
    if missing:
        out['reason'] = 'Awaiting accepted complete trajectories for missing seed identities'
        return out
    for d in DATASETS:
        control = {a:primary_effect(index,d,a,'A00') for a in arms if a != 'A00'}
        result = out['datasets'][d] = dict(vs_a00=control)
        if stage == 'A_CORE':
            h = primary_effect(index,d,'A20','A10')
            result.update(history_vs_adam=h, dynamic_vs_static=primary_effect(index,d,'A30','A10'),
                          signal='POSITIVE_MEAN_SIGNAL' if h['delta_auc']>0 else
                          'PAIR_ONLY' if control['A10']['delta_auc']>0 else 'NO_CONTINUATION_SIGNAL')
            continue
        if stage == 'A_HISTORY':
            winner = preferred((*HISTORIES,'A10'),control,('A10',*HISTORIES))
            matched = {a:primary_effect(index,d,a,'A10') for a in HISTORIES}
            result.update(preferred=winner, history_vs_adam=matched,
                          conditional_vs_collapsed=primary_effect(index,d,'A20','A21'),
                          collapsed_vs_single=primary_effect(index,d,'A21','A22'),
                          conditional_vs_single=primary_effect(index,d,'A20','A22'))
            result['eligible_history'] = (winner != 'A10' and control[winner]['delta_auc']>0
                                          and matched[winner]['delta_auc']>0)
            continue
        h = previous['history_form']
        if stage == 'A_DOSE':
            # A simpler matched Adam takes precedence inside the resolution band.
            # Within the same complexity select the higher macro; exact ties retain the original dose.
            order = sorted(('A10',DOSE[1]), key=lambda a:-control[a]['macro'])
            order += sorted((h,DOSE[0]), key=lambda a:-control[a]['macro'])
            winner = preferred(order,control,order)
            low = winner in DOSE
            chosen_h, adam = (DOSE if low else (h,'A10'))
            matched = primary_effect(index,d,chosen_h,adam)
            result.update(preferred=winner, lambda_pair=.025 if low else .05, matched_adam=adam,
                          full_history=chosen_h, history_vs_adam=matched,
                          reduced_dose_vs_matched_adam=primary_effect(index,d,*DOSE))
            harmed = [c for c in CELLS if control[chosen_h]['delta_auc_cells'][c]<GUARD]
            actionable = winner == chosen_h and control[winner]['delta_auc']>0 and matched['delta_auc']>0
            result['damaged_cells'] = harmed
            # Fixed-panel interventions can also show a present cost when selected endpoints pass.
            if actionable:
                def has_probe(arm):
                    return any((r['dataset'],r['arm'],r['history_form']) == (d,arm,h) for r in localization or [])
                reduced_cost = not control[DOSE[0]]['guard']
                if not reduced_cost and has_probe(DOSE[0]):
                    reduced = localization_evidence(localization,d,DOSE[0],h,CELLS)
                    result['reduced_dose_localization'] = reduced
                    reduced_cost = any(v['confirmed_damaged_cells'] for v in reduced.values())
                if reduced_cost and (harmed or has_probe(chosen_h)):
                    result['localization'] = localization_evidence(localization or [], d,chosen_h,h,harmed or CELLS)
            continue
        cfg = previous['datasets'][d]
        full, adam = cfg['full_history'], cfg['matched_adam']
        candidates = (full,*SCOPE)
        # Predeclared complexity ordering is between history forms. Scope comparisons use macro.
        ordered = sorted(candidates, key=lambda a:(-control[a]['guard'],-control[a]['macro']))
        winner = ordered[0]
        matched = {a:primary_effect(index,d,a,adam) for a in candidates}
        result.update(preferred=winner, lambda_pair=cfg['lambda_pair'], matched_adam=adam,
                      full_history=full, history_vs_adam=matched,
                      vs_full_scope={a:primary_effect(index,d,a,full) for a in SCOPE},
                      eligible_history=control[winner]['delta_auc']>0 and matched[winner]['delta_auc']>0)
    datasets = out['datasets']
    if stage == 'A_CORE':
        if any(r['signal']=='POSITIVE_MEAN_SIGNAL' for r in datasets.values()):
            out['next_stages'] = ['A_HISTORY']
    elif stage == 'A_HISTORY':
        eligible = [d for d in DATASETS if datasets[d]['eligible_history']]
        if eligible:
            target = min(eligible, key=lambda d:(-datasets[d]['history_vs_adam'][datasets[d]['preferred']]['delta_auc'], d))
            h = datasets[target]['preferred']
            out.update(target_dataset=target, history_form=h, scope='S_DENSE')
            current_cost = any(not datasets[d]['vs_a00'][h]['guard'] for d in DATASETS)
            historical_cost = h == 'A20'  # Registered P01 motivation only; no numerical Test input.
            out['dose_motivation'] = dict(current_cell_tradeoff=current_cost, preregistered_P01_cost=historical_cost)
            if current_cost or historical_cost:
                out['next_stages'] = ['A_DOSE']
    else:
        target = previous['target_dataset']
        out.update(target_dataset=target, history_form=previous['history_form'], scope='S_DENSE')
        if stage == 'A_DOSE':
            groups = [v for r in datasets.values() for v in r.get('localization',{}).values()]
            observed_groups = groups+[v for r in datasets.values() for v in r.get('reduced_dose_localization',{}).values()]
            if any(g['confirmed_damaged_cells'] for g in groups):
                out['next_stages'] = ['A_SCOPE']
            elif any(g['raw'] is None or g['norm_matched'] is None for g in observed_groups):
                out['status'] = 'TECHNICAL_BLOCKED'
                out['reason'] = 'Current cell cost needs complete same-trajectory localization before scope decision'
        else:
            chosen = datasets[target]
            if chosen['eligible_history']:
                winner = chosen['preferred']
                out['scope'] = {SCOPE[0]:'STAR_SKIP', SCOPE[1]:'EXPERT'}.get(winner,'S_DENSE')
            else:
                out['scope'] = None
                out['reason'] = 'Target dataset lacks independent history gain for the preferred scope'
    return out


def comparisons(stage, previous, dataset):
    if stage == 'A_CORE':
        return [('A10','A00'),('A20','A10'),('A30','A10'),('A20','A00'),('A30','A00')]
    if stage == 'A_HISTORY':
        return [(a,b) for a in HISTORIES for b in ('A00','A10')] + [
            ('A20','A21'),('A20','A22'),('A21','A22'),('A10','A00')]
    h = previous['history_form']
    if stage == 'A_DOSE':
        return [DOSE,(DOSE[0],h),(DOSE[1],'A10'),(h,'A10'),
                *((a,'A00') for a in (h,'A10',*DOSE))]
    cfg = previous['datasets'][dataset]
    return [(a,b) for a in SCOPE for b in ('A00',cfg['matched_adam'],cfg['full_history'])] + [
        (cfg['full_history'],cfg['matched_adam']),(cfg['full_history'],'A00')]


def write_csv(path, rows, default_fields):
    import os
    from tempfile import NamedTemporaryFile

    fields = list(dict.fromkeys(k for r in rows for k in r)) or default_fields
    with NamedTemporaryFile(mode='w', dir=path.parent, newline='', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(f.name,path)


def paired_plot(entries, path):
    """Plot mean and paired t4 interval together with all five seed differences."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    selected = [r for r in entries if r['cell']=='macro' and r['metric']=='auc' and
                (r['readout'],r['checkpoint_rule']) == PRIMARY]
    figure = Figure(figsize=(10,max(3,.42*len(selected)+1.2)), layout='constrained')
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.axvline(0, color='0.45', lw=.8)
    ax.set_xlabel('Paired change in Validation Macro AUC (method − control)')
    labels = []
    for i,r in enumerate(selected):
        mean = r['mean_delta']
        ax.plot([r['ci95_low'],r['ci95_high']],[i,i], lw=2, color='#236485')
        ax.scatter(mean,i, c='#236485',marker='D',s=30, zorder=3)
        ax.scatter([r[f'delta_{s}'] for s in SEEDS],[i-.15]*5, color='#a14d30',s=12,zorder=3)
        labels.append(f"{r['dataset']}  {r['left']} − {r['right']}")
    ax.set_yticks(range(len(labels)),labels=labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.grid(axis='x', color='0.9', lw=.6)
    ax.set_title('Selected fixed EMA • five seed differences, mean and 95% paired t interval'
                 if selected else 'Awaiting complete five-seed comparisons')
    figure.savefig(path,dpi=160)


def freeze_identities(rows, roles_by_dataset, comparisons_by_dataset, dataset_parameters, scripts):
    """Construct the scalar content for a freeze; this function does not release Test.

    Role keys are display identities (candidate, matched_adam, strongest_published, ...).
    Each value is (arm, readout); a published online baseline can have its own frozen readout.
    """
    index = index_rows(rows)
    if not scripts or any(set(obj) != set(DATASETS) for obj in
                          (roles_by_dataset, comparisons_by_dataset, dataset_parameters)):
        raise ValueError('Require both datasets and identified evaluation/statistics scripts')
    packet = dict(schema=1, checkpoint_rule='earliest strict Validation Macro AUC max E1-E50',
                  dataset_parameters=dataset_parameters, comparisons=comparisons_by_dataset,
                  scripts={str(p):sha(p) for p in (Path(__file__),*scripts)}, selected=[])
    for d in DATASETS:
        roles = roles_by_dataset[d]
        if not {'candidate','strong_internal','strongest_published'} <= set(roles):
            raise ValueError(f'{d} lacks candidate / six-cell guard reference roles')
        comparison_set = {tuple(pair) for pair in comparisons_by_dataset[d]}
        if any(len(pair)!=2 or not set(pair)<=set(roles) for pair in comparison_set):
            raise ValueError(f'Unresolved comparison reference for {d}')
        required = {('candidate','strong_internal'),('candidate','strongest_published')}
        if not required <= comparison_set or not dataset_parameters[d]:
            raise ValueError(f'Unresolved frozen configuration/comparisons for {d}')
        for role,(arm,kind) in roles.items():
            for s in SEEDS:
                r = index[d,arm,s,kind,'selected'][0]
                identity = {k:r[k] for k in ('dataset','arm','seed','readout','checkpoint_rule','epoch',
                            'model_hash','trajectory_id','source_ref','reuse_status')}
                for k in ('config_hash','data_hash','group_hash','source_hash','source_file_sha256',
                          'owner_generation','physical_attempt'):
                    if k in r:
                        identity[k] = r[k]
                packet['selected'].append(dict(role=role, **identity))
    return {**packet,'identity_sha256':digest(packet)}


def write_stage(stage, rows, output, previous=None, localization=None):
    """Materialize stage-specific scalar tables. Incomplete primary sets cannot activate a gate."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    index = index_rows(rows)
    gate = stage_decision(stage,index,previous,localization)
    endpoints, summaries, pairs = [], [], []
    for key,(r,values) in sorted(index.items()):
        identity = {k:v for k,v in r.items() if not (k.startswith(CELLS) or
                    k.endswith(('_auc','_logloss')) or k=='pooled_auc')}
        for c in CELLS:
            endpoints.append(dict({**identity, 'study_id':'conditional_history_shared_update_20261122'}, cell=c,
                                  auc=values[c,'auc'], logloss=values[c,'logloss'], support=int(float(r[c+'_count'])),
                                  positive=r.get(c+'_positive'), negative=r.get(c+'_negative')))
        for c in dict.fromkeys(cell for cell,_ in values if cell not in CELLS):
            endpoints.append(dict({**identity, 'study_id':'conditional_history_shared_update_20261122'}, cell=c,
                                  auc=values.get((c,'auc')), logloss=values.get((c,'logloss'))))
    for d in DATASETS:
        for left,right in comparisons(stage,previous,d):
            for kind in ('fixed_ema','online'):
                for cp in ('selected','E10','E50'):
                    stats, raw = compare(index,d,left,right,kind,cp)
                    summaries.extend(stats)
                    pairs.extend(raw)
    for name, values, fields in [('endpoints',endpoints,['dataset','arm','seed','epoch','cell','auc','logloss']),
                                 ('paired_seed',pairs,['comparison_id','dataset','seed','cell','metric','delta']),
                                 ('paired_summary',summaries,['comparison_id','dataset','cell','metric','n','mean_delta'])]:
        write_csv(out/(name+'.csv'),values,fields)
    paired_plot(summaries,out/'paired_macro_auc.png')
    gate['evidence'] = {name:sha(out/name) for name in ('endpoints.csv','paired_seed.csv','paired_summary.csv')}
    gate['statistics_script_sha256'] = sha(__file__)
    if previous:
        gate['previous_decision_sha256'] = digest(previous)
    if localization:
        gate['localization_records_sha256'] = digest(localization)
    write(out/'decisions.json',gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=STAGES)
    parser.add_argument('--input',type=Path,action='append',required=True,
                        help='Normalized wide CSV or audited REUSE_MANIFEST.json / verified run result.json')
    parser.add_argument('--output',type=Path,required=True,help='Stage-specific output directory')
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--localization',type=Path)
    args = parser.parse_args()
    records = []
    for path in args.input:
        if path.suffix == '.json':
            data = read(path)
            records.extend(completed_rows(path,data) if path.name=='result.json' else audited_reuse(data))
        else:
            with path.open(newline='') as f:
                records.extend(csv.DictReader(f))
    gate = write_stage(args.stage,records,args.output,read(args.previous) if args.previous else None,
                       read(args.localization) if args.localization else None)
    print(json.dumps({k:gate[k] for k in ('stage','status','next_stages')}))


if __name__ == '__main__':
    main()
