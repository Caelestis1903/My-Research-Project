"""Dataset-level five-seed validation selection and immutable endpoint identities."""
import json
import math
import statistics
from pathlib import Path
from experiments.overnight_ab_20261103.selection import CELLS, score
from experiments.overnight_ab_20261103.state import read_json, write_json, sha_file, digest
from . import guardian
from .protocol import SEEDS


def complete_summaries(output):
    prepared = read_json(Path(output)/'PREPARED.json')
    summaries = guardian.request(output,'summaries')['summaries']
    for job in prepared['jobs']:
        key = job['job_id']
        if key not in summaries or not summaries[key]['complete'] or summaries[key]['epoch']!=50:
            raise Incomplete('All candidates require five complete seeds and retained RAM endpoints')
        for epoch in range(1,51):
            row = read_json(guardian.epoch_path(output,key,epoch))
            if set(row['validation']) != {'online',*(c['id'] for c in job['candidates'])}:
                raise ValueError('Missing full-epoch validation estimator')
        if any(set(points) != {'selected','E10','E50'} for points in summaries[key]['snapshots'].values()):
            raise Incomplete('Required RAM endpoint is missing')
    return prepared,summaries


class Incomplete(RuntimeError):
    pass


def rank_family(rows):
    if any(row['seed_ids'] != SEEDS or not row['complete'] for row in rows):
        raise Incomplete('Each HPO candidate requires all five seeds')
    def order(row):
        return (-row['score'],row['measured_cost_seconds'],abs(math.log2(row['half_life_ratio'])),row['candidate_id'])
    unconstrained = min(rows,key=order)
    protected = [r for r in rows if r['cell_protected']]
    chosen = min(protected or rows,key=order)
    return dict(candidate_id=chosen['candidate_id'],unconstrained_score_winner=unconstrained['candidate_id'],
        status='CELL_PROTECTED' if protected else 'VALIDATION_CELL_TRADEOFF')


def select(output):
    from .report import write_csv
    output = Path(output)
    prepared,summaries = complete_summaries(output)
    curves,hpo,selected = [],[],{}
    for dataset in ('movielens','kuairand'):
        peers = sorted([row for row in summaries.values() if row['job']['dataset']==dataset],key=lambda r:r['job']['seed'])
        if [row['job']['seed'] for row in peers] != SEEDS:
            raise Incomplete('Five dataset seeds are required')
        candidates = peers[0]['job']['candidates']
        for row in peers:
            for epoch in range(1,51):
                record = read_json(guardian.epoch_path(output,row['job']['job_id'],epoch))
                for candidate,values in record['validation'].items():
                    curves.append(dict(dataset=dataset,seed=row['job']['seed'],trajectory_id=row['job']['trajectory_id'],
                        candidate_id=candidate,epoch=epoch,model_hash=record['model_hash'][candidate],
                        training_loss=record['training_loss'],clean_loss=record['clean_loss'],**values))
        for candidate in candidates:
            cid = candidate['id']
            values = [r['snapshots'][cid]['selected']['validation'] for r in peers]
            reference = [r['snapshots']['F']['selected']['validation'] for r in peers]
            deltas = {cell:math.fsum(v[cell+'_auc']-f[cell+'_auc'] for v,f in zip(values,reference))/5 for cell in CELLS}
            actual_cost = sum(r['candidate_cost'][cid] for r in peers)
            if candidate['family'] != 'U':
                actual_cost += sum(r['cost']['probe_seconds'] for r in peers)
            row = dict(dataset=dataset,candidate_id=cid,family=candidate['family'],
                half_life_ratio=candidate['half_life_ratio'],refresh_steps=candidate.get('refresh_steps'),
                seed_ids=SEEDS,complete=True,score=math.fsum(score(v) for v in values)/5,
                score_sd=statistics.stdev(score(v) for v in values),
                measured_cost_seconds=actual_cost,cell_protected=all(v>=-.001 for v in deltas.values()),
                selected_epochs=[r['snapshots'][cid]['selected']['epoch'] for r in peers],
                selected_validation_scores=[score(v) for v in values],config=candidate,
                **{cell+'_delta_F':value for cell,value in deltas.items()})
            for metric in values[0]:
                if all(v[metric] is not None for v in values):
                    row[metric+'_mean'] = math.fsum(v[metric] for v in values)/5
                    row[metric+'_sd'] = statistics.stdev(v[metric] for v in values)
            hpo.append(row)
        current = [r for r in hpo if r['dataset']==dataset]
        winners = {f:rank_family([r for r in current if r['family']==f]) for f in ('D','U','M','S')}
        directed = next(r for r in candidates if r['id']==winners['D']['candidate_id'])
        matched = {f:next(r['id'] for r in candidates if r['family']==f and
            r['half_life_ratio']==directed['half_life_ratio'] and r.get('refresh_steps')==directed.get('refresh_steps')) for f in ('M','S')}
        report_ids = sorted({'online','F',*(r['candidate_id'] for r in winners.values()),*matched.values()})
        endpoints = []
        for row in peers:
            for cid in report_ids:
                for endpoint,point in row['snapshots'][cid].items():
                    endpoints.append(dict(seed=row['job']['seed'],candidate_id=cid,checkpoint=endpoint,
                        trajectory_id=row['job']['trajectory_id'],job_id=row['job']['job_id'],**point))
        selected[dataset] = dict(winners=winners,matched_at_D=matched,report_candidates=report_ids,
            candidates=candidates,endpoints=endpoints,cost={k:sum(r['cost'][k] for r in peers)
                if not k.startswith('peak_') else max(r['cost'][k] for r in peers) for k in peers[0]['cost']})
    route = output/'B'; route.mkdir(exist_ok=True)
    write_csv(route/'VALIDATION_CURVES.csv',curves)
    write_csv(route/'HPO_SUMMARY.csv',hpo)
    selection = dict(schema_version=1,route='B',status='HPO_FROZEN',seed_ids=SEEDS,
        manifest_hash=prepared['identity']['manifest_hash'],source_hash=prepared['identity']['source_hash'],
        input_hashes=dict(curves=sha_file(route/'VALIDATION_CURVES.csv'),hpo=sha_file(route/'HPO_SUMMARY.csv')),
        datasets=selected,rule=read_json(prepared['manifest'])['selection'])
    frozen = dict(selection,selection_hash=digest(selection))
    existing = route/'selection_frozen.json'
    if existing.exists() and read_json(existing) != frozen:
        raise ValueError('Previously frozen selection differs')
    write_json(route/'HPO_SELECTION.json',selection)
    write_json(existing,frozen)
    return frozen


def require_frozen(output,path):
    output,path = Path(output),Path(path)
    if not path.exists():
        raise Incomplete('Test access requires frozen hyperparameters and checkpoint hashes')
    frozen = read_json(path)
    unsigned = {k:v for k,v in frozen.items() if k!='selection_hash'}
    prepared = read_json(output/'PREPARED.json')
    if frozen.get('selection_hash') != digest(unsigned) or frozen.get('seed_ids') != SEEDS:
        raise ValueError('Frozen selection checksum or seeds changed')
    if (frozen['source_hash'],frozen['manifest_hash']) != (prepared['identity']['source_hash'],prepared['identity']['manifest_hash']):
        raise ValueError('Frozen selection belongs to another attempt')
    return frozen
