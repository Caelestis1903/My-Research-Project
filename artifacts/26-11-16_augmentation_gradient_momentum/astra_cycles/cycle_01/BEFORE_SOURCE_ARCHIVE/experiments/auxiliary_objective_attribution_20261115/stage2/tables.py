"""Stage 2 raw scalar exports and paired arithmetic."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .protocol import SEEDS, DATASETS, results, epoch_files
ARMS = ('NDP-A',)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def paired(values):
    values = np.asarray(values,dtype=float)
    assert values.shape == (5,) and np.isfinite(values).all()
    mean = float(values.mean()); sd = float(values.std(ddof=1)); radius = 2.776445*sd/np.sqrt(5)
    return dict(mean=mean,sample_sd=sd,ci95_low=mean-radius,ci95_high=mean+radius,
                positive_seeds=int((values > 0).sum()))


def export(attempt):
    attempt = Path(attempt); records = results(attempt)
    assert len(records) == 20 and all(r['epochs'] == 50 and not r['test_accessed'] for r in records)
    by_key = {(r['job']['dataset'],r['job']['config']['arm'],r['job']['seed']):r for r in records}
    assert len(by_key) == 20
    assert set(by_key) == {(d,a,s) for d in DATASETS for a in ('CLEAN',)+ARMS for s in SEEDS}
    endpoints = []; curves = []; costs = []; weights = []; order_hashes = {}
    for result in records:
        j = result['job']; arm = j['config']['arm']
        base = dict(dataset=j['dataset'],arm=arm,seed=j['seed'],key=j['key'])
        epochs = epoch_files(attempt,result); assert set(epochs) == set(range(1,51))
        expected = 1000 if j['dataset'] == 'movielens' else 2450
        assert result['global_step'] == result['counts']['optimizer_updates'] == result['counts']['ema_updates'] == expected
        for epoch,(_,data) in sorted(epochs.items()):
            k = (j['dataset'],j['seed'],epoch)
            if k in order_hashes:
                assert order_hashes[k] == data['epoch_order_hash'], 'Natural loader order mismatch'
            order_hashes[k] = data['epoch_order_hash']
            for kind,values in data['validation'].items():
                curves.append(dict(**base,kind=kind,epoch=epoch,**values))
            for cell,value in data.get('objective_ledger',{}).get('cells',{}).items():
                weights.append(dict(**base,epoch=epoch,cell_label=cell,
                    batch_count=data['objective_ledger']['batches'],**value))
        for kind,values in result['selected_validation'].items():
            selected_epoch = max(epochs,key=lambda e:epochs[e][1]['validation'][kind]['macro_auc'])
            assert selected_epoch == result['selected_epochs'][kind]
            data = epochs[selected_epoch][1]
            assert data['validation'][kind] == values
            assert data['model_hash' if kind == 'online' else 'ema_hash'] == result['selected_hashes'][kind]
            endpoints.append(dict(**base,split='validation',endpoint='selected',kind=kind,
                epoch=selected_epoch,model_hash=result['selected_hashes'][kind],**values))
        for epoch in (10,30,50):
            for kind,values in epochs[epoch][1]['validation'].items():
                endpoints.append(dict(**base,split='validation',endpoint=f'E{epoch}',kind=kind,epoch=epoch,**values))
        costs.append(dict(**base,epochs=50,steps=result['global_step'],worker_seconds=result['worker_seconds'],
            peak_allocated_bytes=result['peak_allocated_bytes'],peak_reserved_bytes=result['peak_reserved_bytes'],
            data_load_seconds=result.get('data_load_seconds'),reused=arm == 'CLEAN'))
    groups = {}
    for row in endpoints:
        groups.setdefault(tuple(row[k] for k in ('dataset','arm','endpoint','kind')),{})[row['seed']] = row
    metrics = sorted({k for row in endpoints for k in row if k.endswith(('_auc','_logloss'))})
    summaries = []; seed_differences = []; differences = []
    for key,values in sorted(groups.items()):
        assert set(values) == set(SEEDS)
        base = dict(zip(('dataset','arm','endpoint','kind'),key))
        for metric in metrics:
            v = [values[s][metric] for s in SEEDS]
            if None not in v:
                summaries.append(dict(**base,metric=metric,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),n=5))
    contrasts = [('NDP-A','CLEAN')]
    for dataset in DATASETS:
        for endpoint in ('selected','E10','E30','E50'):
            for arm,control in contrasts:
                kinds = [('online','online'),('fixed_ema','fixed_ema')]
                if control == 'CLEAN':
                    kinds.append(('fixed_ema','online'))
                for kind,reference_kind in kinds:
                    a = groups[dataset,arm,endpoint,kind]; b = groups[dataset,control,endpoint,reference_kind]
                    base = dict(dataset=dataset,arm=arm,control=control,endpoint=endpoint,kind=kind,reference_kind=reference_kind)
                    for metric in metrics:
                        if any(a[s][metric] is None or b[s][metric] is None for s in SEEDS):
                            continue
                        deltas = [a[s][metric]-b[s][metric] for s in SEEDS]
                        differences.append(dict(**base,metric=metric,**paired(deltas)))
                        seed_differences.extend(dict(**base,metric=metric,seed=s,delta=v) for s,v in zip(SEEDS,deltas))
    directory = attempt/'tables'
    for name,rows in (('ENDPOINTS',endpoints),('VALIDATION_CURVES',curves),('COSTS',costs),('WEIGHTS',weights),
                     ('SUMMARY',summaries),('PAIRED_SEEDS',seed_differences),('PAIRED_SUMMARY',differences)):
        write_csv(directory/f'{name}.csv',rows)
    return dict(status='COMPLETE',trajectories=20,endpoints=len(endpoints),directory=str(directory),test_accessed=False)
