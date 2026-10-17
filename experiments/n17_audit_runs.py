"""Minimal completed-run artifact validation, without selecting or ranking models."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import n17_common as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--budget',type=int,default=100)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-running',action='store_true');a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());state=json.loads(a.manifest.with_suffix('.status.json').read_text())
    rows=[];checks=[];initializations={};populations={}
    for task in manifest['tasks']:
        name=task['name'];status=state['tasks'][name]['status']
        if status!='COMPLETE':
            assert a.allow_running or status=='FAILED','nonterminal tasks remain'
            rows.append(dict(name=name,status=status));continue
        args=task['command'];run=Path(args[args.index('--run-dir')+1]);conf=json.loads((run/'config.json').read_text())
        receipt_file=run/'receipt.json'
        if a.budget==100 and (run/'H100_receipt.json').exists():receipt_file=run/'H100_receipt.json'
        receipt=json.loads(receipt_file.read_text());assert receipt['epochs']>=a.budget
        n=sum(conf['data']['train_domain_counts'].values());updates=math.ceil(n/conf['batch_size'])
        traj=pd.read_csv(run/'trajectory.csv');traj=traj[traj.epoch<=a.budget];assert traj.epoch.tolist()==list(range(1,a.budget+1))
        assert np.array_equal(traj.steps.to_numpy(),np.arange(1,a.budget+1)*updates)
        assert np.array_equal(traj.examples.to_numpy(),np.arange(1,a.budget+1)*n)
        assert np.isfinite(traj[['train_empirical_objective','selection_macro_auc']].to_numpy()).all()
        selected=int(traj.loc[traj.selection_macro_auc.idxmax(),'epoch'])
        if receipt['epochs']==a.budget:assert selected==receipt['selected_epoch'] and receipt['steps']==updates*a.budget and receipt['examples']==n*a.budget
        metric=pd.read_csv(run/'metrics_selection.csv');metric=metric[metric.epoch<=a.budget]
        assert len(metric)==6*a.budget and not metric.duplicated(['epoch','domain','task']).any()
        assert np.isfinite(metric[['auc','logloss']].to_numpy()).all()
        trace=pd.read_csv(run/'batch_trace.csv');trace=trace[trace.epoch<=a.budget]
        assert len(trace)==updates*a.budget and np.array_equal(trace.step,np.arange(1,updates*a.budget+1))
        assert (trace[['d0','d1','d2']].sum(axis=1)==trace.n).all()
        assert (trace.groupby('epoch').n.sum()==n).all()
        expected=np.tile([conf['batch_size']]*(updates-1)+[n-conf['batch_size']*(updates-1)],a.budget)
        assert np.array_equal(trace.n.to_numpy(),expected)
        assert not conf['test_access'] and not conf['engineering']
        if conf.get('all_epoch_models_retained'):assert all((run/'all_models'/f'epoch{e:03d}_model.pt').exists() for e in range(1,a.budget+1))
        # Initialization is independent of q and eta, conditional on environment,
        # population, topology and optimizer seed. The complete source hashes
        # distinguish implementation versions even when commits differ by docs.
        key=(conf['data']['dataset_key'],conf.get('subset_seed',conf.get('population_seed',2022)),conf['seed'],conf.get('topology','original'),conf.get('selected_block','none'))
        if key in initializations:assert initializations[key]==conf['init_sha256']
        initializations[key]=conf['init_sha256']
        population_key=key[:2];identity=conf['data']['split_row_id_hashes']
        if population_key in populations:assert populations[population_key]==identity
        populations[population_key]=identity
        checks.append(dict(name=name,config_sha=c.sha(run/'config.json'),receipt_sha=c.sha(receipt_file),trajectory_sha=c.sha(run/'trajectory.csv'),batch_trace_sha=c.sha(run/'batch_trace.csv'),source_hashes=conf['source_hashes']))
        rows.append(dict(name=name,status='VALIDATED_COMPLETE',budget=a.budget,steps=updates*a.budget,examples=n*a.budget,selected_epoch=selected,actual_epoch_updates=updates,actual_tail=int(expected[-1])))
    c.dump(a.out,dict(status='PARTIAL_VALIDATION' if any(x['status'] not in ('VALIDATED_COMPLETE','FAILED') for x in rows) else 'COMPLETE_VALIDATION',
        manifest_sha=c.sha(a.manifest),source_sha=c.sha(__file__),runs=rows,checks=checks,failed=[x['name'] for x in rows if x['status']=='FAILED'],
        scope='completed raw artifact clocks, finite recorded metrics, earliest selection, q/LR initialization and fixed population identity; no LR ranking or test prediction'))
    print('validated complete',len(checks),'of',len(rows),'failed',sum(x['status']=='FAILED' for x in rows))


if __name__=='__main__':main()
