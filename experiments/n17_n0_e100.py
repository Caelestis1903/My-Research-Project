"""Qualify existing E100 continuation readouts without double-counting prefixes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import n17_common as c

def main():
    out=c.OUT/'N0';manifest=json.loads((out/'readout_manifest.json').read_text());ledger=pd.read_csv(out/'readout_ledger.csv');new=[];cells=[];checks=[]
    for item in manifest['runs']:
        group,seed=item['group'],item['seed']
        if ((ledger.group==group)&(ledger.seed==seed)&(ledger.readout=='common_epoch100')).any():continue
        arms=[];sources=[]
        for name in item['source_runs']:
            run=Path(name);pr=run/'test_prediction_reuse_provenance.json'
            parent=Path(json.loads(pr.read_text())['parent']) if pr.exists() else run
            metrics=parent/'metrics_test.csv'
            if not metrics.exists():break
            part=pd.read_csv(metrics);part=part[part.epoch==100]
            if len(part)!=6:break
            old=json.loads((run/'config.json').read_text());pc=json.loads((parent/'config.json').read_text())
            keys=['seed','arm','optimizer','lr','p','q','batch_size','source_hashes','init_sha256']
            assert all(old[k]==pc[k] for k in keys),'continuation identity changed'
            arms.append(part.sort_values(['domain','task']));sources.append(dict(run=str(run),parent=str(parent),metrics_sha=c.sha(metrics),prefix_provenance_sha=c.sha(pr) if pr.exists() else None))
        if len(arms)!=2:continue
        p=np.asarray(item['train_weights']);r=np.asarray(item['evaluation_weights']);ds=arms[0].domain.to_numpy(dtype=int)
        for metric in ('auc','logloss'):
            for name,w in [('uniform',np.ones(3)/3),('train_p',p),('evaluation_p',r)]:
                sn,sb=[float(np.dot(part[metric],w[ds]/2)) for part in arms]
                new.append(dict(group=group,seed=seed,readout='common_epoch100',metric=metric,weights=name,SN=sn,SB=sb,difference=sb-sn,SN_epoch=100,SB_epoch=100,evaluation_population='test',selection_population='none; fixed clock',actual_seed_unit=True))
            for d,t in arms[0][['domain','task']].itertuples(index=False,name=None):
                sn,sb=[float(part[(part.domain==d)&(part.task==t)][metric].iloc[0]) for part in arms]
                cells.append(dict(group=group,seed=seed,domain=d,task=t,metric=metric,SN=sn,SB=sb,difference=sb-sn))
        checks.append(dict(group=group,seed=seed,status='QUALIFIED_EXISTING_E100',sources=sources,independent_seed_increment=0))
    combined=pd.concat([ledger,pd.DataFrame(new)],ignore_index=True);assert not combined.duplicated(['group','seed','readout','metric','weights']).any()
    combined.to_csv(out/'readout_ledger.csv',index=False);combined.groupby(['group','readout','metric','weights']).agg(n=('seed','nunique'),difference=('difference','mean')).reset_index().to_csv(out/'seed_means.csv',index=False)
    pd.DataFrame(cells).to_csv(out/'additional_E100_cells.csv',index=False);c.dump(out/'E100_qualification.json',dict(status='COMPLETE',qualifications=checks,new_training=0,source_sha=c.sha(__file__),scope='qualified actual same-configuration E100 continuations; no extra independent seed for a prefix'))

if __name__=='__main__':main()
