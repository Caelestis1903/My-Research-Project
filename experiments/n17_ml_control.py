"""Primary-only manifest and symmetric LR selection for conditional N5-C."""
import argparse
import json
from pathlib import Path
import pandas as pd
import n17_common as c


def task(name,seed,arm,opt,lr,phase):
    return dict(name=name,command=[c.PYTHON,'-u','experiments/n17_ml.py','--seed',str(seed),'--subset-seed','2022','--arm',arm,
        '--optimizer',opt,'--lr',str(lr),'--epochs','100','--phase',phase,'--device','{device}','--run-dir',str(c.OUT/'runs'/name)],
        log=str(c.OUT/'logs'/(name+'.log')))


def prepare():
    gate=json.loads((c.OUT/'N1_diagnostic_gates.json').read_text());assert gate['status']=='FROZEN'
    assert any(x['important_residual'] for x in gate['optimizers'].values()),'N5-C NOT_TRIGGERED'
    opt=gate['N2_optimizer'];grid=[.0003,.001,.003,.01] if opt=='adam' else [.03,.1,.3,1.]
    tasks=[task(f'N5pilot_{opt}_lr{lr}_{arm}_seed{seed}',seed,arm,opt,lr,'N5_pilot') for lr in grid for seed in (2030,2031) for arm in ('SN','SB')]
    c.dump(c.OUT/'N5_pilot_manifest.json',dict(cwd=str(c.ROOT),reservation_mib=1900,tasks=tasks,optimizer=opt,
        subset_seed=2022,gate_sha=c.sha(c.OUT/'N1_diagnostic_gates.json'),source_sha=c.sha(__file__),design_sha=c.sha(c.OUT/'N5_C_DESIGN_ADDENDUM.md')))
    print('N5-C optimizer',opt,'pilot tasks',len(tasks))


def select():
    manifest=json.loads((c.OUT/'N5_pilot_manifest.json').read_text());state=json.loads((c.OUT/'N5_pilot_manifest.status.json').read_text())
    assert all(x['status'] in ('COMPLETE','FAILED') for x in state['tasks'].values()),'N5 pilot still running'
    rows=[]
    for item in manifest['tasks']:
        run=c.OUT/'runs'/item['name'];conf=json.loads((run/'config.json').read_text());receipt=run/'receipt.json'
        complete=state['tasks'][item['name']]['status']=='COMPLETE' and receipt.exists() and json.loads(receipt.read_text())['epochs']==100
        rows.append(dict(name=item['name'],seed=conf['seed'],arm=conf['arm'],optimizer=conf['optimizer'],lr=conf['lr'],complete=complete,
            score=float(pd.read_csv(run/'trajectory.csv').selection_macro_auc.max()) if complete else None))
    tab=pd.DataFrame(rows);tab.to_csv(c.OUT/'N5_pilot_response.csv',index=False);valid=tab[tab.complete]
    common=valid.groupby('lr').score.agg(['count','mean']);common=common[common['count']==4];assert len(common)
    common_lr=float(common.sort_values(['mean','lr'],ascending=[False,True]).index[0]);own={}
    for arm in ('SN','SB'):
        candidate=valid[valid.arm==arm].groupby('lr').score.agg(['count','mean']);candidate=candidate[candidate['count']==2];assert len(candidate)
        own[arm]=float(candidate.sort_values(['mean','lr'],ascending=[False,True]).index[0])
    opt=manifest['optimizer'];tasks=[];usage=[];pairs=[]
    for seed in range(2032,2040):
        pair=dict(optimizer=opt,seed=seed,common_lr=common_lr)
        for arm in ('SN','SB'):
            for lr in sorted({common_lr,own[arm]}):
                name=f'N5confirm_{opt}_lr{lr}_{arm}_seed{seed}';tasks.append(task(name,seed,arm,opt,lr,'N5_confirmation'))
                usage.append(dict(optimizer=opt,seed=seed,arm=arm,lr=lr,name=name,common=lr==common_lr,own=lr==own[arm]))
                if lr==common_lr:pair[arm]=str(c.OUT/'runs'/name)
        pairs.append(pair)
    frozen=dict(status='FROZEN',optimizer=opt,common=common_lr,own=own,subset_seed=2022,tie_rule='smaller LR',uses=usage,pairs=pairs,
        pilot_response_sha=c.sha(c.OUT/'N5_pilot_response.csv'),pilot_manifest_sha=c.sha(c.OUT/'N5_pilot_manifest.json'),source_sha=c.sha(__file__),test_read=False)
    c.dump(c.OUT/'N5_winners_frozen.json',frozen)
    c.dump(c.OUT/'N5_confirmation_manifest.json',dict(cwd=str(c.ROOT),reservation_mib=1900,tasks=tasks,winners_sha=c.sha(c.OUT/'N5_winners_frozen.json')))
    print('N5 common',common_lr,'own',own,'confirmation tasks',len(tasks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','select']);a=p.parse_args();prepare() if a.action=='prepare' else select()
