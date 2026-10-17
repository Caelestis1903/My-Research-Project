"""Primary-orchestrator numerical readout of the already frozen N1 selection rules."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import n17_common as c
from v3_train_20261016 import metrics

def paired(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n)) if n>1 else None
    ci=[mean-t.ppf(.975,n-1)*se,mean+t.ppf(.975,n-1)*se] if n>1 else [None,None]
    signs=np.asarray(list(itertools.product((-1,1),repeat=n)));null=np.abs((signs*x).mean(1));p=float(np.mean(null>=abs(mean)-1e-15))
    rng=np.random.default_rng(20261017);boot=x[rng.integers(0,n,(20000,n))].mean(1)
    return dict(n=n,mean=mean,se=se,ci95=[None if v is None else float(v) for v in ci],positive=int(sum(x>0)),negative=int(sum(x<0)),zero=int(sum(x==0)),exact_sign_randomization_p=p,seed_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),raw=x.tolist(),scope='paired independent training seeds; t approximation; sign exchangeability under null')

def command(name,seed,arm,opt,lr,phase='N1_confirmation',epochs=100):
    return dict(name=name,command=[c.PYTHON,'-u','experiments/n17_capture.py','--seed',str(seed),'--arm',arm,'--optimizer',opt,'--lr',str(lr),'--epochs',str(epochs),'--phase',phase,'--device','{device}','--run-dir',str(c.OUT/'runs'/name)],log=str(c.OUT/'logs'/(name+'.log')))

def select():
    manifest=json.loads((c.OUT/'N1_pilot_manifest.json').read_text());status=json.loads((c.OUT/'N1_pilot_manifest.status.json').read_text())
    assert all(v['status'] in ('COMPLETE','FAILED') for v in status['tasks'].values()),'pilot still running'
    rows=[]
    for task in manifest['tasks']:
        run=c.OUT/'runs'/task['name'];config=json.loads((run/'config.json').read_text());receipt=run/'receipt.json'
        complete=receipt.exists() and json.loads(receipt.read_text())['epochs']==100 and status['tasks'][task['name']]['status']=='COMPLETE'
        score=float(pd.read_csv(run/'trajectory.csv').selection_macro_auc.max()) if complete else None
        rows.append(dict(name=task['name'],optimizer=config['optimizer'],lr=config['lr'],arm=config['arm'],seed=config['seed'],complete=complete,score=score))
    tab=pd.DataFrame(rows);tab.to_csv(c.OUT/'N1_pilot_response.csv',index=False);winners={};tasks=[];pairs=[];usage=[]
    for opt,part in tab.groupby('optimizer'):
        valid=part[part.complete];common=valid.groupby('lr').score.agg(['count','mean']);common=common[common['count']==4];assert len(common),'no complete common-eta configuration'
        common_lr=float(common.sort_values(['mean','lr'],ascending=[False,True]).index[0]);own={}
        for arm in ('SN','SB'):
            candidate=valid[valid.arm==arm].groupby('lr').score.agg(['count','mean']);candidate=candidate[candidate['count']==2];assert len(candidate)
            own[arm]=float(candidate.sort_values(['mean','lr'],ascending=[False,True]).index[0])
        winners[opt]=dict(common=common_lr,own=own)
        for seed in range(2032,2040):
            pair=dict(optimizer=opt,seed=seed,common_lr=common_lr)
            for arm in ('SN','SB'):
                for lr in sorted({common_lr,own[arm]}):
                    name=f'N1confirm_{opt}_lr{lr}_{arm}_seed{seed}';tasks.append(command(name,seed,arm,opt,lr))
                    usage.append(dict(optimizer=opt,seed=seed,arm=arm,lr=lr,name=name,common=(lr==common_lr),own=(lr==own[arm])))
                    if lr==common_lr:pair[arm]=str(c.OUT/'runs'/name)
            pairs.append(pair)
    assert len({x['name'] for x in tasks})==len(tasks)
    freeze=dict(status='FROZEN',winners=winners,pilot_manifest_sha=c.sha(c.OUT/'N1_pilot_manifest.json'),source_sha=c.sha(__file__),response_sha=c.sha(c.OUT/'N1_pilot_response.csv'),test_access=False,tie_rule='smaller LR',task_count=len(tasks),uses=usage,pairs=pairs)
    c.dump(c.OUT/'N1_winners_frozen.json',freeze);c.dump(c.OUT/'N1_confirmation_manifest.json',dict(cwd=str(c.ROOT),reservation_mib=1900,tasks=tasks,winners_sha=c.sha(c.OUT/'N1_winners_frozen.json')))
    print(json.dumps(winners,indent=2));print('confirmation runs',len(tasks))

def gates():
    freeze=json.loads((c.OUT/'N1_winners_frozen.json').read_text());b=c.bundle();sp=c.split();records=[];rho=[];pairchecks=[]
    for pair in freeze['pairs']:
        configs=[]
        for arm in ('SN','SB'):
            run=Path(pair[arm]);receipt=json.loads((run/'receipt.json').read_text());assert receipt['epochs']>=100
            conf=json.loads((run/'config.json').read_text());configs.append(conf);traj=pd.read_csv(run/'trajectory.csv').set_index('epoch');assert all(e in traj.index for e in range(1,101))
            selected=int(traj.loc[1:100].selection_macro_auc.idxmax());pred=np.load(run/f'validation_epoch{selected:03d}.npz')['predictions'];cells=metrics(b,pred,'val',sp['diagnostic'])
            for row in cells:records.append(dict(optimizer=pair['optimizer'],seed=pair['seed'],arm=arm,epoch=selected,**row))
            a=float(traj.loc[81:90].train_empirical_objective.mean());z=float(traj.loc[91:100].train_empirical_objective.mean());rr=(a-z)/max(abs(a),1e-12)
            rho.append(dict(optimizer=pair['optimizer'],seed=pair['seed'],arm=arm,rho=rr,complete=True,finite=bool(np.isfinite(traj.loc[1:100].train_empirical_objective).all())))
        assert configs[0]['init_sha256']==configs[1]['init_sha256'] and configs[0]['data']['split_row_id_hashes']==configs[1]['data']['split_row_id_hashes']
        pairchecks.append(dict(optimizer=pair['optimizer'],seed=pair['seed'],paired=True,init_sha256=configs[0]['init_sha256']))
    tab=pd.DataFrame(records);tab.to_csv(c.OUT/'N1_diagnostic_selected_cells.csv',index=False);pd.DataFrame(rho).to_csv(c.OUT/'N1_training_rho.csv',index=False);results={};differences=[]
    for opt,part in tab.groupby('optimizer'):
        wide=part.pivot(index=['seed','domain','task'],columns='arm',values='auc');wide['difference']=wide.SB-wide.SN
        endpoints={'macro':wide.groupby('seed').difference.mean()}
        endpoints.update({f'd{d}t{tt}':v.difference.droplevel(['domain','task']) for (d,tt),v in wide.groupby(['domain','task'])})
        summary={name:paired(values.sort_index().to_numpy()) for name,values in endpoints.items()}
        important=any(abs(v['mean'])>=.003 and (v['positive'] if v['mean']>0 else v['negative'])>=6 for v in summary.values())
        rr=[x for x in rho if x['optimizer']==opt];extension=any(np.mean([x['rho'] for x in rr if x['arm']==arm])>.005 and sum(x['rho']>0 for x in rr if x['arm']==arm)>=6 for arm in ('SN','SB')) and len(rr)==16 and all(x['finite'] for x in rr)
        results[opt]=dict(important_residual=important,diagnostic_summary=summary,extend_H200=bool(extension),training_rho=rr)
        for name,values in endpoints.items():
            for seed,value in values.items():differences.append(dict(optimizer=opt,endpoint=name,seed=int(seed),difference=float(value)))
    selected='adam' if results['adam']['important_residual'] else ('sgd' if results['sgd']['important_residual'] else 'adam')
    quota=results[selected]['important_residual']
    c.dump(c.OUT/'N1_diagnostic_gates.json',dict(status='FROZEN',optimizers=results,N2_optimizer=selected,N2_quota_training=quota,N3_other_SGD_whole_C=all(x['important_residual'] for x in results.values()),test_read=False,source_sha=c.sha(__file__),scope='preregistered resource triggers, not confirmed mechanism significance',pair_checks=pairchecks))
    pd.DataFrame(differences).to_csv(c.OUT/'N1_diagnostic_paired_seed_differences.csv',index=False)
    print('N2/N3 optimizer',selected,'quota',quota,'extensions',{k:v['extend_H200'] for k,v in results.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','gates']);a=p.parse_args();select() if a.action=='select' else gates()
