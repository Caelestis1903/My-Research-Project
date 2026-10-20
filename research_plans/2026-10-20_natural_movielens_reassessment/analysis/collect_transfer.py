"""Collect historical and frozen-test transfer cells with explicit protocol identity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss

ART=Path('/home_new/xzf_natural_movielens_reassessment')
REPORT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'transfer'
OLD={'N19':Path('/home/xzf/Code/phase0-3-n19-git/research_plans/2026-10-18/N19/runs'),
     'CM19':Path('/home/xzf/Code/phase0-3-20261019-git/research_plans/2026-10-19/execution/runs')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def collect(historical_only=False):
    OUT.mkdir(exist_ok=True);cells=[];source=[];configs=[];audit=[];missing=[]
    paths=[]
    for study,root in OLD.items():
        for run in sorted(root.iterdir()):
            if not (run/'config.json').exists():continue
            config=json.loads((run/'config.json').read_text())
            if config.get('engineering') or config.get('smoke'):continue
            population='kuairand_natural_historical' if config['dataset']=='kuairand' else 'movielens_artificial_historical'
            paths.append((study,population,'historical',run,config,None))
    if not historical_only:
        for cfg in json.loads((ART/'transfer_manifest.json').read_text())['runs']:
            run=Path(cfg['run_dir'])
            if not (run/'reassessment_test_receipt.json').exists():
                missing.append(cfg['run_id']);continue
            paths.append((cfg['study'],cfg['population'],cfg['scope'],run,json.loads((run/'config.json').read_text()),cfg))
    for study,population,scope,run,cfg,new in paths:
        path=run/('metrics_test.json' if study=='N19' else 'metrics_test.csv')
        if not path.exists():missing.append(str(path));continue
        f=pd.DataFrame(json.loads(path.read_text())) if study=='N19' else pd.read_csv(path)
        if study=='N19':f=f.rename(columns={'stage':'checkpoint'});f['intervention']='none'
        assert not f.duplicated(['checkpoint','domain','task','intervention']).any(),str(path)
        assert set(f.checkpoint)=={'selected','final'}
        run_id=new['run_id'] if new else study+'__historical__'+run.name
        base=dict(study=study,population=population,scope=scope,run_id=run_id,historical_run_id=run.name,
            dataset=cfg['dataset'],seed=cfg['seed'],optimizer=cfg['optimizer'].lower(),
            model=cfg.get('model',cfg.get('method')),method=cfg.get('method',cfg.get('model')),
            suite=cfg.get('suite','natural'),variant=cfg.get('variant','native'),
            decay_mode=cfg.get('decay_mode','standard'),schedule=cfg.get('schedule','normal_epochs'),
            fraction=cfg.get('fraction',1.),backbone=cfg.get('backbone','m3oe'),
            noise_interaction=cfg.get('noise_interaction',0.),
            stage_group=cfg.get('stage',cfg.get('suite','natural')),
            budget=new['budget'] if new else 'historical_original',lr=cfg['lr'],
            effective_main_support=cfg['train_rows'],original_main_support=cfg.get('original_train_rows',cfg['train_rows']),
            source_level='new_predictions_recalculated' if new else 'historical_metrics_and_config; heavy tensors not claimed available')
        for key,value in base.items():f[key]=value
        counts=np.asarray(cfg['main_domain_counts'],dtype=float) if study=='CM19' else np.asarray(cfg['p'])*cfg['train_rows']
        support_p=counts/counts.sum()
        if study=='CM19' and cfg.get('schedule')=='matched_updates':
            if new:
                dm=cfg['data_population_metadata'];n=dm['main_rows']
                original_counts=np.asarray(cfg['support_main_domain_counts'],dtype=float)
            else:original_counts=np.array([164143,1656704,84980]) if cfg['dataset']=='kuairand' else np.array([32775,2850,142500])
            exposure_p=original_counts/original_counts.sum()
        else:exposure_p=support_p
        tasks=sorted(f.task.unique());n_tasks=len(tasks)
        f['support_p_cell_weight']=f.domain.map(dict(enumerate(support_p/n_tasks)))
        f['exposure_p_cell_weight']=f.domain.map(dict(enumerate(exposure_p/n_tasks)))
        if new:
            for checkpoint,g in f.groupby('checkpoint'):
                prediction=Path(cfg['state_directory'])/f'test_{checkpoint}.npz' if study=='CM19' else run/f'test_{checkpoint}.npz'
                with np.load(prediction) as z:
                    y=z['labels'];domains=z['domains'];mapping=cfg.get('task_ids',[0,1])
                    for row in g.itertuples():
                        ix=domains==row.domain;key='predictions' if row.intervention=='none' else 'neutral_'+row.intervention
                        p=z[key][ix,mapping.index(row.task)];yy=y[ix,row.task]
                        auc=roc_auc_score(yy,p);ll=log_loss(yy,p,labels=[0,1])
                        assert abs(auc-row.auc)<1e-12 and abs(ll-row.logloss)<1e-10
                        audit.append(dict(run_id=run_id,checkpoint=checkpoint,intervention=row.intervention,
                            domain=row.domain,task=row.task,auc_residual=auc-row.auc,logloss_residual=ll-row.logloss,
                            prediction_sha256=sha(prediction)))
        cells.append(f);configs.append(dict(base,config_path=str(run/'config.json'),config_sha256=sha(run/'config.json'),
            native_tasks=json.dumps([int(t) for t in tasks]),parameters=cfg['parameters']))
        source.append(dict(run_id=run_id,metrics_path=str(path),metrics_sha256=sha(path),
            config_path=str(run/'config.json'),config_sha256=sha(run/'config.json')))
    allcells=pd.concat(cells,ignore_index=True);allcells.to_csv(OUT/'all_cell_metrics.csv',index=False)
    pd.DataFrame(configs).to_csv(OUT/'run_config_index.csv',index=False)
    pd.DataFrame(source).to_csv(OUT/'source_index.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'new_prediction_recalculation.csv',index=False)
    rows=[]
    for (run_id,checkpoint,intervention),g in allcells.groupby(['run_id','checkpoint','intervention']):
        meta=next(c for c in configs if c['run_id']==run_id)
        for task in [-1,*sorted(g.task.unique())]:
            gg=g if task==-1 else g[g.task==task]
            rows.append(dict(meta,checkpoint=checkpoint,intervention=intervention,task=task,
                cells=len(gg),epoch=int(gg.epoch.iloc[0]),
                macro_auc=gg.auc.mean(),macro_logloss=gg.logloss.mean(),
                support_p_cell_auc=np.average(gg.auc,weights=gg.support_p_cell_weight),
                support_p_cell_logloss=np.average(gg.logloss,weights=gg.support_p_cell_weight),
                exposure_p_cell_auc=np.average(gg.auc,weights=gg.exposure_p_cell_weight),
                exposure_p_cell_logloss=np.average(gg.logloss,weights=gg.exposure_p_cell_weight),
                eval_p_cell_auc=np.average(gg.auc,weights=gg.n),eval_p_cell_logloss=np.average(gg.logloss,weights=gg.n)))
    pd.DataFrame(rows).to_csv(OUT/'run_task_macro_metrics.csv',index=False)
    (OUT/'collection_status.json').write_text(json.dumps(dict(historical_only=historical_only,run_count=len(configs),
        historical_run_count=sum(c['scope']=='historical' for c in configs),new_run_count=sum(c['scope']!='historical' for c in configs),
        cell_metric_rows=len(allcells),new_prediction_cell_recalculations=len(audit),missing=missing,
        metric_protocol='original sklearn roc_auc_score/log_loss with original saved dtype; differs from EPS1e-12 core protocol',
        task_macro_scope='task=-1 averages only tasks actually emitted by this model; single-task STAR is not a joint model'),indent=2)+'\n')
    print('Collected',len(configs),'runs,',len(allcells),'cell rows; new',sum(c['scope']!='historical' for c in configs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--historical-only',action='store_true')
    collect(p.parse_args().historical_only)
