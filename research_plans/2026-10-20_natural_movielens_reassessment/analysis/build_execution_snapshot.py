"""Join every designed configuration to the attempt ledger without counting retries as seeds."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import math

REPORT=Path(__file__).resolve().parents[1]
ART=Path('/home_new/xzf_natural_movielens_reassessment')

def build():
    rows=[];input_index=[]
    def read(name):
        p=ART/name
        input_index.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        return json.loads(p.read_text())
    def add(study,cfg,requirement='BASE_REQUIRED'):
        population=cfg.get('population','movielens_natural')
        row=dict(study=study,stage=cfg.get('stage',cfg.get('group',study)),scope=cfg.get('scope',cfg.get('group','')),
            logical_run_id=cfg.get('run_id',study+'__'+cfg.get('id','')),population=population,
            seed=cfg['seed'],pilot=cfg.get('group')=='pilot',arm=cfg.get('arm',''),
            optimizer=cfg.get('optimizer',''),lr=cfg.get('lr',''),budget=cfg.get('budget',''),
            epochs=cfg.get('epochs',100),run_dir=cfg.get('run_dir',''),config_path=cfg.get('config_path',''),
            requirement=requirement,state='QUEUED',attempts=0,latest_attempt_id='',completed_epoch='',
            actual_main_steps='',actual_main_examples='',receipt='',source_manifest=input_index[-1]['path'])
        if not row['run_dir']:row['state']='BLOCKED_BY_PILOT_COMMON_LR'
        rows.append(row)
    for cfg in read('extensions_manifest.json')['configurations']:add('V2',cfg)
    for cfg in read('core_manifest.json')['runs']:add('core',cfg)
    for cfg in read('transfer_manifest.json')['runs']:add(cfg['study'],cfg)
    for cfg in read('AO04/manifest.json')['configurations']:add('AO04',cfg)
    for cfg in read('N17/base_manifest.json')['configurations']:add('N17',cfg)
    assert len(rows)==1044
    for name in ('N17/post_pilot_manifest.json','N17/H200_manifest.json'):
        if not (ART/name).exists():continue
        for cfg in read(name)['configurations']:
            if cfg['group']=='quota':
                old=next(r for r in rows if r['study']=='N17' and r['stage']=='quota' and r['arm']==cfg['arm'] and r['seed']==cfg['seed'])
                rows.remove(old);add('N17',cfg)
            elif not any(r['run_dir']==cfg['run_dir'] for r in rows):add('N17',cfg,'CONDITIONAL_TRIGGERED')
    ledger=REPORT/'RUN_LEDGER.csv';attempts=list(csv.DictReader(ledger.open()))
    input_index.append(dict(path=str(ledger),sha256=hashlib.sha256(ledger.read_bytes()).hexdigest()))
    for row in rows:
        if not row['run_dir']:continue
        found=[a for a in attempts if a.get('output_dir')==row['run_dir'] and a.get('execution_kind') in ('','training')]
        row['attempts']=len(found)
        if found:
            complete=[a for a in found if a['status']=='COMPLETE']
            last=(complete or found)[-1]
            row.update(state=last['status'],latest_attempt_id=last['attempt_id'],
                completed_epoch=last['epochs_completed'],actual_main_steps=last['main_steps'],actual_main_examples=last['examples'],receipt=last['receipt'])
    with (REPORT/'EXPERIMENT_EXECUTION_SNAPSHOT.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_configurations=1044,
        conditional_triggered_configurations=sum(r['requirement']=='CONDITIONAL_TRIGGERED' for r in rows),
        counts_by_study_and_state={str(k):n for k,n in collections.Counter((r['study'],r['state']) for r in rows).items()},
        logical_completed=sum(r['state']=='COMPLETE' for r in rows),
        attempt_failures_retained=sum(a['status']=='FAILED' for a in attempts),
        note='Snapshot, not replacement for live RUN_LEDGER; quota templates are required, pilots are not formal seeds; conditional H200 retains H100.',inputs=input_index)
    (REPORT/'EXPERIMENT_EXECUTION_SNAPSHOT.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print({k:v for k,v in summary.items() if k!='inputs'})

if __name__=='__main__':build()
