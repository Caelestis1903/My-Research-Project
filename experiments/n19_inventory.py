"""Research delivery ledger, including unfinished work; never assumes completion."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
from n19_data import ROOT,OUT,dump

def main(final=False):
    manifest=json.loads((OUT/'queue_manifest.json').read_text());rows=[];arrays=[];missing=[]
    for task in manifest['tasks']:
        cmd=task['command']
        if 'experiments/n19_train.py' not in cmd:continue
        engineering='--engineering' in cmd;path=OUT/'runs'/task['name']
        c=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
        receipt=json.loads((path/'receipt.json').read_text()) if (path/'receipt.json').exists() else {}
        row=dict(run_id=task['name'],engineering=engineering,suite=c.get('suite'),dataset=c.get('dataset'),method=c.get('method'),seed=c.get('seed'),
            status=receipt.get('status','NOT_COMPLETE'),epochs_budget=c.get('epochs'),epochs_completed=receipt.get('epochs'),test_complete=(path/'test_receipt.json').exists(),state_directory=c.get('state_directory'))
        rows.append(row)
        if not engineering and (row['status']!='COMPLETE_TRAINING_TEST_DEFERRED' or row['epochs_budget']!=row['epochs_completed']):missing.append(task['name'])
        if final:
            for file in sorted(path.glob('*.npz')):
                arrays.append(dict(run_id=task['name'],kind=file.stem,local_path=str(file),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),git_archived=False))
    registered={r['run_id'] for r in rows}
    for path in sorted((OUT/'runs').glob('*')):
        if path.name in registered or not (path/'config.json').exists():continue
        c=json.loads((path/'config.json').read_text());receipt=json.loads((path/'receipt.json').read_text()) if (path/'receipt.json').exists() else {}
        if not c['engineering']:
            missing.append('UNREGISTERED_FORMAL:'+path.name);continue
        rows.append(dict(run_id=path.name,engineering=True,suite=c['suite'],dataset=c['dataset'],method=c['method'],seed=c['seed'],
            status=receipt.get('status','NOT_COMPLETE'),epochs_budget=c['epochs'],epochs_completed=receipt.get('epochs'),test_complete=False,state_directory=c['state_directory']))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'RUN_LEDGER.csv',index=False)
    formal=frame[~frame.engineering];status=dict(status='TRAINING_INCOMPLETE' if missing else 'TRAINING_COMPLETE_ANALYSIS_PENDING',
        formal_tasks=len(formal),formal_completed=int((formal.status=='COMPLETE_TRAINING_TEST_DEFERRED').sum()),engineering_tasks=int(frame.engineering.sum()),
        unfinished=missing,test_completed=int(formal.test_complete.sum()),all_directions_complete=False)
    if final:
        pd.DataFrame(arrays).to_csv(OUT/'LOCAL_PREDICTION_INDEX.csv',index=False)
        status['local_prediction_arrays']=len(arrays)
    dump(OUT/'INVENTORY_STATUS.json',status);print(json.dumps(status))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
