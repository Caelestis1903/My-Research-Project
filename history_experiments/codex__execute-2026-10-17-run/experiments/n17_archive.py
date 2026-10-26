"""Archive completed-run numerical metadata, keeping large artifacts on their native disk."""
import json
from pathlib import Path
import shutil
import pandas as pd
import n17_common as c

def main():
    records=[]
    for run in sorted((c.OUT/'runs').iterdir()):
        receipt=run/'receipt.json'
        if not receipt.exists():continue
        info=json.loads(receipt.read_text());out=c.OUT/'run_metadata'/run.name;out.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','receipt.json','selection_frozen.json','trajectory.csv','metrics_selection.csv','progress.json'):
            file=run/name
            if file.exists():shutil.copyfile(file,out/name)
        raw={}
        for name in ('batch_trace.csv','quota_weights.csv'):
            path=run/name
            if path.exists():raw[name]=dict(path=str(path.resolve()),sha256=c.sha(path),bytes=path.stat().st_size)
        trace=run/'batch_trace.csv'
        if trace.exists():
            frame=pd.read_csv(trace);exposure=frame.groupby('epoch')[['n','d0','d1','d2']].sum()
            exposure['batches']=frame.groupby('epoch').size();exposure.to_csv(out/'epoch_domain_exposure.csv')
        files={}
        for path in run.iterdir():
            if path.suffix=='.pt' and path.name!='last_resume.pt':files[path.name]=dict(path=str(path.resolve()),sha256=c.sha(path),bytes=path.stat().st_size)
        models={p.name:dict(path=str(p.resolve()),sha256=c.sha(p),bytes=p.stat().st_size) for p in (run/'all_models').glob('*_model.pt')}
        record=dict(run=run.name,receipt=info,artifact_directory=str(run.resolve()),states=files,all_epoch_models=models,local_raw_traces=raw)
        c.dump(out/'local_artifacts.json',record);records.append(record)
    c.dump(c.OUT/'completed_artifact_index.json',dict(status='SNAPSHOT',completed_runs=len(records),runs=records,raw_training_log_uploaded=False))
    print('archived completed run metadata',len(records))

if __name__=='__main__':main()
