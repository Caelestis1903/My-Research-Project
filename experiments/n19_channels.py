"""Selected/final measurement-channel recovery, with no test access or fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from n19_data import OUT
from n19_objectives import NoiseChannel

def main():
    rows=[]
    for path in sorted((OUT/'runs').glob('noise__*')):
        if not (path/'receipt.json').exists():continue
        c=json.loads((path/'config.json').read_text())
        if c['engineering'] or c['method'] not in ['audit_fixed','ind_channel','axis_channel','nested_ind','nested_axis','joint_axis','profile_axis','profile_nested','profile_joint','audit_joint']:continue
        axis=c['method'] in ['axis_channel','nested_axis','joint_axis','profile_axis','profile_nested','profile_joint']
        tag=c['run_id'].split('__seed')[1].split('__',1)[1] if '__' in c['run_id'].split('__seed')[1] else ''
        channel=NoiseChannel(axis)
        for stage,file in [('selected','selected.pt'),('final','last.pt')]:
            state=torch.load(Path(c['state_directory'])/file,map_location='cpu',weights_only=False)
            channel.load_state_dict(state['channel'])
            with torch.no_grad():a,b=channel.rates()
            for name,estimated in [('a',a.numpy()),('b',b.numpy())]:
                truth=np.array(c['noise_rates'][name])
                for d in range(3):
                    for t in range(2):rows.append(dict(run_id=c['run_id'],method=c['method'],tag=tag,seed=c['seed'],stage=stage,epoch=state['epoch'],domain=d,task=t,rate=name,true=truth[d,t],estimated=float(estimated[d,t]),error=float(estimated[d,t]-truth[d,t]),squared_error=float((estimated[d,t]-truth[d,t])**2),effective_channel_parameters=8 if axis else 12))
    target=OUT/'development_readout';target.mkdir(exist_ok=True)
    if rows:
        frame=pd.DataFrame(rows);frame.to_csv(target/'channel_recovery.csv',index=False)
        summary=frame.groupby(['method','tag','seed','stage']).squared_error.mean().pow(.5).rename('rmse').reset_index();summary.to_csv(target/'channel_rmse.csv',index=False)
        print(summary.groupby(['method','tag','stage']).rmse.agg(['mean','std']).to_string())

if __name__=='__main__':main()
