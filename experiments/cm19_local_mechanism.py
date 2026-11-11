"""Post-training common-state local optimizer counterfactuals; raw data only."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cm19_train import OUT,append,dump
from cm19_legacy_data import population
from cm19_models import make_model


def local(run_id,device='cpu',labels=('step000000','step000001','step000005','step000010','step000025','step000050','step000100','step000200','step000500','selected','final')):
    out=OUT/'runs'/run_id;cfg=json.loads((out/'config.json').read_text());art=Path(cfg['state_directory']);b=population(cfg['dataset'],'natural')
    torch.set_num_threads(2);m=make_model(b.features,cfg['seed'],cfg['model'],cfg['variant']).to(device);named=dict(m.named_parameters());records=[];moment_rows=[]
    for label in labels:
        file=art/(label+'.pt')
        if not file.exists():continue
        state=torch.load(file,map_location=device,weights_only=False);m.load_state_dict(state['model']);m.eval();m.zero_grad(set_to_none=True)
        with np.load(art/f'probe_{label}_meta.npz') as data:ix=data['row_indices'].copy()
        x=torch.tensor(b.train_x[ix],device=device).T;y=torch.tensor(b.train_y[ix][:,m.task_ids],device=device)
        p=m(x);loss=F.binary_cross_entropy(p,y);loss.backward();p0=p.detach().clone();base_loss=float(loss.detach());before=m.strengths()
        history={};hyper={}
        for opt in state['optimizers']:
            for group in opt['param_groups']:
                names=cfg['optimizer_groups'][group['group_name']]['names']
                for pid,name in zip(group['params'],names):history[name]=opt['state'].get(pid,{});hyper[name]=group
        a,bb=m.modulation_groups();target_ids={id(p) for p in a+bb};target=[(n,p) for n,p in named.items() if id(p) in target_ids and p.requires_grad and p.grad is not None]
        saved={n:p.detach().clone() for n,p in target}
        for name,p in target:
            hh=history.get(name,{});gg=hyper[name];step=int(hh.get('step',0));v=hh.get('exp_avg_sq',torch.zeros_like(p));mm=hh.get('exp_avg',torch.zeros_like(p));beta2=gg['betas'][1]
            denom=(v/(1-beta2**step)).sqrt()+gg['eps'] if step else torch.zeros_like(v)+gg['eps'];eta=gg['lr']/denom
            moment_rows.append(dict(run_id=run_id,label=label,step=state['steps'],parameter=name,parameter_norm=float(p.detach().norm()),data_gradient_norm=float(p.grad.norm()),m_norm=float(mm.norm()),v_norm=float(v.norm()),preconditioner_median=float(eta.median()),preconditioner_p95=float(torch.quantile(eta.flatten(),.95)),preconditioner_max=float(eta.max()),n=p.numel(),note='preconditioner scale, not update size or function learning rate'))
        for kind in ['adam_wd','adam_zero','adamw']:
            with torch.no_grad():
                for name,p in target:
                    p.copy_(saved[name]);hh=history.get(name,{});gg=hyper[name];b1,b2=gg['betas'];step=int(hh.get('step',0))+1;wd=1e-5 if kind!='adam_zero' else 0.
                    grad=p.grad+(wd*p if kind=='adam_wd' else 0.);mm=b1*hh.get('exp_avg',torch.zeros_like(p))+(1-b1)*grad;vv=b2*hh.get('exp_avg_sq',torch.zeros_like(p))+(1-b2)*grad.square()
                    if kind=='adamw':p.mul_(1-gg['lr']*wd)
                    p.sub_(gg['lr']*(mm/(1-b1**step))/((vv/(1-b2**step)).sqrt()+gg['eps']))
                pp=m(x);after=m.strengths();delta=(pp-p0).abs();risk=float(F.binary_cross_entropy(pp,y));neutral={}
                for mode in m.interventions():neutral[mode]=float((pp-m(x,neutral=mode)).abs().mean())
                row=dict(run_id=run_id,label=label,step=state['steps'],kind=kind,baseline_loss=base_loss,loss=risk,delta_loss=risk-base_loss,delta_p=float(delta.mean()),delta_p_max=float(delta.max()),counterfactual_scope='eval-mode train-meta gradient; only modulation-group parameters move, common model and moment state')
                row.update({'neutral_delta_p_'+k:v for k,v in neutral.items()});row.update({'before_'+k:v for k,v in before.items()});row.update({'after_'+k:v for k,v in after.items()});records.append(row)
                for name,p in target:p.copy_(saved[name])
    if records:pd.DataFrame(records).to_csv(out/'offline_local_counterfactual.csv',index=False)
    if moment_rows:pd.DataFrame(moment_rows).to_csv(out/'offline_moment_statistics.csv',index=False)
    dump(out/'local_mechanism_receipt.json',dict(status='COMPLETE_RAW_DIAGNOSTIC',run_id=run_id,counterfactual_rows=len(records),moment_rows=len(moment_rows),training_updates=0,test_read=False))
    print(json.dumps(dict(run_id=run_id,status='COMPLETE_RAW_DIAGNOSTIC',rows=len(records))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();local(a.run_id,a.device)
