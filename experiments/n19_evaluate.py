"""Frozen selected/final readout and physical factor export; no optimization."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from n19_data import OUT,population,dump
from n19_models import Model
from n19_objectives import clean_predictions,NoiseChannel
from n19_train import metrics

class CompactRank(Model):
    def __init__(self,features,seed,method,state):
        super().__init__(features,seed,method)
        already_compact='active_left.0' in state
        if not already_compact:self.load_state_dict(state)
        left=[];right=[]
        for d in range(3):
            if already_compact:
                left.append(nn.Parameter(state[f'active_left.{d}'].detach().clone()));right.append(nn.Parameter(state[f'active_right.{d}'].detach().clone()))
            else:
                active=self.rank_mask[d].bool();left.append(nn.Parameter(self.left[d,:,active].detach().clone()));right.append(nn.Parameter(self.right[d,active].detach().clone()))
        del self.left,self.right,self.rank_mask,self.rank_frozen
        self.active_left=nn.ParameterList(left);self.active_right=nn.ParameterList(right);self.rank_mode=False
        if already_compact:self.load_state_dict(state)
    def slots(self,masks=None):return [a@b for a,b in zip(self.active_left,self.active_right)]

@torch.no_grad()
def evaluate(run,device):
    freeze=OUT/'TEST_FREEZE.json'
    if not freeze.exists():raise RuntimeError('Test decisions have not been frozen')
    decision=json.loads(freeze.read_text())
    if run.name not in decision['runs']:raise RuntimeError('Run absent from frozen readout manifest')
    c=json.loads((run/'config.json').read_text());assert not c['engineering']
    if (run/'test_receipt.json').exists():return
    torch.set_num_threads(2);torch.cuda.set_device(device);data=population(c['dataset'],c['suite'],c.get('noise_interaction',0.))
    state_dir=Path(c['state_directory']);m=Model(data.features,c['seed'],c['method'],c.get('backbone')).to(device).eval()
    started=time.time();records=[];diagnostics={}
    for stage,filename in [('selected','selected.pt'),('final','last.pt')]:
        state=torch.load(state_dir/filename,map_location=device,weights_only=False);m.load_state_dict(state['model'])
        predictions=[];flips=[]
        for start in range(0,len(data.test_x),c['batch_size']):
            x=torch.tensor(data.test_x[start:start+c['batch_size']],device=device,dtype=torch.float32).T;p=m(x)
            if c['method']=='task_mask':
                masked=x.clone();masked[-2]=0;p=torch.stack([m(masked)[:,0],p[:,1]],1)
            p=clean_predictions(p,c['method']);predictions.append(p.cpu().numpy())
            if c['suite']=='stability':
                xx=x.clone();xx[-2]=3-xx[-2];q=m(xx)
                if c['method']=='task_mask':q=torch.stack([m(masked)[:,0],q[:,1]],1)
                flips.append((p-q).abs().cpu().numpy())
        pred=np.concatenate(predictions)
        np.savez_compressed(run/f'test_{stage}.npz',predictions=pred,labels=data.test_y,domains=data.test_x[:,-1],row_ids=data.test_ids)
        for row in metrics(pred,data.test_y,data.test_x[:,-1]):row.update(stage=stage,epoch=state['epoch']);records.append(row)
        if flips:
            gap=np.concatenate(flips);diagnostics[stage+'_nuisance_flip_abs_probability']=gap.mean(0).tolist()
        if state['channel'] is not None:
            channel=NoiseChannel(c['method'] in ['axis_channel','nested_axis','joint_axis','profile_axis','profile_nested','profile_joint']).to(device)
            channel.load_state_dict(state['channel']);aa,bb=channel.rates()
            estimated=np.stack([aa.cpu().numpy(),bb.cpu().numpy()]);truth=np.stack([data.noise_rates['a'],data.noise_rates['b']])
            diagnostics[stage+'_noise_channel']=dict(a=estimated[0].tolist(),b=estimated[1].tolist(),rmse=float(np.sqrt(((estimated-truth)**2).mean())),
                per_rate_errors=(estimated-truth).tolist())
        if c['method'].startswith('mask_'):
            diagnostics[stage+'_observation_state']=dict(copula=state['copula'].cpu().tolist(),labels=state['total_labels'],paired=state['total_pairs'],policy_counts=state.get('policy_counts'))
        if getattr(m,'mix_mode',False):
            diagnostics[stage+'_effective_router_logits']=m.router_logits().cpu().tolist()
            x=torch.tensor(data.test_x[:512],device=device,dtype=torch.float32).T
            _,info=m(x,details=True)
            diagnostics[stage+'_source_weights']=[dict(domain=d,task=t,weights=w.cpu().tolist()) for d,t,w in info['router_weights']]
        if getattr(m,'rank_mode',False):
            compact=CompactRank(data.features,c['seed'],c['method'],{k:v.cpu() for k,v in state['model'].items()}).to(device).eval()
            x=torch.tensor(data.test_x[:1024],device=device,dtype=torch.float32).T
            error=float((m(x)-compact(x)).abs().max());assert error<2e-5
            torch.save(dict(model=compact.cpu().state_dict(),active_ranks=[int(a.shape[1]) for a in compact.active_left],method=c['method'],seed=c['seed'],features=data.features),state_dir/f'compact_{stage}.pt')
            diagnostics[stage+'_physical_export']=dict(parameters=sum(p.numel() for p in compact.parameters()),max_prediction_error=error,
                path=str(state_dir/f'compact_{stage}.pt'),active_ranks=[int(a.shape[1]) for a in compact.active_left])
    dump(run/'metrics_test.json',records);dump(run/'diagnostics_test.json',diagnostics)
    dump(run/'test_receipt.json',dict(status='COMPLETE_FROZEN_TEST_READOUT',selected_and_final=True,
        frozen_manifest_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),wall_seconds=time.time()-started,optimization_steps=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',required=True);a=p.parse_args();evaluate(a.run,torch.device(a.device))
