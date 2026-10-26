"""P8-W: measured objective/gradient equivalence, not full D3 or SAF."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from v2_20261016 import OUT,sha,dump,fixed_model,group
from stage_S1_data import load_stage_s1_bundle


def main(run,device='cpu'):
    config=json.loads((run/'config.json').read_text());seed=config['seed'];assert config['dataset']=='kuairand'
    torch.set_num_threads(2);torch.manual_seed(971)
    b=load_stage_s1_bundle('kuairand',None,'natural',seed);model=fixed_model(b,seed).to(device)
    probe=OUT/'probes'/'kuairand'/config['optimizer']/f'seed{seed}'
    with np.load(probe/'pool_rows.npz',allow_pickle=False) as z:indices=z['indices'].reshape(-1)
    x=torch.tensor(b.train_x[indices],device=device).T;y=torch.tensor(b.train_y[indices],device=device);domains=x[-1].long()
    p=np.array([b.train_counts[d] for d in range(3)],dtype=float);p/=p.sum()
    alpha=torch.tensor(p,dtype=torch.float64,device=device);row_weight=alpha[domains]/256
    initial=torch.load(run/'epoch00.pt',map_location='cpu',weights_only=False);model.load_state_dict(initial['model'],strict=True)
    with torch.no_grad():
        embedding=model.embedding(x,model.features,squeeze_dim=False).double()
        attention=torch.softmax(torch.bmm(embedding,embedding.transpose(1,2))/math.sqrt(embedding.shape[-1]),dim=-1)
        entropy=-(attention*torch.log(attention.clamp_min(1e-12))).sum(-1).mean(-1)
        initial_entropy_std=float(entropy.std(unbiased=False));weight_source='initial'
        if initial_entropy_std<=1e-8:
            # Preregistered degeneracy gate; never pick weights from selected/test.
            warm=torch.load(run/'epoch10.pt',map_location='cpu',weights_only=False);model.load_state_dict(warm['model'],strict=True)
            embedding=model.embedding(x,model.features,squeeze_dim=False).double()
            attention=torch.softmax(torch.bmm(embedding,embedding.transpose(1,2))/math.sqrt(embedding.shape[-1]),dim=-1)
            entropy=-(attention*torch.log(attention.clamp_min(1e-12))).sum(-1).mean(-1);weight_source='fixed_epoch10'
        weight=.1+.5*torch.clamp((entropy-entropy.mean())/(2*entropy.std(unbiased=False).clamp_min(1e-12))+.5,.1,1)
    normalized=weight.double()/float((weight.double()*row_weight).sum())
    target=OUT/'analysis'/'weight_equivalence'/f'seed{seed}';target.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(target/'frozen_complexity_weights.npz',row_ids=b.train_row_ids[indices],weight=weight.cpu().numpy(),normalized=normalized.cpu().numpy())
    names=list(dict(model.named_parameters()));params=list(model.parameters());records=[]
    for key,file in [('initial','epoch00.pt'),('selected','selected_state.pt'),('final','epoch50.pt')]:
        state=torch.load(run/file,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);model.eval()
        terms=F.binary_cross_entropy(model(x),y,reduction='none').mean(1).double()
        factors={'original':torch.ones_like(weight),'weighted':weight,'normalized':normalized,'cancelled':weight*(1/weight)}
        grads={};values={}
        for i,(name,factor) in enumerate(factors.items()):
            loss=(terms*row_weight*factor).sum();values[name]=float(loss.detach())
            gg=torch.autograd.grad(loss,params,retain_graph=i<len(factors)-1,allow_unused=True)
            grads[name]={g:torch.cat([(v if v is not None else torch.zeros_like(par)).detach().flatten() for n,par,v in zip(names,params,gg) if group(n)==g]).double().cpu().numpy() for g in ('C','P','R','E')}
        for condition in factors:
            for block in grads[condition]:
                difference=np.linalg.norm(grads[condition][block]-grads['original'][block])
                records.append(dict(snapshot=key,epoch=state['epoch'],condition=condition,block=block,
                    population_objective_on_pool=values[condition],objective_difference=values[condition]-values['original'],
                    gradient_difference_norm=float(difference),baseline_gradient_norm=float(np.linalg.norm(grads['original'][block]))))
                if condition=='cancelled':assert difference<1e-6 and abs(values[condition]-values['original'])<1e-7
        # Exact finite support expectation of each corrected estimator.
        detached=terms.detach()
        for condition,factor in factors.items():
            expected=[]
            for q in (p,np.ones(3)/3):
                expected.append(sum(float(q[d]*p[d]/q[d]*(detached[domains==d]*factor[domains==d]).mean()) for d in range(3)))
            assert abs(expected[0]-expected[1])<1e-10 and abs(expected[0]-values[condition])<1e-7
    pd.DataFrame(records).to_csv(target/'objective_gradient_comparison.csv',index=False)
    dump(target/'receipt.json',dict(status='PASS',source_sha256=sha(__file__),seed=seed,run=str(run),
        inverse_cancellation_identity=True,normalized_weights_can_change_objective=True,
        interpretation='D3-inspired fixed complexity control only; no faithful full D3/SAF efficacy or noncoverage claim',
        weight_source=weight_source,initial_entropy_std=initial_entropy_std,
        nonconstant_weights=bool(float(weight.std())>0),weight_min=float(weight.min()),weight_max=float(weight.max())))
    print('P8-W complete',seed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();main(a.run,a.device)
