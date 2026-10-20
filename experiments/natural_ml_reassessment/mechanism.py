"""Offline fixed-theta, fixed-state diagnostics, separate from formal training.

Every covariance trace uses every parameter coordinate. The finite-pool oracle
uses 32 real rows/domain and is explicitly distinct from the full-population
Monte Carlo estimate. The routing risk Jacobian reattaches only the gate input
on a deep clone and verifies that predictions are identical.
"""
import argparse
import copy
import inspect
from pathlib import Path
import textwrap
import types
import numpy as np
import torch
import torch.nn.functional as F
from .bootstrap import activate, ROOT, dump, sha
activate()
from stage2d_20261016 import Sampler
from stage_S1_models import build_model
from stage_S3_gradient_access import partition_parameters
from stage_E1_instrumentation import preserve_rng_state
from .data import load_population

def flat_gradient(model,b,rows,weights,device):
    x=torch.as_tensor(b.train_x[rows],device=device).T
    y=torch.as_tensor(b.train_y[rows],device=device)
    terms=F.binary_cross_entropy(model(x),y,reduction='none').mean(1)
    loss=(terms*torch.as_tensor(weights,device=device,dtype=terms.dtype)[x[-1].long()]).mean()
    parameters=list(model.parameters())
    gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(parameters,gradients)])

def block_indices(model):
    partition=partition_parameters(model);belongs={n:g for g,ns in partition['groups'].items() for n in ns}
    result={g:[] for g in partition['groups']};start=0
    for name,p in model.named_parameters():
        result[belongs[name]].append((start,start+p.numel()));start+=p.numel()
    return result

def squared_blocks(vector,blocks):
    return {g:sum(float(vector[a:z].double().square().sum()) for a,z in ranges) for g,ranges in blocks.items()}

class Moments:
    def __init__(self):self.n=0;self.mean=None;self.m2=None
    def add(self,x):
        x=x.double();self.n+=1
        if self.mean is None:self.mean=x.clone();self.m2=torch.zeros_like(x);return
        delta=x-self.mean;self.mean+=delta/self.n;self.m2+=delta*(x-self.mean)
    def report(self,blocks):
        traces=squared_blocks(torch.sqrt(torch.clamp(self.m2/(self.n-1),min=0)),blocks)
        return dict(samples=self.n,mean_squared_norm=float(self.mean.square().sum()),
                    covariance_trace=sum(traces.values()),block_covariance_trace=traces,
                    mean_standard_error_norm=float(torch.sqrt(self.m2.sum()/((self.n-1)*self.n))))

def risk_clone(model):
    clone=copy.deepcopy(model)
    method=model.forward.__func__
    source=textwrap.dedent(inspect.getsource(method))
    if 'emb.detach()' not in source:raise RuntimeError('unrecognized routing derivative source')
    namespace=dict(method.__globals__)
    exec(compile(source.replace('emb.detach()','emb'),'<diagnostic_prediction_risk_jacobian>','exec'),namespace)
    clone.forward=types.MethodType(namespace['forward'],clone)
    return clone

def finite_pool(model,b,alpha,p,device,blocks):
    selected=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:32] for d in range(3)])
    if len(selected)!=96:raise RuntimeError('finite oracle requires 32 real rows/domain')
    # Each row gradient has the two-task BCE mean and no population factor.
    row_gradients=torch.stack([flat_gradient(model,b,np.array([row]),np.ones(3),device) for row in selected])
    domain_means=row_gradients.reshape(3,32,-1).mean(1)
    oracle=sum(float(alpha[d])*domain_means[d] for d in range(3))
    result=dict(row_ids=b.train_row_ids[selected].tolist(),rows_per_domain=32,
                objective_mean_gradient_norm=float(oracle.double().norm()),identity={},monte_carlo={})
    for label,q in [('SN',p),('SB',np.ones(3)/3)]:
        factors=np.repeat(alpha/q,32)
        probabilities=np.repeat(q/32,32)
        weighted=row_gradients*torch.as_tensor(factors,device=device,dtype=row_gradients.dtype)[:,None]
        expected=(weighted.double()*torch.as_tensor(probabilities,device=device)[:,None]).sum(0)
        error=expected-oracle.double()
        second={g:0. for g in blocks}
        for i in range(96):
            sq=squared_blocks(weighted[i].double()-expected,blocks)
            for g in blocks:second[g]+=float(probabilities[i])*sq[g]/40960
        result['identity'][label]=dict(full_coordinate_error_norm=float(error.norm()),
            exact_batch_covariance_trace=sum(second.values()),block_exact_batch_covariance_trace=second,
            batch_size=40960,denominator='fixed actual batch size; IID with replacement')
        # Exact finite-pool simulation via counts equals sampling every one of
        # the 40960 rows, without 40960 repeated matrix rows in memory.
        rng=np.random.default_rng(20221020+(label=='SB'));moments=Moments()
        for _ in range(512):
            counts=torch.as_tensor(rng.multinomial(40960,probabilities),device=device,dtype=weighted.dtype)
            moments.add(counts@weighted/40960)
        report=moments.report(blocks);report['mc_error_norm_vs_exact_oracle']=float((moments.mean-oracle.double()).norm())
        result['monte_carlo'][label]=report
    return result,domain_means,selected

def adam_grid(model,state,domain_means,alpha,blocks):
    """Counterfactual m/v/current grid derived from ONE baseline state.

    A one-step synthetic history event supplies full/target m and v. Then a
    current-gradient step uses all eight m/v/current combinations. No gradient,
    moment, parameter or RNG from another training trajectory is used.
    """
    full=sum(float(alpha[d])*domain_means[d] for d in range(3));target=float(alpha[2])*domain_means[2]
    shared_names=set(partition_parameters(model)['groups']['C'])
    opt=state.get('optimizer',{})
    if not opt or not opt.get('param_groups'):return dict(applicable=False,reason='no Adam baseline optimizer state')
    names=state.get('optimizer_parameter_names',list(dict(model.named_parameters())))
    groups=opt['param_groups'];name_state={};name_group={}
    position=0
    for pg in groups:
        for pid in pg['params']:
            name=names[position];position+=1;name_state[name]=opt['state'].get(pid,{});name_group[name]=pg
    results=[]
    for msource in ('full','target'):
        for vsource in ('full','target'):
            for current in ('full','target'):
                norm=0.;offset=0
                for name,param in model.named_parameters():
                    count=param.numel();gf=full[offset:offset+count].reshape_as(param);gt=target[offset:offset+count].reshape_as(param);offset+=count
                    if name not in shared_names:continue
                    pg=name_group[name];s=name_state[name];b1,b2=pg['betas'];wd=pg['weight_decay'];eps=pg['eps'];lr=pg['lr']
                    t=int(s.get('step',0));m=s.get('exp_avg',torch.zeros_like(param)).to(param.device);v=s.get('exp_avg_sq',torch.zeros_like(param)).to(param.device)
                    effective={'full':gf+wd*param.detach(),'target':gt+wd*param.detach()}
                    mh=b1*m+(1-b1)*effective[msource];vh=b2*v+(1-b2)*effective[vsource].square()
                    mn=b1*mh+(1-b1)*effective[current];vn=b2*vh+(1-b2)*effective[current].square()
                    delta=-lr*(mn/(1-b1**(t+2)))/(torch.sqrt(vn/(1-b2**(t+2)))+eps)
                    norm+=float(delta.double().square().sum())
                results.append(dict(m_source=msource,v_source=vsource,current_source=current,applied_shared_delta_norm=norm**.5))
    return dict(applicable=True,definition='same baseline theta/state; synthetic single history event followed by current event; no cross-trajectory causal interpretation',grid=results)

def run_diagnostic(run_dir,checkpoint,output,device='cuda:0'):
    run_dir=Path(run_dir);state=torch.load(run_dir/checkpoint,map_location='cpu',weights_only=False)
    cfg=state.get('config') or __import__('json').loads((run_dir/'config.json').read_text())
    b=load_population(cfg['dataset_config']);p=np.array([b.train_counts[d] for d in range(3)],float);p/=p.sum()
    alpha=np.asarray(cfg.get('alpha',p if cfg.get('objective_weighting','sample')=='sample' else np.ones(3)/3),float)
    model=build_model(b.features,3,2,'m3oe',seed=cfg['seed']).to(device)
    model.load_state_dict(state['model']);modules=dict(model.named_modules())
    for name,tau in state.get('model_tau',{}).items():modules[name].tau=tau
    model.eval();blocks=block_indices(model)
    result=dict(checkpoint=checkpoint,checkpoint_sha256=sha(run_dir/checkpoint),dataset_config=cfg['dataset_config'],
                alpha=alpha.tolist(),p=p.tolist(),mc_independent_batches=512,batch_size=40960,
                gradient_definition='training_derivative_with_original_routing_detach',full_coordinate=True)
    with preserve_rng_state():
        oracle,dm,selected=finite_pool(model,b,alpha,p,device,blocks);result['finite_stratified_pool_oracle']=oracle
        means={}
        for label,q in [('SN',p),('SB',np.ones(3)/3)]:
            sampler=Sampler(b,q,20221020,117);moments=Moments()
            for _ in range(512):moments.add(flat_gradient(model,b,sampler.sample(40960),alpha/q,device))
            result[label]=moments.report(blocks);means[label]=moments.mean
        result['same_target_mc_mean_difference_norm']=float((means['SN']-means['SB']).norm())
        result['adam_same_baseline_grid']=adam_grid(model,state,dm,alpha,blocks)
        clone=risk_clone(model)
        x=torch.as_tensor(b.train_x[selected],device=device).T
        with torch.no_grad():
            discrepancy=float((clone(x)-model(x)).abs().max())
        if discrepancy!=0.:raise AssertionError('risk Jacobian clone changes predictions')
        training=flat_gradient(model,b,selected,alpha/np.ones(3)*3,device)
        risk=flat_gradient(clone,b,selected,alpha*3,device)
        result['prediction_risk_jacobian']=dict(definition='gate input reattached on deep clone; same prediction function values',
              prediction_max_abs_difference=discrepancy,all_coordinate_gradient_norm=float(risk.double().norm()),
              training_derivative_norm=float(training.double().norm()),difference_norm=float((risk-training).double().norm()),
              block_difference_squared_norm=squared_blocks(risk-training,blocks))
    dump(output,result)

def configurations(runs):
    tasks=[]
    for cfg in runs:
        if cfg['stage']!='AC-2D' or cfg['arm'] not in ('SN','SB'):continue
        for checkpoint in ('initial_state.pt','epoch09.pt','epoch39.pt','selected_state.pt','epoch49.pt'):
            tasks.append(dict(run_id=cfg['run_id'],run_dir=cfg['run_dir'],checkpoint=checkpoint,
                  output=str(ROOT/'diagnostics'/cfg['run_id']/(checkpoint+'.json')),
                  mc_batches=512,batch_size=40960,oracle_rows_per_domain=32,status='WAIT_FOR_FORMAL_CHECKPOINT'))
    return tasks

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    run_diagnostic(args.run_dir,args.checkpoint,args.output,args.device)
