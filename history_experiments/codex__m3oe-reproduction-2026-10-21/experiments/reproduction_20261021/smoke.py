#!/usr/bin/env python
"""Basic checks only: paired initialization, formula, optimizer and empty cells."""
import sys,copy,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(HERE.parents[1]),str(HERE)]
import torch
from data import load_data
from protocol import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='movielens',choices=['movielens','kuairand']); p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=512); p.add_argument('--output-root',default='/home_new/xzf_m3oe_reproduction_20261021'); args=p.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    b,manifest=load_data(args.dataset,Path(args.output_root)/'cache')
    # Deterministic diagnostic batch, includes every physical domain.
    idx=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:max(1,args.batch_size//3)] for d in range(3)])
    x=torch.from_numpy(b.train_x[idx]).T.to(args.device); y=torch.from_numpy(b.train_y[idx]).to(args.device)
    models={}; receipts={}; outputs={}
    for arm in ['O','B','D','G']:
        m,r=make_model(b,arm,2022,args.device); models[arm]=m; receipts[arm]=r; m.train()
        output=m(x)
        assert (output[0] if isinstance(output,tuple) else output).shape==y.shape
        assert isinstance(output,tuple)==(arm in ['D','G'])
        m.eval(); assert m(x).shape==y.shape
        m.train(); cells,info=batch_loss(m,x,y,arm,0,12345)
        opt,aux=optimizers(m,arm)
        assert (aux is not None)==(arm=='O')
        apply_backward(cells,m,arm); opt.step(); outputs[arm]={'loss':sum(float(c.detach()) for c in cells),'parameters':sum(p.numel() for p in m.parameters()),'module':m.__class__.__module__}
        if aux:
            assert all(any(p is q for group in opt.param_groups for q in group['params']) for group in aux.param_groups for p in group['params'])
    assert len({r['canonical_backbone_sha256'] for r in receipts.values()})==1
    assert receipts['D']['complete_initial_sha256']==receipts['G']['complete_initial_sha256']
    d,_=make_model(b,'D',2022,args.device); g,_=make_model(b,'G',2022,args.device)
    dopt,_=optimizers(d,'D'); gopt,_=optimizers(g,'G')
    dc,di=batch_loss(d,x,y,'D',0,333); gc,gi=batch_loss(g,x,y,'G',0,333)
    assert torch.equal(di['pred'],gi['pred'])
    # G0 means balancing off, D exact control; compare full one-step state.
    apply_backward(dc,d,'D'); apply_backward(gc,g,'D'); dopt.step(); gopt.step()
    dg_diff={n:float((p-g.state_dict()[n]).abs().max()) for n,p in d.state_dict().items() if not torch.equal(p,g.state_dict()[n])}
    assert not dg_diff,dg_diff
    d,_=make_model(b,'D',2022,args.device); g,_=make_model(b,'G',2022,args.device)
    dopt,_=optimizers(d,'D'); gopt,_=optimizers(g,'G')
    dc,_=batch_loss(d,x,y,'D',0,333); gc,_=batch_loss(g,x,y,'G',0,333)
    apply_backward(dc,d,'D'); apply_backward(gc,g,'G',tau=0)
    grad_diff={}; none_diff=[]
    for (n,p),(ng,q) in zip(d.named_parameters(),g.named_parameters()):
        if (p.grad is None)!=(q.grad is None): none_diff.append(n)
        if p.grad is not None and q.grad is not None:
            v=float((p.grad-q.grad).abs().max())
            if v: grad_diff[n]=v
    dopt.step();gopt.step()
    update_diff={n:float((p-g.state_dict()[n]).abs().max()) for n,p in d.state_dict().items() if not torch.equal(p,g.state_dict()[n])}
    m,_=make_model(b,'G',2022,args.device); em=x[:,:5].clone(); em[-1]=0
    ec,_=batch_loss(m,em,y[:5],'G',0,333); apply_backward(ec,m,'G'); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # Probe must leave parameter values, RNG and existing grads unchanged.
    m,_=make_model(b,'G',2022,args.device); cc,ii=batch_loss(m,x,y,'G',0,444)
    before=state_hash(m.state_dict()); rng=torch.get_rng_state().clone(); crng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    pr,_=probe(m,cc,ii,.5)
    assert before==state_hash(m.state_dict()) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(crng,torch.cuda.get_rng_state_all()))
    assert all(p.grad is None for p in m.parameters())
    report=dict(dataset=args.dataset,device=args.device,batch_size=len(idx),checks='PASS',receipts=receipts,one_step=outputs,D_vs_G_balancing_off_update_diff=dg_diff,tau0_vs_off_gradient_none_difference=none_diff,tau0_vs_off_gradient_max_difference=grad_diff,tau0_vs_off_update_max_difference=update_diff,empty_domain_check='PASS',probe_state_rng_grad_check='PASS',raw_sha256=manifest['raw_sha256'])
    out=HERE/('smoke_'+args.dataset+'.json'); json_write(out,report); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
