#!/usr/bin/env python
"""Stage2 minimal real-data one-batch checks; never evaluates held-out data."""
import argparse,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parents[1]),str(HERE)]
import numpy as np
import torch
import protocol as base
import stage2_protocol as controls
from data import load_data,sha_file

def capture_batch(model,x,y,arm,scarce,noise_seed,control=False):
    captured=[]
    hook=model.skip_conn.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach().clone()))
    try:
        cells,info=(controls.batch_loss if control else base.batch_loss)(model,x,y,arm,scarce,noise_seed)
    finally:hook.remove()
    return cells,info,captured[0]
def group_name(name):
    return 'complexity' if name.startswith('dbo_complexity.') else 'specificity' if name.startswith('dbo_specificity.') else 'backbone'
def difference(left,right):
    result={group:dict(max_abs=0.,parameters_differing=0,none_differences=[]) for group in ['backbone','specificity','complexity']}
    for name,a in left.items():
        b=right[name]; group=result[group_name(name)]
        if (a is None)!=(b is None):group['none_differences'].append(name)
        if a is not None and b is not None:
            val=float((a-b).abs().max());group['max_abs']=max(group['max_abs'],val);group['parameters_differing']+=int(val>0)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['movielens','kuairand'],default='movielens');p.add_argument('--device',default='cuda:0');p.add_argument('--output-root',default='/home_new/xzf_m3oe_reproduction_20261021');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bundle,manifest=load_data(args.dataset,Path(args.output_root)/'cache')
    index=np.concatenate([np.flatnonzero(bundle.train_x[:,-1]==d)[:170] for d in range(3)])
    x=torch.from_numpy(bundle.train_x[index]).T.to(args.device);y=torch.from_numpy(bundle.train_y[index]).to(args.device)
    scarce=min(bundle.train_counts,key=lambda d:(bundle.train_counts[d],d));noise_seed=2022000000
    report=dict(dataset=args.dataset,seed=2022,status='PASS',real_count=len(index),copy_count=int((x[-1]==scarce).sum()),controls={})
    for arm in ['Gtau0','Dconst01']:
        meta=controls.control_metadata(arm);parent=meta['parent_arm']
        m,receipt=controls.make_model(bundle,arm,2022,args.device);pm,pr=base.make_model(bundle,parent,2022,args.device)
        assert receipt==pr
        references=controls.parent_reference(args.output_root,args.dataset,arm,2022,receipt,manifest)
        cells,info,flat=capture_batch(m,x,y,arm,scarce,noise_seed,True)
        pc,pi,pflat=capture_batch(pm,x,y,parent,scarce,noise_seed)
        assert torch.equal(flat,pflat) and torch.equal(info['pred'],pi['pred'])
        assert torch.equal(info['bce'],pi['bce']) and torch.equal(info['domain'],pi['domain']) and torch.equal(info['aug'],pi['aug'])
        assert info['pred'].shape==(len(index)+report['copy_count'],2)
        assert torch.equal(flat[:len(index)],m.embedding(x,m.features,squeeze_dim=True))
        expected_unnoised=m.embedding(torch.cat([x,x[:,x[-1]==scarce]],dim=1),m.features,squeeze_dim=True)
        assert float((flat[len(index):]-expected_unnoised[len(index):]).abs().max())>0
        opt,_=controls.optimizers(m,arm);po,_=base.optimizers(pm,parent)
        initial={n:v.detach().clone() for n,v in m.state_dict().items()}
        result=dict(metadata=meta,initialization=receipt,parent_references=references,predictions_match_parent=True,augmented_labels_bce_match=True,noise_flat_embeddings_match=True)
        if arm=='Gtau0':
            assert torch.equal(info['w'],pi['w']) and all(torch.equal(a,b) for a,b in zip(cells,pc))
            grads=base.gradients(cells,list(m.parameters()))
            manual=[sum((g[j] for g in grads if g[j] is not None),torch.zeros_like(parameter)) for j,parameter in enumerate(m.parameters())]
            norms,scales=controls.apply_backward(cells,m,arm,gs=grads)
            assert scales==[1.]*6
            assert all(torch.equal(parameter.grad,expected) for parameter,expected in zip(m.parameters(),manual))
            # A matched D ordinary backward audits the known path distinction.
            base.apply_backward(pc,pm,'D')
            result['tau0_vs_ordinary_gradient_difference']=difference({n:p.grad for n,p in m.named_parameters()},{n:p.grad for n,p in pm.named_parameters()})
            opt.step();po.step()
            result['tau0_vs_ordinary_update_difference']=difference({n:p.detach()-initial[n] for n,p in m.named_parameters()},{n:p.detach()-initial[n] for n,p in pm.named_parameters()})
            m.eval();pm.eval()
            with torch.no_grad():result['tau0_vs_ordinary_prediction_max_abs']=float((m(x)-pm(x)).abs().max())
            result.update(scales=scales,manual_per_cell_zero_filled_sum_bitwise_matches=True,weights_and_six_cells_match_parent=True)
        else:
            constant=torch.full_like(info['w'],.1)
            assert torch.equal(info['w'],constant) and not info['w'].requires_grad
            for k,loss in enumerate(cells):
                t,d=divmod(k,3);mask=info['domain']==d
                expected=info['bce'][mask,t].mean()*torch.tensor(.1,dtype=torch.float32,device=x.device)
                assert torch.allclose(loss,expected,atol=3e-8,rtol=3e-7)
            observed,gs=base.probe(m,cells,info,0.)
            assert observed['auxiliary_gradient_norm']==0
            controls.apply_backward(cells,m,arm)
            assert all(p.grad is None for n,p in m.named_parameters() if n.startswith('dbo_'))
            opt.step()
            assert any(not torch.equal(p,initial[n]) for n,p in m.named_parameters() if group_name(n)=='backbone')
            assert all(torch.equal(p,initial[n]) for n,p in m.named_parameters() if n.startswith('dbo_'))
            result.update(all_weights_equal_float32_constant=True,constant_dtype=str(info['w'].dtype),constant_value=float(info['w'][0]),six_cell_formula_matches=True,auxiliary_gradient_norm=0.,auxiliary_parameters_unchanged=True,backbone_updated=True,weight_probe=observed['w'])
        result['auxiliary_parameters']=controls.auxiliary_receipt(m,opt)
        report['controls'][arm]=result
    # Formal routing is explicit even for effective tau0; no training is invoked.
    import stage2_run
    for arm in controls.CONTROLS:
        a=stage2_run.parser().parse_args(['--arm',arm]);assert stage2_run.is_formal(a)
        a.smoke=True;assert not stage2_run.is_formal(a)
    report['formal_control_routing']='PASS'
    frozen={name:sha_file(HERE/name) for name in ['protocol.py','run.py','schedule_runs.py']}
    assert frozen==__import__('json').loads((HERE/'stage2_base_hashes.json').read_text())
    report['stage1_sources_unchanged']=frozen
    base.json_write(HERE/f'smoke_stage2_{args.dataset}.json',report)
    print(__import__('json').dumps(report,indent=2))
if __name__=='__main__':main()
