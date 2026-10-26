"""Fixed-theta AO04 fusion sensitivity on a prespecified validation probe."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .ao04 import OUT,prepare,check,sources
from .bootstrap import sha,dump

def execute(run_dir,device='cuda:0'):
    check();run=Path(run_dir);cfg=json.loads((run/'config.json').read_text())
    assert json.loads((run/'receipt.json').read_text())['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
    b,r=prepare(device)
    from stage_S1_metrics import safe_auc,safe_logloss
    # Fixed before new results: first eight original validation rows of each domain.
    ix=np.concatenate([np.flatnonzero(b.val_x[:,-1]==d)[:8] for d in range(3)])
    x=torch.tensor(b.val_x[ix],device=device,dtype=torch.float32).T
    y=b.val_y[ix];domains=b.val_x[ix,-1].astype(int)
    info=json.loads((run/'run_manifest.json').read_text())
    checkpoints={'initial':Path(info['paired_init_state_path']),
        'selected':run/'selected_checkpoint.pth','final':run/'final_checkpoint.pth'}
    folder=run/'local_fusion_diagnostics';folder.mkdir(exist_ok=False);rows=[];metadata=[]
    for label,path in checkpoints.items():
        before=sha(path);state=torch.load(path,map_location='cpu',weights_only=False)
        arm='B1' if cfg['arm']=='B0' else cfg['arm']
        model=r.build_ao04_model(r.features_from_schema(b.features),arm,seed=cfg['seed']).to(device).eval()
        result=model.load_state_dict(state,strict=False)
        assert not result.unexpected_keys and set(result.missing_keys)<=({'q_rho'} if label=='initial' or cfg['arm']=='B0' else set())
        captured={};hooks=[]
        def hook(name):
            def capture(module,args,out):captured[name]=out
            return capture
        for d,layer in enumerate(model.domain_expert):hooks.append(layer.register_forward_hook(hook('E'+str(d))))
        for cell,layer in enumerate(model.tower):hooks.append(layer.register_forward_hook(hook('L'+str(cell))))
        pred=model(x);q=model.q_rho;original=q.detach().clone();rho=torch.sigmoid(q).detach()
        jac=[]
        for i,d in enumerate(domains):
            for task in range(2):
                logit=captured['L'+str(task*3+int(d))][i,0]
                jac.append(torch.autograd.grad(logit,q,retain_graph=True)[0].detach().cpu().numpy())
        jac=np.asarray(jac).reshape(len(ix),2,-1)
        experts=torch.stack([captured['E'+str(d)].detach() for d in range(3)],dim=1).cpu().numpy()
        differences=np.stack([experts[:,j]-experts[:,k] for j,k in [(1,2),(0,2),(0,1)]],axis=1)
        a=float(model._weight_exp_d().detach());bd=float(model._weight_bal_d().detach())
        np.savez_compressed(folder/(label+'_sensitivity.npz'),row_ids=b.val_row_ids[ix],labels=y,domains=domains,
            baseline_predictions=pred.detach().cpu().numpy(),domain_experts=experts,other_expert_differences=differences,
            logit_jacobian_wrt_q=jac,logit_jacobian_wrt_rho=jac/(rho.cpu().numpy()*(1-rho.cpu().numpy())),rho=rho.cpu().numpy())
        for h in hooks:h.remove()
        def record(case,pp):
            for d in range(3):
                for task in range(2):
                    chosen=domains==d;yy=y[chosen,task];ss=pp[chosen,task]
                    auc=safe_auc(yy,ss) if len(np.unique(yy))==2 else None
                    rows.append(dict(checkpoint=label,case=case,domain=d,task=task,n=int(chosen.sum()),
                        auc=auc,logloss=safe_logloss(yy,ss),auc_defined=auc is not None))
        record('baseline',pred.detach().cpu().numpy())
        with torch.no_grad():
            for index in range(q.numel()):
                for delta in (-.1,.1):
                    q.copy_(original);q[index]+=delta;pp=model(x).cpu().numpy()
                    np.savez_compressed(folder/f'{label}_q{index}_{delta:+g}.npz',predictions=pp)
                    record(f'q{index}_{delta:+g}',pp)
            q.copy_(original)
        assert sha(path)==before
        metadata.append(dict(checkpoint=label,source=str(path),sha256=before,
            a_D=a,b_D=bd,a_D_times_one_minus_b_D=a*(1-bd),
            training_arm=cfg['arm'],probe_arm=arm,
            B0_probe='same weights with B1 q=0 local counterfactual' if cfg['arm']=='B0' else None,
            derivative='raw tower logit wrt q and rho; original training graph, eval mode',
            perturbation='single raw q coordinate +/-0.1; other theta fixed'))
    pd.DataFrame(rows).to_csv(folder/'probe_cell_risks.csv',index=False)
    dump(folder/'receipt.json',dict(status='COMPLETE_RAW_DIAGNOSTIC',training_updates=0,test_read=False,
        probe='first8 validation rows per physical domain; local sensitivity, not full validation performance',
        checkpoints=metadata,sources=sources()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();execute(a.run_dir,a.device)
