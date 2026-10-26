"""Stage2 controls reuse the frozen Stage1 model and optimizer paths."""
from pathlib import Path
import json
import torch
import protocol as parent
from protocol import *

CONTROLS={
    'Gtau0':dict(stage='stage2',control_id='Gtau0',parent_arm='G',gradient_path='per_cell_autograd_norm_scale_sum_zero_fill',effective_tau=0.0,weight_mode='learned_specificity',constant_weight=None),
    'Dconst01':dict(stage='stage2',control_id='Dconst01',parent_arm='D',gradient_path='ordinary_summed_backward_grad_none',effective_tau=None,weight_mode='fixed_constant_no_gradient',constant_weight=0.1),
}

def control_metadata(control): return dict(CONTROLS[control])
def make_model(bundle,arm,seed,device):
    return parent.make_model(bundle,CONTROLS[arm]['parent_arm'],seed,device)
def optimizers(model,arm): return parent.optimizers(model,CONTROLS[arm]['parent_arm'])
def batch_loss(model,x,y,arm,scarce,noise_seed,noise_std=.1):
    cells,info=parent.batch_loss(model,x,y,CONTROLS[arm]['parent_arm'],scarce,noise_seed,noise_std)
    if arm=='Dconst01':
        # Retain exactly the parent augmented forward, but disconnect the weights.
        info['w']=torch.full_like(info['w'],0.1,requires_grad=False)
        cells=[]
        for t in range(2):
            for d in range(3):
                mask=info['domain']==d
                cells.append((info['bce'][mask,t]*info['w'][mask]).mean() if mask.any() else info['pred'].sum()*0)
    return cells,info

def apply_backward(cells,model,arm,tau=None,gs=None):
    meta=CONTROLS[arm]
    # Gtau0 MUST retain the full G gradient/norm/scale/zero-fill path.
    return parent.apply_backward(cells,model,meta['parent_arm'],meta['effective_tau'] if meta['effective_tau'] is not None else .5,gs)

def auxiliary_receipt(model,optimizer):
    return {n:dict(has_gradient=p.grad is not None,has_optimizer_state=p in optimizer.state) for n,p in model.named_parameters() if n.startswith('dbo_')}

def parent_reference(root,dataset,control,seed,initialization,manifest):
    """Read only parent configuration/data receipts; never test or completion files."""
    from data import sha_file
    parent_arm=CONTROLS[control]['parent_arm']
    configs=sorted((Path(root)/'formal'/dataset/parent_arm/f'seed{seed}').glob('attempt_*/config.json'))
    if not configs: raise RuntimeError('Missing frozen Stage1 parent configuration: '+str((dataset,parent_arm,seed)))
    receipts=[]
    for path in configs:
        config=json.loads(path.read_text()); data_path=path.with_name('data_manifest.json'); data=json.loads(data_path.read_text())
        assert config['initialization']==initialization,('Parent initialization mismatch',str(path))
        for key in ['raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes','features']:
            assert data[key]==manifest[key],('Parent data mismatch',key,str(path))
        receipts.append(dict(config=str(path),config_sha256=sha_file(path),data_manifest=str(data_path),data_manifest_sha256=sha_file(data_path),initialization_matches=True,data_matches=True))
    return receipts
