"""Task-specific supervised replacement with the original actual 2N denominator."""
import torch
from torch.nn import functional as F
from .objectives import LossTerms,loss_terms,source_losses
from .conditional_views import draw_conditional_views
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import draw_view,digest_tensor
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.protocol import view_key


def conditional_loss_terms(model,x,y,root,config):
    if config['gamma']==0:
        return loss_terms(model,x,y,config)
    mask=x[-1]==0
    if not bool(mask.any()):
        return loss_terms(model,x,y,config)
    key=view_key(config['dataset'],config['seed'],'train',config['step'],'gaussian',1)
    base=draw_view(model,int(mask.sum()),'gaussian',key,x.device)
    views=draw_conditional_views(model,x[:,mask],root,base)
    if torch.equal(root[0],root[1]):
        total,info=source_losses(model,x,y,0,dict(num_views=1,view_family='gaussian',pairing='iid'),
            config['dataset'],config['seed'],config['step'],mix_lambda=config['lambda_aug'],kappa=config['kappa'],
            explicit_views=[views[0]])
        real=info['real']; augmented=info['augmented']
        terms={(0,t):config['lambda_aug']*(augmented[0][:,t]-real[mask,t]).sum()/y.numel() for t in range(2)}
    else:
        real=F.binary_cross_entropy(model(x),y,reduction='none')
        augmented=[F.binary_cross_entropy(model(x[:,mask],view=view,kappa=config['kappa'])[:,t],y[mask,t],reduction='none')
            for t,view in enumerate(views)]
        terms={(0,t):config['lambda_aug']*(augmented[t]-real[mask,t]).sum()/y.numel() for t in range(2)}
        total=(real.sum()+config['lambda_aug']*sum(augmented[t].sum()-real[mask,t].sum() for t in range(2)))/y.numel()
        info=dict(real=real,mask=mask,views=views,extra_views=int(mask.sum())*2,forwards=3)
    info['conditional_metadata']=dict(base_key=key,base_seed=base['seed'],base_noise_hash=base['hash'],
        root_hash=digest_tensor(root),base_noise_squared_norm_mean=float(base['noise'].square().sum(-1).mean()),
        transformed_noise_squared_norm_mean=[float(v['noise'].square().sum(-1).mean()) for v in views],
        expected_field_noise_squared_norm=16.)
    return LossTerms(total,real.mean(1).mean(),terms,info)
