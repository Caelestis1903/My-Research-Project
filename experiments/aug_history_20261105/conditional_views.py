"""Two task noises from one unchanged source Gaussian draw."""
import torch
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import digest_tensor


@torch.no_grad()
def draw_conditional_views(model,x_target,root,base_view):
    base=base_view['noise']
    if root.shape!=(2,len(model.perturb_slices),16,16) or base.shape!=(x_target.shape[1],len(model.perturb_slices),16):
        raise ValueError('Conditional view field layout differs')
    views=[]
    for task in range(2):
        identity=torch.eye(16,dtype=root.dtype,device=root.device).expand_as(root[task])
        noise=base if torch.equal(root[task],identity) else torch.einsum('nfi,fji->nfj',base,root[task].to(base))
        views.append(dict(base_view,noise=noise,hash=digest_tensor(noise),task=task))
    return views
