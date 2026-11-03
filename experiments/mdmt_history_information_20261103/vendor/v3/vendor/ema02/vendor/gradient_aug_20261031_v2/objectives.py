"""Sample budget preserved while replacing a fixed share with K input views."""
import torch
from torch.nn import functional as F
from .views import draw_view
from .protocol import view_key


def losses(model,x,y,target,config,dataset,seed,step,*,mix_lambda=.5,kappa=.1,explicit_views=None):
    prediction=model(x)
    real=F.binary_cross_entropy(prediction,y,reduction='none')
    mask=x[-1]==target
    scalar=real.mean(1).mean();augmented=[];views=[]
    active=config['num_views']>0 and mix_lambda!=0 and bool(mask.any())
    if config['view_family']=='gaussian' and kappa==0:active=False
    if active:
        n=int(mask.sum())
        if explicit_views is None:
            for v in range(1,config['num_views']+1):
                key=view_key(dataset,seed,'train',step,config['view_family'],v)
                views.append(draw_view(model,n,config['view_family'],key,x.device,
                    negative_of=views[0] if config['pairing']=='antithetic' and v==2 else None))
        else:views=explicit_views
        for view in views:
            augmented.append(F.binary_cross_entropy(model(x[:,mask],view=view,kappa=kappa),y[mask],reduction='none'))
        scalar=(real.sum()+mix_lambda*(sum(v.sum() for v in augmented)/len(augmented)-real[mask].sum()))/real.numel()
    return scalar,dict(pred=prediction,real=real,mask=mask,augmented=augmented,views=views,
        extra_views=int(mask.sum())*len(augmented),forwards=1+len(augmented))
