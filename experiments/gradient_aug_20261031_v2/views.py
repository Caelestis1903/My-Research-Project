"""Explicit, independently seeded non-routing embedding views."""
import hashlib
import torch
from .vendor_ga31.model import MDMTRec as SourceModel, make_model as source_model
from .protocol import derive_seed


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def draw_view(model,rows,family,key,device,negative_of=None):
    if negative_of is not None:
        return dict(family='gaussian',noise=-negative_of['noise'],key=negative_of['key']+'|negative',
                    seed=negative_of['seed'],hash=digest_tensor(-negative_of['noise']))
    gen=torch.Generator(device=device).manual_seed(derive_seed(key))
    if family=='gaussian':
        noise=torch.stack([torch.randn((rows,end-start),generator=gen,device=device)
                           for start,end in model.perturb_slices],dim=1)
        return dict(family=family,noise=noise,key=key,seed=derive_seed(key),hash=digest_tensor(noise))
    if family=='fieldmask':
        fields=torch.randint(len(model.perturb_slices),(rows,),device=device,generator=gen)
        return dict(family=family,fields=fields,key=key,seed=derive_seed(key),hash=digest_tensor(fields))
    raise ValueError(f'Unknown family {family}')


def apply_view(embedding,slices,view,kappa=.1):
    result=embedding.clone()
    for index,(lo,hi) in enumerate(slices):
        clean=embedding[:,lo:hi]
        if view['family']=='gaussian':
            scale=clean.square().mean(1,keepdim=True).sqrt().detach()
            result[:,lo:hi]=clean+kappa*scale*view['noise'][:,index,:]
        elif view['family']=='fieldmask':
            result[:,lo:hi]=clean*(view['fields']!=index).unsqueeze(1)
        else:raise ValueError(view['family'])
    return result


class MDMTRec(SourceModel):
    def forward(self,x,test_flag=False,*,view=None,kappa=.1):
        if view is None:return super().forward(x,test_flag=test_flag)
        handle=self.embedding.register_forward_hook(lambda module,inputs,output:apply_view(output,self.perturb_slices,view,kappa))
        try:return super().forward(x,test_flag=test_flag)
        finally:handle.remove()


def make_model(bundle,conf,reference=False):
    model=source_model(bundle,conf,reference=reference)
    if not reference:model.__class__=MDMTRec
    return model
