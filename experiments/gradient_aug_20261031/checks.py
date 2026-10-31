"""Required pre-run identities and a full-size CUDA batch; no test metrics."""
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import make_model
from .objectives import losses
from .protocol import setup,optimizer,cpu_copy,equal,snapshot,restore,rng_state,restore_rng,tensor_bytes
from .probe import combine,groups,vector,geometry


def run_checks(bundles,p,device,out):
    setup(2022,device);results={}
    for dataset,bundle in bundles.items():
        conf=p['datasets'][dataset]
        model=make_model(bundle,conf);initial=cpu_copy(model.state_dict())
        x=torch.as_tensor(bundle.train_x[:32],device=device).T
        y=torch.as_tensor(bundle.train_y[:32],device=device)
        reference=make_model(bundle,conf,reference=True).to(device);reference.load_state_dict(initial)
        model=model.to(device);a=optimizer(model,p);b=optimizer(reference,p)
        actual,_=losses(model,x,y,conf['augmented_domain'],2022000001,enabled=False)
        oracle=F.binary_cross_entropy(reference(x),y)
        torch.testing.assert_close(model(x),reference(x),atol=1e-6,rtol=1e-5)
        actual.backward();oracle.backward()
        for left,right in zip(model.parameters(),reference.parameters()):
            assert (left.grad is None)==(right.grad is None)
            if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,atol=1e-6,rtol=1e-5)
        a.step();b.step()
        for left,right in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)
        del model,reference,a,b
        torch.cuda.empty_cache()
        peaks=[]
        for arm in p['arms']:
            setup(2022,device);m=make_model(bundle,conf).to(device);m.load_state_dict(initial);opt=optimizer(m,p)
            x=torch.as_tensor(bundle.train_x[:40960],device=device).T
            y=torch.as_tensor(bundle.train_y[:40960],device=device)
            before_rng=rng_state(device);torch.cuda.reset_peak_memory_stats(device)
            value,info=losses(m,x,y,conf['augmented_domain'],2022000001,enabled=arm['augmentation_enabled'])
            assert equal(before_rng,rng_state(device)), 'augmentation consumed global RNG'
            (arm['loss_scale']*value).backward();opt.step();torch.cuda.synchronize(device)
            peaks.append(dict(arm=arm['id'],rows=len(y),loss_finite=bool(torch.isfinite(value)),
                              peak_memory_bytes=torch.cuda.max_memory_allocated(device)))
            # The joint copy must bind optimizer states to its own model parameters, including None semantics.
            clone,clone_opt=copy.deepcopy((m,opt));assert all(cp is op for cp,op in zip(clone.parameters(),clone_opt.param_groups[0]['params']))
            before=cpu_copy((m.state_dict(),opt.state_dict()));saved_rng=rng_state(device)
            clone_opt.zero_grad(set_to_none=True)
            smallx=x[:,:32];smally=y[:32]
            loss,_=losses(clone,smallx,smally,conf['augmented_domain'],2022000002)
            loss.backward();clone_opt.step()
            assert equal(before,(m.state_dict(),opt.state_dict()))
            restore_rng(saved_rng,device);assert equal(saved_rng,rng_state(device))
            del m,opt,clone,clone_opt
            torch.cuda.empty_cache()
        results[dataset]=dict(data_identity='PASS',reference_prediction_gradient_adam='PASS',
                              global_rng_isolation='PASS',joint_deepcopy_binding_source_immutable='PASS',
                              full_batch=peaks,model_state_bytes=tensor_bytes(initial),test_metrics=False)
    return results
