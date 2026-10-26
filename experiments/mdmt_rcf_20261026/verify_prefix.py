"""Required small numerical checks; never a performance experiment."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .protocol import RoleBundle,derive_rng,load_config
from .prefix import make_native,configure
from .state import capture_memory_state,restore_memory_state,state_hash

def verify_native_required(config=None,device='cpu'):
    configure();data=RoleBundle('movielens');model,init=make_native(data.schema,5201,'movielens',device)
    ids=data.ids('fit')[:64];x=torch.as_tensor(np.array(data.x[ids]),device=device);y=torch.as_tensor(np.array(data.y[ids]),device=device)
    packet=model.encode(x);actual=packet.logits.sigmoid();original=model.native(x.T)
    forward_error=float((actual-original).abs().max())
    assert forward_error<=1e-6
    assert packet.h.shape==(64,256)
    loss=F.binary_cross_entropy_with_logits(packet.logits,y,reduction='sum')/(2*len(ids))
    explicit=torch.stack([F.binary_cross_entropy_with_logits(packet.logits[i,q],y[i,q]) for i in range(64) for q in range(2)]).mean()
    assert torch.allclose(loss,explicit,atol=1e-7)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    def step():
        optimizer.zero_grad(set_to_none=True);l=F.binary_cross_entropy_with_logits(model(x),y);l.backward();optimizer.step()
    step();snap=capture_memory_state({'model':model,'optimizer':optimizer},positions={'query':1,'aux':1,'dual':[1.,1.]});step();expected=state_hash({'m':model.state_dict(),'o':optimizer.state_dict()})
    positions=restore_memory_state(snap,{'model':model,'optimizer':optimizer});step();got=state_hash({'m':model.state_dict(),'o':optimizer.state_dict()})
    assert got==expected
    return dict(status='PASS',scope='native_small_real_batch',canonical_copy=init['canonical_backbone_hash']==init['copied_backbone_hash'],native_forward_max_error=forward_error,ram_resume_exact=got==expected,positions=positions,model_files_written=False,sample_mean_error=float(abs(loss.detach()-explicit.detach())))

def verify_readers_required(config):
    from scipy.special import expit
    from .readers import objective_gradient,fit_group,fit_reader
    from .baselines import DirectHead
    from .protocol import PrefixBundle,RoleView,FeatureBank
    rng=np.random.default_rng(7751);x=rng.normal(size=(120,3));b=rng.normal(size=120);y=rng.integers(0,2,120).astype(float);theta=np.r_[1.1,.2,.1,-.3,.5]
    value,grad=objective_gradient(theta,b,x,y,.1)
    numeric=np.array([(objective_gradient(theta+np.eye(len(theta))[j]*1e-5,b,x,y,.1)[0]-objective_gradient(theta-np.eye(len(theta))[j]*1e-5,b,x,y,.1)[0])/2e-5 for j in range(len(theta))])
    gradient_error=float(np.max(np.abs(grad-numeric)));assert gradient_error<1e-8
    answer,check=fit_group(b,x,y,.1,[(None,None)]*3,config)
    assert check['kkt_residual']<1e-5 and check['bounds'][2:]==[(None,None)]*3
    bounded,bc=fit_group(b,x,y,.1,[(-1,1)]*3,config);assert np.max(np.abs(bounded[2:]))<=1
    # Balanced binary fixture: all complete policies equal OFF exactly.
    class Ledger:
        def access(self,*args,**kwargs):return None
    views={};values={};columns={};bounds={}
    for role in ('P_read','P_cal','Q_explore'):
        domain=np.repeat(np.arange(3),16);group=np.tile(np.repeat(np.arange(2),8),3);y=np.tile(np.arange(2),24)
        views[role]=RoleView(np.arange(48),domain,np.stack([y,y],1).astype(np.float32),np.zeros((48,256),np.float32),np.zeros((48,2),np.float32),np.stack([group,group],1))
        for t in range(3):
            for q in range(2):values[role,t,q]=np.empty((16,0));columns[t,q]=[];bounds[t,q]=[]
    bundle=PrefixBundle('movielens',5201,views,{r:np.zeros((48,2)) for r in views},None,None,None,None,{},dict(test='synthetic_numerical_fixture'),Ledger())
    bank=FeatureBank(values,columns,bounds,dict(method='GROUP_CAL'))
    result=fit_reader(bundle,bank,'GROUP_CAL',config)
    assert result.selection['selected']=='OFF' and np.array_equal(result.probabilities['Q_explore'],expit(bundle.base_logits['Q_explore']))
    # A real cached native minibatch verifies direct output learns from exact-zero output weight.
    data=RoleBundle('movielens');model,_=make_native(data.schema,5201,'movielens','cpu');ids=data.ids('fit')[:64]
    with torch.no_grad():packet=model.encode(torch.as_tensor(np.array(data.x[ids])));raw=torch.cat([packet.h,packet.logits],1)
    mean=raw.mean(0).numpy();std=np.maximum(raw.std(0,unbiased=False).numpy(),1e-3)
    head=DirectHead(mean,std,19,float(data.y[ids,0].mean()));opt=torch.optim.Adam(head.parameters(),lr=.001)
    yy=torch.as_tensor(np.array(data.y[ids,0]));loss=F.binary_cross_entropy_with_logits(head(raw),yy);loss.backward();out_grad=float(head.output.weight.grad.norm());hidden_first=float(head.hidden.weight.grad.norm());opt.step();opt.zero_grad(set_to_none=True);F.binary_cross_entropy_with_logits(head(raw),yy).backward();hidden_second=float(head.hidden.weight.grad.norm())
    assert out_grad>0 and hidden_first==0 and hidden_second>0
    return dict(status='PASS',scope='required_numerical_and_real_direct_batch',reader_gradient_max_error=gradient_error,unbounded_full_coefficients=True,kkt_residual=check['kkt_residual'],bounded_coefficients=True,whole_six_head_off_exact=True,direct_real_batch=dict(output_first_gradient=out_grad,hidden_first_gradient=hidden_first,hidden_second_gradient=hidden_second),new_performance='NOT_MEASURED',model_files_written=False)
