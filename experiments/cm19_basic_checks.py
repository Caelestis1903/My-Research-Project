"""Finite checks of experimental semantics, not generic reliability tests."""
import copy,json
import numpy as np
import torch
import torch.nn.functional as F
from cm19_train import OUT,dump
from cm19_legacy_data import population
from cm19_models import make_model
from cm19_legacy_models import Model as Legacy
from cm19_hamur_reference import MLP_adap_2_layer_1_adp
from stage_S1_models import features_from_schema

def main():
    torch.set_num_threads(2);b=population('movielens','natural');ix=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:32] for d in range(3)])
    x=torch.tensor(b.train_x[ix],device='cuda:0').T;y=torch.tensor(b.train_y[ix],device='cuda:0');report={}
    for name in ['fixed','original','original_sample','star0','star1','star_joint','pepnet','hamur0','hamur1','hamur_joint','mmoe','ple']:
        m=make_model(b.features,2022,name).cuda().eval();p=m(x);cc=[]
        assert p.shape==(len(ix),len(m.task_ids)) and torch.isfinite(p).all()
        loss=F.binary_cross_entropy(p,y[:,m.task_ids]);loss.backward()
        for label,pp in zip(['shared','slot'],m.modulation_groups()):
            cc.append(dict(group=label,parameters=sum(q.numel() for q in pp),gradient_norm=sum(float(q.grad.norm()) for q in pp if q.grad is not None)))
        interventions={t:float((p-m(x,t)).abs().mean()) for t in m.interventions()}
        report[name]=dict(shape=list(p.shape),gradients=cc,interventions=interventions,finite=True)
        if name in ['fixed','original','original_sample','mmoe','ple']:
            old=Legacy(b.features,2022,name).cuda().eval()
            err=float((p-old(x)).abs().max());report[name]['legacy_forward_max_error']=err;assert err<2e-6
    # HAMUR optimized contraction must retain the author's original forward.
    ref=MLP_adap_2_layer_1_adp(features_from_schema(b.features),3,[256,128],[64],35).cuda().eval()
    opt=make_model(b.features,2022,'hamur0').cuda().eval();ref.load_state_dict(opt.state_dict())
    xx={i:x[i] for i in range(len(b.features.columns))};xx['domain_indicator']=x[-1]
    pr=ref(xx);po=opt(x)[:,0];err=float((pr-po).abs().max())
    report['hamur_reference_max_error']=err
    F.binary_cross_entropy(pr,y[:,0]).backward();F.binary_cross_entropy(po,y[:,0]).backward()
    graderr={n:float((p.grad-dict(opt.named_parameters())[n].grad).abs().max()) for n,p in ref.named_parameters() if p.grad is not None}
    report['hamur_reference_gradient_errors']=graderr
    report['hamur_reference_pass']=err<2e-5 and max(graderr.values())<2e-4
    # Exact zero-decay alias check, same float32 backend, including state histories.
    p=torch.nn.Parameter(torch.tensor([.1,-.4,2.],device='cuda:0'));q=torch.nn.Parameter(p.detach().clone())
    aa=torch.optim.Adam([p],lr=.003,weight_decay=0,foreach=False);aw=torch.optim.AdamW([q],lr=.003,weight_decay=0,foreach=False)
    errors=[]
    for j in range(10):
        p.grad=torch.tensor([.2+j*.01,-.5,.7],device='cuda:0');q.grad=p.grad.clone();aa.step();aw.step();errors.append(float((p-q).abs().max()))
    report['wd0_alias_max_errors']=errors;assert max(errors)==0
    report['status']='PASS' if report['hamur_reference_pass'] else 'BLOCKED_HAMUR_ASSOCIATIVITY'
    dump(OUT/'basic_checks.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
