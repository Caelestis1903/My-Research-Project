"""Cross-model path study. See frozen DESIGN_SPECIFICATION; no training logic."""
import copy, math
import torch
from torch import nn
import torch.nn.functional as F
from stage_S1_models import build_model, features_from_schema
from MDMTRec.basic.layers import EmbeddingLayer
from cm19_legacy_models import Model as LegacyModel
from cm19_hamur_reference import MLP_adap_2_layer_1_adp


def mlp(dims, norm=False):
    ls=[]
    for a,b in zip(dims[:-2],dims[1:-1]):
        ls.append(nn.Linear(a,b))
        if norm: ls.append(nn.LayerNorm(b))
        ls.append(nn.ReLU())
    ls.append(nn.Linear(dims[-2],dims[-1]))
    return nn.Sequential(*ls)

class PartitionNorm(nn.Module):
    def __init__(self,width):
        super().__init__();self.shared_scale=nn.Parameter(torch.ones(width));self.domain_scale=nn.Parameter(torch.ones(3,width))
        self.shared_shift=nn.Parameter(torch.zeros(width));self.domain_shift=nn.Parameter(torch.zeros(3,width))
        self.register_buffer('mean',torch.zeros(3,width));self.register_buffer('var',torch.ones(3,width))
    def forward(self,x,d):
        out=torch.empty_like(x)
        for k in range(3):
            ix=d==k;xx=x[ix]
            if not len(xx):continue
            if self.training:
                mu=xx.mean(0);va=xx.var(0,unbiased=False)
                with torch.no_grad():self.mean[k].lerp_(mu,.1);self.var[k].lerp_(va*(len(xx)/max(len(xx)-1,1)),.1)
            else:mu=self.mean[k];va=self.var[k]
            out[ix]=(xx-mu)/torch.sqrt(va+1e-5)*(self.shared_scale*self.domain_scale[k])+self.shared_shift+self.domain_shift[k]
        return out

class StarLayer(nn.Module):
    def __init__(self,i,o):
        super().__init__();self.shared=nn.Parameter(torch.empty(i,o));self.slot=nn.ParameterList([nn.Parameter(torch.empty(i,o)) for _ in range(3)])
        self.bias=nn.Parameter(torch.zeros(o));self.domain_bias=nn.ParameterList([nn.Parameter(torch.zeros(o)) for _ in range(3)])
        nn.init.xavier_uniform_(self.shared)
        for p in self.slot:nn.init.xavier_uniform_(p)
    def forward(self,x,d,variant,off):
        y=torch.empty(len(x),len(self.bias),device=x.device)
        for k in range(3):
            ix=d==k
            w=self.shared+self.slot[k] if variant=='additive' else self.shared*self.slot[k]
            y[ix]=x[ix]@(w*0 if off else w)+self.bias+self.domain_bias[k]
        return y

class STAR(nn.Module):
    def __init__(self,schema,tasks,variant):
        super().__init__();self.features=features_from_schema(schema);self.embedding=EmbeddingLayer(self.features);self.tasks=tasks;self.variant=variant
        w=sum(f.embed_dim for f in self.features);self.pn=PartitionNorm(w)
        self.layers=nn.ModuleList([StarLayer(w,256),StarLayer(256,128),StarLayer(128,len(tasks))])
        self.aux=mlp([w,64,len(tasks)]);self.register_buffer('bypass_scale',torch.tensor(.1 if variant=='bypass_scale' else 1.))
        self.last={}
    def forward(self,x,neutral=None):
        d=x[-1].long();e=self.embedding(x,self.features,squeeze_dim=True);h=self.pn(e,d)
        acts=[];off=neutral=='modulation' or self.variant=='no_modulation'
        for i,l in enumerate(self.layers):
            h=l(h,d,self.variant,off)
            if i<len(self.layers)-1:h=F.relu(h)
            acts.append(h.detach())
        aux=self.aux(e)*self.bypass_scale
        if self.variant=='no_bypass' or neutral=='bypass':aux=aux*0
        if neutral=='main':h=h*0
        self.last={'main_rms':float(h.detach().square().mean().sqrt()),'bypass_rms':float(aux.detach().square().mean().sqrt())}
        return torch.sigmoid(h+aux)
    def interventions(self):return ['modulation','main','bypass']
    def modulation_groups(self):
        return [p for l in self.layers for p in [l.shared]],[p for l in self.layers for p in l.slot]
    def strengths(self):
        r={**self.last}
        for i,l in enumerate(self.layers):
            r[f'layer{i}_shared_norm']=float(l.shared.detach().norm())
            for k in range(3):
                r[f'layer{i}_slot{k}_norm']=float(l.slot[k].detach().norm())
                r[f'layer{i}_product{k}_norm']=float((l.shared.detach()*l.slot[k].detach()).norm())
        return r

class M3OE(LegacyModel):
    def __init__(self,schema,seed,method,variant):
        super().__init__(schema,seed,method);self.variant=variant;self.last={}
        self.register_buffer('bypass_scale',torch.tensor(.1 if variant=='bypass_scale' else 1.))
    def forward(self,x,neutral=None):
        m=self.base;d=x[-1].long();e=m.embedding(x,m.features,squeeze_dim=True)
        skip=m.skip_conn(e)*self.bypass_scale
        if self.variant=='no_bypass' or neutral=='bypass':skip=skip*0
        value=torch.empty(len(d),m.star_dim[1],device=x.device)
        for k in range(3):
            ix=d==k;w=m.shared_weight+m.slot_weight[k] if self.variant=='additive' else m.shared_weight*m.slot_weight[k]
            if self.variant=='no_modulation' or neutral=='modulation':w=w*0
            value[ix]=e[ix]@w+m.slot_bias[k]+m.shared_bias
        main=m.star_mlp(value)
        if neutral=='main':main=main*0
        emb=main+skip;self.last={'main_rms':float(main.detach().square().mean().sqrt()),'bypass_rms':float(skip.detach().square().mean().sqrt())}
        shared=torch.stack([f(emb) for f in m.expert],1);dex=torch.stack([f(emb) for f in m.domain_expert],1);tex=torch.stack([f(emb) for f in m.task_expert],1)
        out=torch.empty(len(d),2,device=x.device)
        for t in range(2):
            tw=m._weight_bal_t()*tex[:,t]+(1-m._weight_bal_t())*tex[:,1-t]
            for k in range(3):
                ix=t*3+k;where=d==k;gate=m.gate[ix](emb.detach())
                sh=(gate[:,:,None]*shared).sum(1)
                dm=m._weight_bal_d()*dex[:,k]+(1-m._weight_bal_d())/2*(dex[:,(k+1)%3]+dex[:,(k+2)%3])
                out[where,t]=torch.sigmoid(m.tower[ix]((sh+m._weight_exp_d()*dm+m._weight_exp_t()*tw)[where]).squeeze(1))
        return out
    def interventions(self):return ['modulation','main','bypass']
    def modulation_groups(self):return [self.base.shared_weight],list(self.base.slot_weight)
    def strengths(self):
        m=self.base;r={**self.last,'shared_norm':float(m.shared_weight.detach().norm())}
        for k in range(3):r[f'slot{k}_norm']=float(m.slot_weight[k].detach().norm());r[f'product{k}_norm']=float((m.shared_weight.detach()*m.slot_weight[k].detach()).norm())
        return r

class GateNU(nn.Module):
    def __init__(self,i,o):super().__init__();self.net=mlp([i,64,o])
    def forward(self,x):return 2*torch.sigmoid(self.net(x))

class PEPNet(nn.Module):
    def __init__(self,schema,variant):
        super().__init__();self.features=features_from_schema(schema);self.embedding=EmbeddingLayer(self.features);self.variant=variant
        w=sum(f.embed_dim for f in self.features);self.domain=nn.Embedding(3,16)
        # Available prior fields, not labels; IDs absent in KR are not invented.
        self.prior_indices=[i for i,c in enumerate(schema.columns) if c!='domain_indicator'];self.prior_features=[self.features[i] for i in self.prior_indices]
        prior_w=sum(f.embed_dim for f in self.prior_features);self.ep=GateNU(w+16,w)
        dims=[w,256,128,1];self.towers=nn.ModuleList([nn.ModuleList([nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:])]) for _ in range(2)])
        self.pp=nn.ModuleList([GateNU(prior_w+w,2*a) for a in dims[:-1]]);self.last={}
    def forward(self,x,neutral=None):
        e=self.embedding(x,self.features,squeeze_dim=True);eg=self.ep(torch.cat([self.domain(x[-1].long()),e.detach()],1))
        ep_off=neutral in ['ep','both'] or self.variant in ['ep_neutral','both_neutral']
        pp_off=neutral in ['pp','both'] or self.variant in ['pp_neutral','both_neutral']
        z=e*(torch.ones_like(eg) if ep_off else eg)
        prior=self.embedding(x,self.prior_features,squeeze_dim=True);ginput=torch.cat([prior,z.detach()],1)
        hh=[z,z];deviations=[]
        for i,g in enumerate(self.pp):
            gates=g(ginput).chunk(2,1);deviations.extend([(a.detach()-1).abs().mean() for a in gates])
            for t in range(2):
                hh[t]=self.towers[t][i](hh[t]*(torch.ones_like(gates[t]) if pp_off else gates[t]))
                if i<2:hh[t]=F.relu(hh[t])
        self.last={'ep_abs_from1':float((eg.detach()-1).abs().mean()),'ep_std':float(eg.detach().std()),'pp_abs_from1':float(torch.stack(deviations).mean())}
        return torch.sigmoid(torch.cat(hh,1))
    def interventions(self):return ['ep','pp','both']
    def modulation_groups(self):return list(self.ep.parameters()),list(self.pp.parameters())
    def strengths(self):return self.last

class HAMUR(MLP_adap_2_layer_1_adp):
    """Author-code math with associative contraction and explicit interventions.

    Source batch-stat DN and shared factors retained; not paper-perfect DN.
    """
    def __init__(self,schema,tasks,variant):
        super().__init__(features_from_schema(schema),3,[256,128],[64],35)
        self.tasks=tasks;self.variant=variant;self.last={}
        if len(tasks)==2:
            for ls in self.layer_list:ls[6]=nn.Linear(128,2)
        if variant=='static':self.static_core=nn.Parameter(torch.eye(self.k).repeat(3,1,1))
        if variant=='private_hyper':self.private_hyper=nn.ModuleList([copy.deepcopy(self.hyper_net) for _ in range(3)])
    def forward(self,x,neutral=None):
        d=x[-1].long();e=self.embedding(x,self.features,squeeze_dim=True);result=torch.empty(len(d),len(self.tasks),device=x.device)
        rms=[];cores=[]
        for k,ls in enumerate(self.layer_list):
            h=ls[2](ls[1](ls[0](e)));h=ls[5](ls[4](ls[3](h)))
            if self.variant=='static':core=self.static_core[k].expand(len(e),-1,-1)
            else:core=(self.private_hyper[k](e) if self.variant=='private_hyper' else self.hyper_net(e)).reshape(-1,self.k,self.k)
            # Equivalent to h @ (U @ core @ V), avoids per-instance full matrices.
            z=torch.bmm((h@self.u[0]).unsqueeze(1),core).squeeze(1)@self.v[0]+self.b_list[0];z=torch.sigmoid(z)
            z=torch.bmm((z@self.u[1]).unsqueeze(1),core).squeeze(1)@self.v[1]+self.b_list[1]
            z=(z-z.mean(0))/torch.sqrt(z.var(0)+self.eps);z=self.gamma1*z+self.bias1
            if neutral=='adapter' or self.variant=='no_adapter':z=z*0
            rms.append(z.detach().square().mean().sqrt());cores.append(core.detach().square().mean().sqrt())
            pp=torch.sigmoid(ls[6](h+z));result[d==k]=pp[d==k]
        self.last={'adapter_rms':float(torch.stack(rms).mean()),'hyper_core_rms':float(torch.stack(cores).mean())}
        return result
    def interventions(self):return ['adapter']
    def modulation_groups(self):return list(self.hyper_net.parameters()),list(self.u)+list(self.v)
    def strengths(self):return self.last

class Experts(nn.Module):
    def __init__(self,schema,seed,name,variant):
        super().__init__();self.base=build_model(schema,3,2,name,seed=seed);self.name=name;self.variant=variant;self.last={}
        if variant in ['private','private_matched']:
            w=self.base.input_dim
            width=(170 if name=='mmoe' else 426) if variant=='private_matched' else 256
            self.private=nn.ModuleList([mlp([w,width,64],True) for _ in range(6)])
    def forward(self,x,neutral=None):
        m=self.base;e=m.embedding(x,m.features,squeeze_dim=True);d=x[-1].long();out=torch.empty(len(d),2,device=x.device);usage=[]
        if hasattr(self,'private'):outputs=[f(e) for f in self.private]
        elif self.name=='mmoe':
            sh=torch.stack([f(e) for f in m.core.experts],1)
            if neutral=='shared':sh=sh*0
            outputs=[]
            for i in range(6):
                g=torch.softmax(m.core.gate[i](e),1);outputs.append((g[:,:,None]*sh).sum(1));usage.append(g.detach().mean(0))
        else:
            sh=torch.stack([f(e) for f in m.core.shared_experts],1)
            if neutral=='shared':sh=sh*0
            outputs=[]
            for i in range(6):
                pr=torch.stack([f(e) for f in m.core.specific_experts[i]],1);g=torch.softmax(m.core.gate[i](e),1)
                outputs.append((g[:,:,None]*torch.cat([pr,sh],1)).sum(1));usage.append(g[:,1:].detach().sum(1).mean().view(1))
        for t in range(2):
            for k in range(3):
                i=t*3+k;ix=d==k;out[ix,t]=torch.sigmoid(m.tower[i](outputs[i][ix]).squeeze(1))
        self.last={'shared_usage':float(torch.cat(usage).mean()) if usage else 0.}
        return out
    def interventions(self):return ['shared']
    def modulation_groups(self):
        c=self.base.core;ls=c.experts if self.name=='mmoe' else c.shared_experts
        return list(ls.parameters()),[]
    def strengths(self):return self.last


def make_model(schema,seed,name,variant='native'):
    torch.manual_seed(seed)
    if name in ['star_private_joint','hamur_private_joint']:m=IndependentJoint(schema,seed,name.split('_')[0]);tasks=[0,1]
    elif name in ['fixed','original','original_sample']:m=M3OE(schema,seed,name,variant);tasks=[0,1]
    elif name.startswith('star'):
        tasks=[0,1] if name=='star_joint' else [int(name[-1])];m=STAR(schema,tasks,variant)
    elif name.startswith('hamur'):
        tasks=[0,1] if name=='hamur_joint' else [int(name[-1])];m=HAMUR(schema,tasks,variant)
    elif name=='pepnet':tasks=[0,1];m=PEPNet(schema,variant)
    else:tasks=[0,1];m=Experts(schema,seed,name,variant)
    m.task_ids=tasks
    frozen=[]
    if variant=='no_modulation':
        a,b=m.modulation_groups();frozen.extend(a+b)
    if variant=='no_bypass':frozen.extend((m.aux if isinstance(m,STAR) else m.base.skip_conn).parameters())
    if isinstance(m,PEPNet):
        if variant in ['ep_neutral','both_neutral']:frozen.extend(m.ep.parameters())
        if variant in ['pp_neutral','both_neutral']:frozen.extend(m.pp.parameters())
    if isinstance(m,HAMUR):
        if variant in ['static','private_hyper','no_adapter']:frozen.extend(m.hyper_net.parameters())
        if variant=='no_adapter':frozen.extend(list(m.u)+list(m.v)+list(m.b_list)+[m.gamma1,m.bias1])
    if isinstance(m,Experts) and hasattr(m,'private'):frozen.extend(m.base.core.parameters())
    for p in frozen:p.requires_grad_(False)
    return m

class IndependentJoint(nn.Module):
    """Two fully independent single-task networks, mean-task BCE control."""
    def __init__(self,schema,seed,family):
        super().__init__();self.models=nn.ModuleList([make_model(schema,seed,family+str(t)) for t in range(2)])
    def forward(self,x,neutral=None):return torch.cat([m(x,neutral) for m in self.models],1)
    def interventions(self):return self.models[0].interventions()
    def modulation_groups(self):
        gg=[m.modulation_groups() for m in self.models];return [p for a,b in gg for p in a],[p for a,b in gg for p in b]
    def strengths(self):return {f't{t}_{k}':v for t,m in enumerate(self.models) for k,v in m.strengths().items()}
