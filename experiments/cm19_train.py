"""Full-budget raw training/evaluation, no scientific interpretation."""
import argparse, copy, hashlib, json, math, os, random, subprocess, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss
from cm19_legacy_data import population, dump
from cm19_models import make_model

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-19/execution'
ART=Path(os.environ.get('CM19_ARTIFACT_ROOT','/home_new/xzf_cm19_artifacts'))
STEPS={0,1,5,10,25,50,100,200,500,1000,1500,2000,2350}
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def cells(p,y,d,tasks):
    records=[]
    for domain in range(3):
        ix=d==domain
        for j,t in enumerate(tasks):
            yy=y[ix,t];pp=p[ix,j];n=int(ix.sum())
            r=dict(domain=domain,task=t,n=n,positives=int(yy.sum()),positive_rate=float(yy.mean()),
                   auc=float(roc_auc_score(yy,pp)),logloss=float(log_loss(yy,pp,labels=[0,1])))
            const=np.clip(yy.mean(),1e-7,1-1e-7);base=float(-(yy*np.log(const)+(1-yy)*np.log1p(-const)).mean())
            r.update(constant_logloss=base,skill_logloss=base-r['logloss']);records.append(r)
    return records

def append(path,rows):
    if not rows:return
    path=Path(path);pd.DataFrame(rows).to_csv(path,mode='a',header=not path.exists(),index=False)

def state_hash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Runner:
    def __init__(self,c,device,smoke=False):
        self.c=c;self.device=torch.device(device);self.smoke=smoke
        torch.cuda.set_device(self.device);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats(self.device)
        seed_all(c['seed']);self.b=population(c['dataset'],'natural');self.model=make_model(self.b.features,c['seed'],c['model'],c['variant']).to(self.device)
        self.tasks=self.model.task_ids;self.name=c['run_id'];self.out=OUT/('engineering' if smoke else 'runs')/self.name;self.art=ART/('engineering' if smoke else 'runs')/self.name
        self.out.mkdir(parents=True,exist_ok=False);self.art.mkdir(parents=True,exist_ok=False)
        self.x=torch.tensor(self.b.train_x,device=self.device);self.y=torch.tensor(self.b.train_y,device=self.device)
        self.main=self.b.main.copy();self.original_n=len(self.main)
        if c.get('fraction',1.)<1:
            target=1 if c['dataset']=='kuairand' else 2;rng=np.random.default_rng(19101919)
            pool=self.main[self.b.train_x[self.main,-1]==target];pool=rng.permutation(pool)[:int(len(pool)*c['fraction'])]
            self.main=np.r_[self.main[self.b.train_x[self.main,-1]!=target],pool]
        if c['variant']=='bypass_scale':
            pool=np.concatenate([self.b.meta[self.b.train_x[self.b.meta,-1]==d][:512] for d in range(3)])
            self.model.eval()
            with torch.no_grad():
                self.model(self.x[torch.tensor(pool,device=self.device)].T);v=self.model.last
                self.model.bypass_scale.mul_(v['main_rms']/max(v['bypass_rms'],1e-12))
        self.rng=np.random.default_rng(c['seed']*1000+19)
        shared,slot=self.model.modulation_groups();shared=[p for p in shared if p.requires_grad];slot=[p for p in slot if p.requires_grad];ids_shared={id(p) for p in shared};ids_slot={id(p) for p in slot}
        assert not ids_shared&ids_slot
        other=[p for p in self.model.parameters() if p.requires_grad and id(p) not in ids_shared|ids_slot];mode=c['decay_mode']
        self.groups={'shared':shared,'slot':slot,'other':other};self.opts=[];self.group_opts={};self.group_hyper={}
        grouped={}
        for label,params in self.groups.items():
            if not params:continue
            wd=0. if mode=='wd0' or (mode in ['shared_wd0','both_wd0'] and label=='shared') or (mode in ['slot_wd0','both_wd0'] and label=='slot') else 1e-5
            cls=torch.optim.AdamW if c['optimizer']=='adamw' or mode=='local_adamw' and label!='other' else torch.optim.Adam
            group=dict(params=params,weight_decay=wd,group_name=label)
            grouped.setdefault(cls,[]).append(group)
        for cls,groups in grouped.items():
            opt=cls(groups,lr=c['lr'],betas=(.9,.999),eps=1e-8,foreach=False);self.opts.append(opt)
            for group in opt.param_groups:self.group_opts[group['group_name']]=opt;self.group_hyper[group['group_name']]=group
        self.archopt=None
        if c['model']=='original':
            aa=[p for n,p in self.model.named_parameters() if '._weight_' in n]
            self.archopt=(torch.optim.AdamW if c['optimizer']=='adamw' else torch.optim.Adam)(aa,lr=c['lr'],weight_decay=1e-5,foreach=False)
        self.probes={};probe_ids={}
        rng=np.random.default_rng(191019)
        for name,pool,x in [('meta',self.b.meta,self.b.train_x),('validation',self.b.development,self.b.val_x)]:
            ix=np.concatenate([rng.choice(pool[x[pool,-1]==d],min(512,int((x[pool,-1]==d).sum())),False) for d in range(3)])
            self.probes[name]=ix;probe_ids[name]=getattr(self.b,'train_ids' if name=='meta' else 'val_ids')[ix].tolist()
        self.steps=0;self.arch_steps=0;self.examples=0;self.best=-math.inf;self.selected=0;self.start=time.time()
        self.config={**c,'parameters':sum(p.numel() for p in self.model.parameters()),'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),'task_ids':self.tasks,'train_rows':len(self.main),'original_train_rows':self.original_n,
                     'main_ids_sha256':hashlib.sha256(self.main.tobytes()).hexdigest(),'probe_ids':probe_ids,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                     'test_metrics_read':False,'test_bytes_loaded_from_historical_pickle':True,'from_scratch':True,'dtype':'float32','optimizer_count':len(self.opts)+(self.archopt is not None),
                     'optimizer_implementation_objects':len(self.opts)+(self.archopt is not None),'nonoverlapping_main_groups':True,'state_directory':str(self.art),'original_architecture_overlap':self.archopt is not None,
                     'selection':'earliest strict maximum macro target-cell selection AUC', 'main_domain_counts':np.bincount(self.b.train_x[self.main,-1].astype(int),minlength=3).tolist(),
                     'bypass_scale':float(self.model.bypass_scale) if hasattr(self.model,'bypass_scale') else None,'optimizer_groups':{k:dict(class_name=type(v).__name__,weight_decay=self.group_hyper[k]['weight_decay'],names=[n for n,p in self.model.named_parameters() if any(p is q for q in self.groups[k])]) for k,v in self.group_opts.items()}}
        dump(self.out/'config.json',self.config)
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments').glob('cm19*.py')};dump(self.out/'source_hashes.json',sources)
        dump(self.out/'initialization.json',dict(model_state_sha256=state_hash(self.model.state_dict())))
        self.last_grad={}
        self.arch_order=np.random.default_rng(c['seed']*1000+20).permutation(self.main);self.arch_cursor=0

    def save(self,epoch,name):
        state=dict(model=self.model.state_dict(),optimizers=[o.state_dict() for o in self.opts],arch_optimizer=self.archopt.state_dict() if self.archopt else None,
                   numpy_rng=self.rng.bit_generator.state,arch_order=self.arch_order,arch_cursor=self.arch_cursor,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(self.device),python_rng=random.getstate(),numpy_global=np.random.get_state(),
                   epoch=epoch,steps=self.steps,arch_steps=self.arch_steps,examples=self.examples,best=self.best,selected=self.selected,config=self.config)
        tmp=self.art/(name+'.tmp');torch.save(state,tmp);tmp.replace(self.art/name)

    @torch.no_grad()
    def predict(self,split):
        self.model.eval();x=getattr(self.b,split+'_x');pp=[]
        for o in range(0,len(x),self.c['batch_size']):pp.append(self.model(torch.tensor(x[o:o+self.c['batch_size']],device=self.device).T).cpu().numpy())
        return np.concatenate(pp)

    @torch.no_grad()
    def diagnose(self,epoch,label):
        self.model.eval();records=[]
        for split,ix in self.probes.items():
            xx=self.b.train_x[ix] if split=='meta' else self.b.val_x[ix];yy=self.b.train_y[ix] if split=='meta' else self.b.val_y[ix]
            x=torch.tensor(xx,device=self.device).T;p=self.model(x);strength=self.model.strengths();arr={'p':p.cpu().numpy(),'labels':yy,'domains':xx[:,-1],'row_indices':ix}
            for mode in self.model.interventions():
                q=self.model(x,neutral=mode);dp=(p-q).abs();lp=torch.logit(p.clamp(1e-7,1-1e-7));lq=torch.logit(q.clamp(1e-7,1-1e-7))
                arr['neutral_'+mode]=q.cpu().numpy();cc=cells(p.cpu().numpy(),yy,xx[:,-1],self.tasks);nn=cells(q.cpu().numpy(),yy,xx[:,-1],self.tasks)
                for j,(a,b) in enumerate(zip(cc,nn)):
                    d=a['domain'];t=a['task'];ti=self.tasks.index(t);delta=dp[x[-1]==d,ti]
                    records.append(dict(epoch=epoch,step=self.steps,arch_steps=self.arch_steps,examples=self.examples,label=label,probe=split,intervention=mode,domain=d,task=t,n=len(delta),
                        delta_p=float(delta.mean()),p50=float(delta.median()),p95=float(torch.quantile(delta,.95)),max_delta=float(delta.max()),nonzero=int((delta>0).sum()),
                        delta_logit=float((lp[x[-1]==d,ti]-lq[x[-1]==d,ti]).abs().mean()),auc=a['auc'],neutral_auc=b['auc'],delta_auc=b['auc']-a['auc'],logloss=a['logloss'],neutral_logloss=b['logloss'],delta_logloss=b['logloss']-a['logloss']))
            np.savez_compressed(self.art/f'probe_{label}_{split}.npz',**arr)
            append(self.out/'path_strength.csv',[dict(epoch=epoch,step=self.steps,examples=self.examples,label=label,probe=split,**strength)])
        append(self.out/'path_diagnostics.csv',records)
        append(self.out/'gradient_updates.csv',[dict(epoch=epoch,step=self.steps,label=label,**g) for g in self.last_grad.values()])
        if label!='selected':self.save(epoch,f'{label}.pt')

    def objective(self,p,y,d):
        yy=y[:,self.tasks]
        if self.c['model']=='original':return sum(F.binary_cross_entropy(p[d==k,j],yy[d==k,j]) for k in range(3) for j in range(len(self.tasks)) if (d==k).any())
        return F.binary_cross_entropy(p,yy)

    def update(self,rr):
        rows=torch.tensor(rr,device=self.device);x=self.x[rows].T;y=self.y[rows];self.model.train()
        for opt in self.opts:opt.zero_grad(set_to_none=True)
        p=self.model(x);loss=self.objective(p,y,x[-1].long());assert torch.isfinite(loss),'nonfinite loss';loss.backward()
        observe=(self.steps+1 in STEPS)
        if observe:
            self.last_grad={};self.counterfactual(rr)
            before={id(p):p.detach().clone() for g in self.groups.values() for p in g if p.grad is not None}
            for label,params in self.groups.items():
                opt=self.group_opts.get(label)
                if not opt:continue
                pp=[p for p in params if p.grad is not None];wd=self.group_hyper[label]['weight_decay']
                self.last_grad[label]=dict(group=label,gradient_observation_step=self.steps+1,data_grad_norm=math.sqrt(sum(float(p.grad.square().sum()) for p in pp)),
                    wd_vector_norm=wd*math.sqrt(sum(float(p.detach().square().sum()) for p in pp)),
                    m_norm=math.sqrt(sum(float(opt.state[p].get('exp_avg',torch.zeros((),device=self.device)).square().sum()) for p in pp)),
                    v_norm=math.sqrt(sum(float(opt.state[p].get('exp_avg_sq',torch.zeros((),device=self.device)).square().sum()) for p in pp)),
                    active_parameters=sum(p.numel() for p in pp),weight_decay=wd,optimizer=type(opt).__name__)
        for opt in self.opts:opt.step()
        if observe:
            for label,params in self.groups.items():
                if label in self.last_grad:self.last_grad[label]['actual_update_norm']=math.sqrt(sum(float((p.detach()-before[id(p)]).square().sum()) for p in params if id(p) in before))
        self.steps+=1;self.examples+=len(rr)
        self.online_pred=p.detach().cpu().numpy()
        return float(loss.detach())

    @torch.no_grad()
    def counterfactual(self,rr):
        # Common parameters, common data gradients and historical moments; only target groups move.
        if self.c['model'] not in ['fixed','star0','star1']:return
        target=[(label,p) for label in ['shared','slot'] for p in self.groups[label] if p.grad is not None]
        original={id(p):p.detach().clone() for _,p in target};self.model.eval()
        ix=self.probes['meta'];x=self.x[torch.tensor(ix,device=self.device)].T;y=self.y[torch.tensor(ix,device=self.device)]
        p0=self.model(x).clone();records=[]
        for kind in ['adam_wd','adam_zero','adamw']:
            for label,p in target:
                p.copy_(original[id(p)]);opt=self.group_opts[label];state=opt.state[p];step=int(state.get('step',0))+1
                m=state.get('exp_avg',torch.zeros_like(p));v=state.get('exp_avg_sq',torch.zeros_like(p));wd=1e-5 if kind!='adam_zero' else 0.
                grad=p.grad+(wd*p if kind=='adam_wd' else 0.)
                mm=.9*m+.1*grad;vv=.999*v+.001*grad.square()
                update=self.c['lr']*(mm/(1-.9**step))/(torch.sqrt(vv/(1-.999**step))+1e-8)
                if kind=='adamw':p.mul_(1-self.c['lr']*wd)
                p.sub_(update)
            pp=self.model(x);records.append(dict(step=self.steps+1,kind=kind,loss=float(self.objective(pp,y,x[-1].long())),delta_p=float((pp-p0).abs().mean())))
        for _,p in target:p.copy_(original[id(p)])
        append(self.out/'local_counterfactual.csv',records);self.model.train()

    def fit(self):
        self.diagnose(0,'step000000');n=len(self.main);base_steps=math.ceil(self.original_n/self.c['batch_size']);total_steps=self.c.get('max_steps')
        epoch=0
        while True:
            epoch+=1;start=time.time()
            if self.c.get('schedule')=='matched_updates':
                pools=[]
                for d in range(3):
                    pool=self.main[self.b.train_x[self.main,-1]==d];count=int((self.b.train_x[self.b.main,-1]==d).sum())
                    pools.append(self.rng.choice(pool,count,replace=len(pool)<count))
                order=self.rng.permutation(np.concatenate(pools))
            else:order=self.rng.permutation(self.main)
            if self.smoke:order=order[:min(len(order),self.c['batch_size']*2)]
            total=0.;examples=0;online=[]
            for offset in range(0,len(order),self.c['batch_size']):
                if total_steps and self.steps>=total_steps:break
                rr=order[offset:offset+self.c['batch_size']];val=self.update(rr);total+=val*len(rr);examples+=len(rr);online.append((self.online_pred,self.b.train_y[rr],self.b.train_x[rr,-1]))
                if self.steps in STEPS:self.diagnose(epoch,f'step{self.steps:06d}')
            if self.archopt:
                if self.arch_cursor>=len(self.arch_order):
                    self.arch_order=self.rng.permutation(self.main);self.arch_cursor=0
                rr=torch.tensor(self.arch_order[self.arch_cursor:self.arch_cursor+self.c['batch_size']],device=self.device);self.arch_cursor+=len(rr)
                self.model.train();self.model.zero_grad(set_to_none=True)
                F.binary_cross_entropy(self.model(self.x[rr].T),self.y[rr][:,self.tasks]).backward();self.archopt.step();self.arch_steps+=1
            pred=self.predict('val');records=[]
            for split,ix in [('selection',self.b.selection),('development',self.b.development)]:
                cc=cells(pred[ix],self.b.val_y[ix],self.b.val_x[ix,-1],self.tasks)
                for r in cc:r.update(epoch=epoch,split=split)
                records.extend(cc)
            op=np.concatenate([v[0] for v in online]);oy=np.concatenate([v[1] for v in online]);od=np.concatenate([v[2] for v in online]);oc=cells(op,oy,od,self.tasks)
            for r in oc:r.update(epoch=epoch,split='training_online')
            records.extend(oc)
            score=float(np.mean([r['auc'] for r in records if r['split']=='selection']));improved=score>self.best
            if improved:self.best=score;self.selected=epoch;self.save(epoch,'selected.pt')
            row=dict(epoch=epoch,objective=total/max(examples,1),steps=self.steps,arch_steps=self.arch_steps,examples=self.examples,selection_auc=score,
                development_auc=float(np.mean([r['auc'] for r in records if r['split']=='development'])),selected_epoch=self.selected,epoch_seconds=time.time()-start,elapsed_seconds=time.time()-self.start)
            append(self.out/'trajectory.csv',[row]);append(self.out/'metrics_validation.csv',records)
            self.save(epoch,'last.pt')
            if epoch in [1,2,5,10]:self.diagnose(epoch,f'epoch{epoch:03d}')
            dump(self.out/'progress.json',dict(status='TRAINING',**row));print(json.dumps(row),flush=True)
            if self.smoke or total_steps and self.steps>=total_steps or not total_steps and epoch>=self.c['epochs']:break
        self.diagnose(epoch,'final')
        state=torch.load(self.art/'selected.pt',map_location=self.device,weights_only=False);self.model.load_state_dict(state['model']);final_steps=self.steps;final_examples=self.examples;final_arch_steps=self.arch_steps
        self.steps=state['steps'];self.examples=state['examples'];self.arch_steps=state['arch_steps'];self.last_grad={};self.diagnose(self.selected,'selected')
        dump(self.out/'selection_frozen.json',dict(selected_epoch=self.selected,epochs=epoch,steps=final_steps,examples=final_examples,test_read=False))
        dump(self.out/'receipt.json',dict(status='COMPLETE_ENGINEERING' if self.smoke else 'COMPLETE_TRAINING_TEST_DEFERRED',epochs=epoch,steps=final_steps,arch_steps=final_arch_steps,examples=final_examples,
             selected_epoch=self.selected,wall_seconds=time.time()-self.start,parameters=self.config['parameters'],peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2,
             from_scratch=True,test_metrics_read=False,final_state_sha256=state_hash(torch.load(self.art/'last.pt',map_location='cpu',weights_only=False)['model'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());runner=None
    try:runner=Runner(cfg,a.device,a.smoke);runner.fit()
    except Exception:
        if runner:dump(runner.out/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()
