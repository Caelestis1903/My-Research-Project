"""N19 full from-scratch training runner. Raw outputs only; no conclusions."""
import argparse,copy,json,math,os,random,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,log_loss
from n19_data import ROOT,OUT,ART,population,dump
from n19_models import Model
from n19_objectives import NoiseChannel,clean_predictions,task_masks

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def metrics(pred,y,domains):
    records=[]
    for d in range(3):
        ix=domains==d
        for t in range(2):
            yy=y[ix,t];pp=pred[ix,t]
            records.append(dict(domain=d,task=t,n=int(ix.sum()),positives=int(yy.sum()),
                 auc=float(roc_auc_score(yy,pp)),logloss=float(log_loss(yy,pp,labels=[0,1]))))
    return records

class Runner:
    def __init__(self,a):
        self.a=a;self.run_id=f'{a.suite}__{a.dataset}__{a.method}__seed{a.seed}'
        if a.tag:self.run_id+='__'+a.tag
        self.out=OUT/'runs'/self.run_id;self.state_dir=ART/self.run_id
        self.out.mkdir(parents=True,exist_ok=a.resume);self.state_dir.mkdir(parents=True,exist_ok=a.resume)
        self.device=torch.device(a.device);torch.cuda.set_device(self.device);torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats(self.device);seed_all(a.seed);self.data=population(a.dataset,a.suite)
        self.model=Model(self.data.features,a.seed,a.method).to(self.device)
        self.channel=NoiseChannel(a.method in ['axis_channel','joint_axis']).to(self.device) if a.method in ['ind_channel','axis_channel','nested_ind','joint_axis','audit_fixed'] else None
        if a.method=='audit_fixed':
            b=self.data;idx=b.main[b.audit[b.main]];d=b.train_x[idx,-1].astype(int);rates=np.zeros((2,3,2));counts=[]
            for domain in range(3):
                for task in range(2):
                    for clean in range(2):
                        ix=(d==domain)&(b.train_y[idx,task]==clean)
                        flips=int((b.observed[idx[ix],task]!=clean).sum());n=int(ix.sum())
                        rates[clean,domain,task]=np.clip((flips+.5)/(n+1),1e-5,.35-1e-5)
                        counts.append(dict(domain=domain,task=task,clean=clean,n=n,flips=flips))
            scaled=torch.tensor(rates/.35,device=self.device,dtype=torch.float32)
            with torch.no_grad():self.channel.raw.copy_(torch.logit(scaled))
            for parameter in self.channel.parameters():parameter.requires_grad_(False)
            if not a.resume:dump(self.out/'audit_calibration.json',dict(counts=counts,a=rates[0].tolist(),b=rates[1].tolist()))
        params=list(self.model.parameters())+([p for p in self.channel.parameters() if p.requires_grad] if self.channel else [])
        self.opt=torch.optim.Adam(params,lr=a.lr,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        arch=[p for n,p in self.model.named_parameters() if '._weight_' in n]
        self.archopt=torch.optim.Adam(arch,lr=a.lr,weight_decay=1e-5) if a.method=='original' else None
        self.ema=copy.deepcopy(self.model).eval() if a.method in ['mask_copula_cv','mask_independent_cv'] else None
        if self.ema:
            for p in self.ema.parameters():p.requires_grad_(False)
        # The simulator is separated from policy observations; hidden labels are
        # zero-filled before losses and never enter copula statistics unqueried.
        b=self.data
        self.x=torch.tensor(b.train_x,device=self.device,dtype=torch.float32)
        self.y=torch.tensor(b.train_y,device=self.device,dtype=torch.float32)
        self.noisy=torch.tensor(b.observed,device=self.device,dtype=torch.float32)
        self.audit=torch.tensor(b.audit,device=self.device,dtype=torch.bool)
        self.env=torch.tensor(b.environment,device=self.device,dtype=torch.long)
        self.copula=torch.zeros(3,device=self.device);self.steps=0;self.best=-math.inf;self.selected=0;self.start_epoch=0
        self.mask_generator=torch.Generator(device=self.device).manual_seed(a.seed*1000+19)
        self.train_rng=np.random.default_rng(a.seed*1000+19)
        self.validation_p=None;self.total_labels=0;self.total_pairs=0
        self.config=dict(run_id=self.run_id,seed=a.seed,population_seed=2022,dataset=a.dataset,suite=a.suite,method=a.method,
            epochs=a.epochs,lr=a.lr,batch_size=a.batch_size,train_rows=len(b.main),train_meta_rows=len(b.meta),
            p=b.p.tolist(),source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            optimizer='Adam',optimizer_count=2 if self.archopt else 1,weight_decay=1e-5,
            original_protocol_overlap=bool(self.archopt),objective='Equal group mean sum + epoch Sample architecture update' if self.archopt else 'Sample mean-row mean-task BCE or declared suite objective',
            learned_fusion=a.method in ['original','original_sample'],fusion_fixed_sigmoid1=a.method not in ['original','original_sample','mmoe','ple'],
            parameters=sum(p.numel() for p in self.model.parameters()),noise_channel_parameters=sum(p.numel() for p in self.channel.parameters()) if self.channel else 0,
            noise_rates=b.noise_rates,selection='earliest strict maximum six-cell macro AUC on selection half; rank arms eligible only after epoch10 pruning',
            engineering=a.engineering,
            test_read=False,from_scratch=True,backbone_frozen=False,review='primary self-review under revised start.md',
            data_source=b.metadata,state_directory=str(self.state_dir))
        if a.resume:
            saved=json.loads((self.out/'config.json').read_text())
            for key in ['run_id','epochs','lr','batch_size','parameters']:assert saved[key]==self.config[key]
            state=torch.load(self.state_dir/'last.pt',map_location=self.device,weights_only=False)
            self.model.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer'])
            if self.channel:self.channel.load_state_dict(state['channel'])
            if self.archopt:self.archopt.load_state_dict(state['arch_optimizer'])
            if self.ema:self.ema.load_state_dict(state['ema'])
            self.copula.copy_(state['copula']);self.train_rng.bit_generator.state=state['sampler_rng']
            torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state(state['torch_cuda'].cpu(),self.device)
            self.mask_generator.set_state(state['mask_rng'].cpu())
            self.steps=state['steps'];self.best=state['best'];self.selected=state['selected'];self.start_epoch=state['epoch']
            self.total_labels=state['total_labels'];self.total_pairs=state['total_pairs']
            for name in ['trajectory.csv','metrics_validation.csv']:
                path=self.out/name
                if path.exists():
                    frame=pd.read_csv(path);frame[frame.epoch<=self.start_epoch].to_csv(path,index=False)
            self.config=saved
        else:
            dump(self.out/'config.json',self.config)
            source=self.out/'source_snapshot';source.mkdir()
            for name in ['n19_data.py','n19_models.py','n19_objectives.py','n19_train.py']:
                shutil.copyfile(ROOT/'experiments'/name,source/name)
        self.started=time.time()

    def save(self,epoch,name):
        state=dict(epoch=epoch,model=self.model.state_dict(),optimizer=self.opt.state_dict(),
            channel=self.channel.state_dict() if self.channel else None,ema=self.ema.state_dict() if self.ema else None,
            arch_optimizer=self.archopt.state_dict() if self.archopt else None,copula=self.copula,
            sampler_rng=self.train_rng.bit_generator.state,mask_rng=self.mask_generator.get_state(),
            torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(self.device),steps=self.steps,
            best=self.best,selected=self.selected,total_labels=self.total_labels,total_pairs=self.total_pairs)
        temp=self.state_dir/(name+'.tmp');torch.save(state,temp);temp.replace(self.state_dir/name)

    @torch.no_grad()
    def predict(self,split):
        b=self.data;x=getattr(b,split+'_x');outputs=[];self.model.eval()
        for start in range(0,len(x),self.a.batch_size):
            xx=torch.tensor(x[start:start+self.a.batch_size],device=self.device,dtype=torch.float32).T
            pp=self.model(xx)
            if self.a.method=='task_mask':
                masked=xx.clone();masked[-2]=0;pp=torch.stack([self.model(masked)[:,0],pp[:,1]],1)
            pp=clean_predictions(pp,self.a.method);outputs.append(pp.cpu().numpy())
        return np.concatenate(outputs)

    def supervised_loss(self,prob,rows):
        a=self.a;y=self.y[rows];domains=self.x[rows,-1].long()
        if self.channel:
            clean_audit=torch.where(self.audit[rows,None],y,torch.zeros_like(y))
            return self.channel.loss(clean_predictions(prob,a.method),self.noisy[rows],clean_audit,self.audit[rows],domains,a.method=='joint_axis')
        if a.method=='audit_only':
            ix=self.audit[rows];return F.binary_cross_entropy(prob[ix],y[ix]) if ix.any() else prob.sum()*0
        if a.suite=='noise' and a.method!='clean_oracle':
            y=torch.where(self.audit[rows,None],y,self.noisy[rows])
        terms=F.binary_cross_entropy(prob,y,reduction='none')
        if self.archopt:return sum(terms[domains==d,t].mean() for d in range(3) for t in range(2) if (domains==d).any())
        return terms.mean()

    def step(self,rr):
        a=self.a;rows=torch.tensor(rr,device=self.device,dtype=torch.long);x=self.x[rows].T;y=self.y[rows];d=x[-1].long()
        self.model.train();self.opt.zero_grad(set_to_none=True)
        details=a.method in ['regret_router','coalition_router'] or a.method.startswith('mask_')
        result=self.model(x,details=details);prob,info=result if details else (result,{})
        loss=self.supervised_loss(prob,rows) if not a.method.startswith('mask_') else None
        extra={}
        if a.suite=='stability':
            if a.method in ['task_cf','task_dro','global_cf']:
                cf=x.clone();cf[-2]=3-cf[-2];flipped=self.model(cf)
                if a.method=='task_cf':loss=loss+((prob[:,0]-flipped[:,0])**2).mean()
                elif a.method=='global_cf':loss=.5*(loss+F.binary_cross_entropy(flipped,y))
                else:
                    t0=torch.maximum(F.binary_cross_entropy(prob[:,0],y[:,0],reduction='none'),F.binary_cross_entropy(flipped[:,0],y[:,0],reduction='none'))
                    loss=.5*(t0.mean()+F.binary_cross_entropy(prob[:,1],y[:,1]))
            elif a.method=='task_mask':
                masked=x.clone();masked[-2]=0;p0=self.model(masked)[:,0]
                loss=.5*(F.binary_cross_entropy(p0,y[:,0])+F.binary_cross_entropy(prob[:,1],y[:,1]))
            elif a.method=='vrex':
                terms=F.binary_cross_entropy(prob[:,0],y[:,0],reduction='none');env=self.env[rows]
                groups=torch.stack([terms[env==e].mean() for e in [0,1]])
                loss=loss+groups.var(unbiased=False)
        if a.method in ['regret_router','coalition_router']:
            kl=[]
            for (domain,task,where,alternatives),(_,_,weights) in zip(info['source_probs'],info['router_weights']):
                costs=F.binary_cross_entropy(alternatives,y[where,task,None].expand_as(alternatives),reduction='none').mean(0)
                if a.method=='coalition_router':costs=-costs
                target=torch.softmax(-costs.detach()/.1,0)
                kl.append(torch.sum(target*(target.clamp_min(1e-8).log()-weights.clamp_min(1e-8).log())))
            aux=torch.stack(kl).mean();loss=loss+.1*aux;extra['router_kl']=float(aux.detach())
        if a.method.startswith('mask_'):
            mask,r=task_masks(a.method,d,self.mask_generator,self.copula)
            observed=torch.where(mask,y,torch.zeros_like(y))
            terms=F.binary_cross_entropy(prob,observed,reduction='none')
            both=mask.all(1)
            if a.method in ['mask_copula','mask_copula_cv'] and both.any():
                g0=torch.autograd.grad(terms[both,0].sum(),info['emb'],retain_graph=True)[0]
                g1=torch.autograd.grad(terms[both,1].sum(),info['emb'],retain_graph=True)[0]
                dot=(g0.detach()*g1.detach()).sum(1)
                for domain in range(3):
                    ix=d==domain
                    if ix.any():self.copula[domain].mul_(.95).add_((dot[ix]/r[ix]).mean(),alpha=.05)
            if self.ema:
                with torch.no_grad():teacher=self.ema(x)
                baseline=F.binary_cross_entropy(prob,teacher,reduction='none')
                loss=(baseline+(terms-baseline)*mask/.55).mean()
            else:loss=(terms*mask/.55).mean()
            self.total_labels+=int(mask.sum());self.total_pairs+=int(both.sum())
            extra.update(labels=int(mask.sum()),paired=int(both.sum()),copula=self.copula.detach().cpu().tolist())
        assert torch.isfinite(loss),'nonfinite loss'
        loss.backward();self.opt.step();self.steps+=1
        if self.ema:
            with torch.no_grad():
                for teacher,student in zip(self.ema.parameters(),self.model.parameters()):teacher.mul_(.99).add_(student,alpha=.01)
        return float(loss.detach()),extra

    def prune_rank(self):
        m=self.model
        if not getattr(m,'rank_mode',False) or bool(m.rank_frozen):return None
        # Fix the factor-coordinate gauge before comparing component deletion.
        # Both arms receive the SAME spectral canonicalization and moment reset.
        errors=[]
        with torch.no_grad():
            for d in range(3):
                original=m.left[d]@m.right[d]
                u,s,vh=torch.linalg.svd(original,full_matrices=False)
                root=s[:32].clamp_min(0).sqrt()
                m.left[d].copy_(u[:,:32]*root[None,:]);m.right[d].copy_(root[:,None]*vh[:32])
                errors.append(float((m.left[d]@m.right[d]-original).norm()/original.norm().clamp_min(1e-12)))
        for parameter in [m.left,m.right]:self.opt.state[parameter].clear()
        if self.a.method=='rank_uniform':mask=torch.zeros_like(m.rank_mask);mask[:,:16]=1;scores=None
        else:
            chosen=[];rng=np.random.default_rng(191903)
            for d in range(3):
                pool=self.data.meta[self.data.train_x[self.data.meta,-1]==d];chosen.extend(rng.choice(pool,min(512,len(pool)),False))
            rr=torch.tensor(chosen,device=self.device);m.eval();probe=m.rank_mask.detach().clone().requires_grad_(True)
            p=m(self.x[rr].T,rank_masks=probe);terms=F.binary_cross_entropy(p,self.y[rr],reduction='none')
            scores=torch.zeros_like(probe)
            for d in range(3):
                for t in range(2):
                    ix=self.x[rr,-1]==d;g=torch.autograd.grad(terms[ix,t].mean(),probe,retain_graph=True)[0]
                    # Deletion changes mask by -1; protect max positive risk increase.
                    scores[d]=torch.maximum(scores[d],-g[d].detach())
            mask=torch.zeros_like(probe);score=scores.cpu().numpy()
            for d in range(3):mask[d,np.argsort(-score[d],kind='stable')[:4]]=1
            ranked=sorted([(float(score[d,r]),d,r) for d in range(3) for r in range(32) if not mask[d,r]],key=lambda x:(-x[0],x[1],x[2]))
            for _,d,r in ranked[:36]:mask[d,r]=1
        m.rank_mask.copy_(mask);m.rank_frozen.fill_(True)
        return dict(active_ranks=mask.sum(1).cpu().tolist(),mask=mask.cpu().tolist(),scores=scores.cpu().tolist() if scores is not None else None,
            basis='SVD canonical factor coordinates; uniform keeps top16/domain',spectral_reconstruction_relative_errors=errors,
            factor_optimizer_moments_reset_for_both_arms=True)

    def utility_update(self):
        if self.a.method!='utility_router':return
        rng=np.random.default_rng(191904+self.steps);chosen=[]
        for d in range(3):
            pool=self.data.meta[self.data.train_x[self.data.meta,-1]==d];chosen.extend(rng.choice(pool,min(256,len(pool)),False))
        rr=torch.tensor(chosen,device=self.device);p=self.model(self.x[rr].T)
        loss=F.binary_cross_entropy(p,self.y[rr]);grad=torch.autograd.grad(loss,self.model.mix_logits)[0]
        with torch.no_grad():self.model.mix_logits.add_(grad,alpha=-.1)

    @torch.no_grad()
    def reliability_update(self,epoch):
        if self.a.method!='reliability_interaction' or epoch%5:return
        started=time.time();rng=np.random.default_rng(self.a.seed*10000+epoch);halves=[[],[]]
        for d in range(3):
            pool=self.data.meta[self.data.train_x[self.data.meta,-1]==d]
            count=min(256,len(pool)//2);chosen=rng.choice(pool,2*count,False)
            halves[0].extend(chosen[:count]);halves[1].extend(chosen[count:])
        assert not set(halves[0]).intersection(halves[1])
        batches=[torch.tensor(h,device=self.device) for h in halves]
        m=self.model;m.eval();slopes=torch.zeros((2,3,2,3),device=self.device);eps=.1
        for d in range(3):
            for t in range(2):
                for k in range(3):
                    original=m.mix_logits[d,t,k].clone();risks=[]
                    for sign in [-1,1]:
                        m.mix_logits[d,t,k].copy_(original+sign*eps)
                        pair=[]
                        for rows in batches:
                            p=m(self.x[rows].T);ix=self.x[rows,-1]==d
                            pair.append(F.binary_cross_entropy(p[ix,t],self.y[rows][ix,t]))
                        risks.append(torch.stack(pair))
                    m.mix_logits[d,t,k].copy_(original)
                    slopes[:,d,t,k]=(risks[1]-risks[0])/(2*eps)
        evidence=torch.where(slopes[0]*slopes[1]>0,1.,.25)
        m.interaction_confidence.mul_(.9).add_(evidence,alpha=.1)
        dump(self.out/f'reliability_epoch{epoch:03d}.json',dict(epoch=epoch,
            slopes=slopes.cpu().tolist(),evidence=evidence.cpu().tolist(),
            confidence=m.interaction_confidence.cpu().tolist(),half_sizes=[len(h) for h in halves],
            disjoint_train_meta=True,extra_seconds=time.time()-started))

    def fit(self):
        a=self.a;n=len(self.data.main);start=time.time()
        for epoch in range(self.start_epoch+1,a.epochs+1):
            before=time.time();order=self.train_rng.permutation(self.data.main);total=0.;extras=[]
            if a.engineering:order=order[:min(len(order),a.batch_size*2)]
            for offset in range(0,len(order),a.batch_size):
                rr=order[offset:offset+a.batch_size];value,extra=self.step(rr);total+=value*len(rr)
                if extra:extras.append(extra)
            if self.archopt:
                # Literal original implementation gives fusion both optimizers.
                rr=order[:a.batch_size];rows=torch.tensor(rr,device=self.device)
                self.model.zero_grad(set_to_none=True);F.binary_cross_entropy(self.model(self.x[rows].T),self.y[rows] if a.suite!='noise' else torch.where(self.audit[rows,None],self.y[rows],self.noisy[rows])).backward();self.archopt.step()
            self.utility_update()
            self.reliability_update(epoch)
            prune=self.prune_rank() if epoch==10 else None
            if prune:dump(self.out/'rank_allocation.json',prune)
            pred=self.predict('val');records=[]
            for label,ix in [('selection',self.data.selection),('development',self.data.development)]:
                cells=metrics(pred[ix],self.data.val_y[ix],self.data.val_x[ix,-1])
                for r in cells:r.update(epoch=epoch,split=label)
                records.extend(cells)
            select=float(np.mean([r['auc'] for r in records if r['split']=='selection']))
            dev=float(np.mean([r['auc'] for r in records if r['split']=='development']))
            eligible=not getattr(self.model,'rank_mode',False) or epoch>=10
            improved=eligible and select>self.best
            if improved:self.best=select;self.selected=epoch
            stats=dict(epoch=epoch,objective=total/len(order),selection_auc=select,development_auc=dev,
                steps=self.steps,examples=epoch*n,epoch_seconds=time.time()-before,elapsed_seconds=time.time()-start,
                selected_epoch=self.selected,parameters=self.model.deployed_parameters(),
                labels_queried=self.total_labels,paired_queries=self.total_pairs)
            self.append('metrics_validation.csv',records);self.append('trajectory.csv',[stats])
            if improved:
                self.save(epoch,'selected.pt');np.savez_compressed(self.out/'validation_selected.npz',predictions=pred,labels=self.data.val_y,domains=self.data.val_x[:,-1],row_ids=self.data.val_ids)
            if epoch in [1,5,10,20,30,40,50,100]:self.save(epoch,f'epoch{epoch:03d}.pt')
            self.save(epoch,'last.pt');dump(self.out/'progress.json',dict(status='TRAINING',**stats))
            if extras:dump(self.out/'mechanism_last.json',dict(epoch=epoch,batches=extras))
            if self.channel:
                aa,bb=self.channel.rates();dump(self.out/'noise_estimates.json',dict(epoch=epoch,a=aa.detach().cpu().tolist(),b=bb.detach().cpu().tolist()))
            print(json.dumps(stats),flush=True)
        np.savez_compressed(self.out/'validation_final.npz',predictions=pred,labels=self.data.val_y,domains=self.data.val_x[:,-1],row_ids=self.data.val_ids)
        dump(self.out/'selection_frozen.json',dict(selected_epoch=self.selected,budget=a.epochs,test_read=False))
        dump(self.out/'receipt.json',dict(status='COMPLETE_ENGINEERING' if a.engineering else 'COMPLETE_TRAINING_TEST_DEFERRED',epochs=a.epochs,steps=self.steps,
            selected_epoch=self.selected,best_selection_auc=self.best,wall_seconds=time.time()-self.started,
            peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2,parameters=self.model.deployed_parameters(),
            labels_queried=self.total_labels,pairs_queried=self.total_pairs,from_scratch=True,backbone_frozen=False,test_read=False))

    def append(self,file,rows):
        path=self.out/file;pd.DataFrame(rows).to_csv(path,mode='a',header=not path.exists(),index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['kuairand','movielens'],required=True)
    p.add_argument('--suite',choices=['natural','stability','noise','sampling'],default='natural')
    p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',required=True)
    p.add_argument('--epochs',type=int);p.add_argument('--lr',type=float);p.add_argument('--batch-size',type=int,default=40960)
    p.add_argument('--tag',default='');p.add_argument('--resume',action='store_true');p.add_argument('--engineering',action='store_true');a=p.parse_args()
    if a.epochs is None:a.epochs=50 if a.dataset=='kuairand' else 100
    if a.lr is None:a.lr=.003 if a.dataset=='kuairand' else .001
    runner=None
    try:runner=Runner(a);runner.fit()
    except Exception:
        if runner:dump(runner.out/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
