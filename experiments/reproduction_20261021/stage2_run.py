#!/usr/bin/env python
"""Single reviewed run; outputs always live in a new attempt directory."""
import argparse,csv,importlib,os,sys,time,traceback,platform,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE)]
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from data import load_data,sha_file
from stage2_protocol import *

def parser():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['movielens','kuairand'],default='movielens'); p.add_argument('--arm',choices=['Gtau0','Dconst01'],default='Gtau0'); p.add_argument('--seed',type=int,default=2022); p.add_argument('--device',default='cuda:0'); p.add_argument('--output-root',default='/home_new/xzf_m3oe_reproduction_20261021'); p.add_argument('--epochs',type=int,default=50); p.add_argument('--prepare-data',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--max-steps',type=int,default=0); p.add_argument('--threads',type=int,default=4); p.add_argument('--tau',type=float,default=None); return p

def csv_append(p,row):
    exists=p.exists()
    with p.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists: w.writeheader()
        w.writerow(row)
def source_receipt():
    paths=list((ROOT/'experiments/data_generate/MDMTRec').rglob('*.py'))+list((ROOT/'experiments/grad/MDMTRec').rglob('*.py'))+list(HERE.glob('*.py'))
    return {str(p.resolve()):sha_file(p) for p in sorted(paths)}
def loaders(b,seed):
    gens={n:torch.Generator().manual_seed(seed+offset) for n,offset in [('main',11000),('aux',22000),('val',33000),('test',44000)]}
    def make(split,shuffle,g):
        x=getattr(b,split+'_x'); y=getattr(b,split+'_y')
        return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=40960,shuffle=shuffle,drop_last=False,num_workers=0,generator=gens[g])
    return make('train',True,'main'),make('train',True,'aux'),make('val',False,'val'),make('test',False,'test'),gens

def is_formal(args):
    return not args.smoke and args.epochs==50 and args.max_steps==0 and args.seed in (2022,2023,2024,2025)

def run(args):
    metadata=control_metadata(args.arm)
    fixed_tau=metadata['effective_tau'] if metadata['effective_tau'] is not None else .5
    if args.tau is not None and args.tau!=fixed_tau: raise ValueError('Stage2 tau is fixed by the reviewed control')
    args.tau=fixed_tau
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    b,manifest=load_data(args.dataset,Path(args.output_root)/'cache')
    if args.prepare_data:
        print(json.dumps({'dataset':args.dataset,'counts':b.train_counts,'raw_sha256':manifest['raw_sha256']}),flush=True); return
    formal=is_formal(args)
    parent=Path(args.output_root)/('stage2' if formal else 'stage2_pipeline')/args.dataset/args.arm/f'seed{args.seed}'
    parent.mkdir(parents=True,exist_ok=True)
    attempt=parent/time.strftime('attempt_%Y%m%d_%H%M%S')
    attempt.mkdir(exist_ok=False)
    class Tee:
        def __init__(self,stream,path): self.stream=stream; self.f=path.open('a',buffering=1)
        def write(self,s): self.stream.write(s); self.f.write(s)
        def flush(self): self.stream.flush(); self.f.flush()
    sys.stdout=Tee(sys.stdout,attempt/'stdout.log'); sys.stderr=Tee(sys.stderr,attempt/'stderr.log')
    print('ATTEMPT',attempt,flush=True)
    json_write(attempt/'data_manifest.json',manifest)
    np.savez_compressed(attempt/'split_row_ids.npz',train=b.train_row_ids,validation=b.val_row_ids,test=b.test_row_ids)
    model,init=make_model(b,args.arm,args.seed,args.device); optimizer,auxopt=optimizers(model,args.arm)
    references=parent_reference(args.output_root,args.dataset,args.arm,args.seed,init,manifest)
    train,aux_loader,val,test,gens=loaders(b,args.seed); aux_iter=iter(aux_loader)
    scarce=min(b.train_counts,key=lambda d:(b.train_counts[d],d))
    config=vars(args).copy(); config.update(formal=formal,protocol='ported-train-only',batch_size=40960,lr=.01,auxiliary_lr=.001 if auxopt else None,weight_decay=1e-5,adam_betas=[.9,.999],adam_eps=1e-8,objective='sum_t,d mean_i BCE; D/G mean_i(weight_i*BCE); no six-cell division',fusion='learned-sigmoid-double-Adam' if args.arm=='O' else 'fixed1',scarce_domain=scarce,noise_std=.1,noise_target='all flattened embeddings of copied samples including domain embedding',selection='strict validation macro AUC; earliest tie; patience10; always restore best',initialization=init,sources=source_receipt(),torch_version=torch.__version__,python_version=platform.python_version(),cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,tf32=False,amp=False,optimizer_parameter_names=[n for n,p in model.named_parameters()],aux_optimizer_parameter_names=[n for n,p in model.named_parameters() if n.startswith('_weight_')] if auxopt else [],aux_loader='persistent shuffled training iterator with isolated generator',source_pairing='grad model paired-from-supplied-model / ported')
    config.update(metadata); config.update(tau=metadata['effective_tau'],parent_references=references,stage2_design_sha256=sha_file(HERE/'DESIGN_STAGE2.md'),objective='sum_t,d mean_i(0.1*BCE_i), no six-cell division' if args.arm=='Dconst01' else 'sum_t,d mean_i((0.1+0.5*q_i)*BCE_i), detached mean real weight on copies; G gradient path tau0')
    json_write(attempt/'config.json',config)
    if str(args.device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(args.device)
    best=-float('inf'); best_epoch=0; bad=0; step=0; started=time.monotonic(); status='epoch_limit'
    try:
        for epoch in range(1,args.epochs+1):
            model.train(); objective=0.; real_sums=np.zeros((2,3)); real_counts=np.zeros(3); epoch_start=time.monotonic()
            for batch_index,(xb,yb) in enumerate(train):
                x=xb.T.to(args.device); y=yb.to(args.device)
                cells,info=batch_loss(model,x,y,args.arm,scarce,args.seed*1000000+step)
                mechanism=None; gs=None
                if batch_index==0:
                    mechanism,gs=probe(model,cells,info,args.tau if args.arm=='Gtau0' else 0.)
                    before=[p.detach().clone() for p in model.parameters()]
                apply_backward(cells,model,args.arm,args.tau,gs)
                if mechanism is not None: raw=[p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
                optimizer.step(); step+=1; objective+=sum(float(c.detach()) for c in cells)
                n=info['n']; dom=info['domain'][:n]; real=info['bce'][:n].detach()
                for d in range(3):
                    m=dom==d; real_counts[d]+=int(m.sum())
                    for t in range(2): real_sums[t,d]+=float(real[m,t].sum())
                if mechanism is not None:
                    delta=[p.detach()-old for p,old in zip(model.parameters(),before)]
                    mechanism.update(epoch=epoch,step=step,adam_delta_norm=float(norm(delta)),adam_delta_raw_cosine=cosine(delta,raw),adam_m_norm=float(torch.stack([s['exp_avg'].square().sum() for s in optimizer.state.values()]).sum().sqrt()),adam_v_norm=float(torch.stack([s['exp_avg_sq'].square().sum() for s in optimizer.state.values()]).sum().sqrt()))
                    mechanism.update(auxiliary_parameters=auxiliary_receipt(model,optimizer))
                    csv_append(attempt/'mechanism_metrics.csv',{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in mechanism.items()})
                    del gs,before,raw,delta
                del cells,info,x,y
                if args.max_steps and batch_index+1>=args.max_steps: break
            aux_loss=None
            if auxopt:
                try: ax,ay=next(aux_iter)
                except StopIteration: aux_iter=iter(aux_loader); ax,ay=next(aux_iter)
                model.zero_grad(set_to_none=True); ap=model(ax.T.to(args.device)); al=F.binary_cross_entropy(ap,ay.to(args.device)); al.backward(); auxopt.step(); aux_loss=float(al.detach())
            vm=evaluate(model,val,args.device)
            row={'epoch':epoch,'step':step,'train_optimized_group_sum':objective/(batch_index+1),'epoch_seconds':time.monotonic()-epoch_start,'auxiliary_pooled_bce':aux_loss}
            for t in range(2):
                for d in range(3): row[f'train_real_d{d}t{t}_bce']=float(real_sums[t,d]/real_counts[d]) if real_counts[d] else float('nan')
            row.update({'val_'+k:v for k,v in vm.items()}); row.update({'fusion_'+k:v for k,v in fusion(model,args.arm).items()})
            row['is_best']=int(vm['macro_auc']>best); csv_append(attempt/'epoch_metrics.csv',row)
            json_write(attempt/'last_epoch_metrics.json',row)
            print(json.dumps({'epoch':epoch,'steps':step,'validation_macro_auc':vm['macro_auc'],'validation_macro_logloss':vm['macro_logloss'],'seconds':row['epoch_seconds']}),flush=True)
            if vm['macro_auc']>best:
                best=vm['macro_auc']; best_epoch=epoch; bad=0
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),aux_optimizer=auxopt.state_dict() if auxopt else None,rng=rng_state(),loader_generators={k:g.get_state() for k,g in gens.items()},epoch=epoch,step=step,validation=vm,stage2=metadata,auxiliary_parameters=auxiliary_receipt(model,optimizer))
                torch.save(state,attempt/'best.pt'); del state
            else: bad+=1
            if bad>=10: status='earlystop'; break
        checkpoint=torch.load(attempt/'best.pt',map_location=args.device,weights_only=False); model.load_state_dict(checkpoint['model']); selected_auxiliary=checkpoint['auxiliary_parameters']; del checkpoint
        tm=evaluate(model,test,args.device); csv_append(attempt/'test_metrics.csv',{'selected_epoch':best_epoch,**tm})
        completion=dict(status=status,valid=True,formal=formal,dataset=args.dataset,arm=args.arm,seed=args.seed,selected_epoch=best_epoch,stopped_epoch=epoch,steps=step,wall_seconds=time.monotonic()-started,best_validation_macro_auc=best,peak_memory_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else None,checkpoint=str(attempt/'best.pt'),checkpoint_sha256=sha_file(attempt/'best.pt'),attempt=str(attempt),test_metrics=tm)
        completion.update(metadata); completion.update(auxiliary_parameters_selected=selected_auxiliary,parent_references=references)
        json_write(attempt/'completion.json',completion); print('COMPLETED',json.dumps(completion),flush=True)
    except BaseException as error:
        json_write(attempt/'completion.json',dict(valid=False,status='failed',stage='stage2',control_id=args.arm,parent_arm=metadata['parent_arm'],error=repr(error),steps=step,wall_seconds=time.monotonic()-started,attempt=str(attempt))); traceback.print_exc(); raise
if __name__=='__main__': run(parser().parse_args())
