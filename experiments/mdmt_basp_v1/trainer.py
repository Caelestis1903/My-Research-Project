"""Natural-row adaptation clock, RAM state transactions and V-only selection."""
from pathlib import Path
import copy
import time
import numpy as np
import torch
from .config import rng,require_implemented
from .roles import tensor
from .prefix import packet,synchronize
from .identity import capture,restore,cpu_copy,freeze,write_json,append_jsonl,capture_rng,restore_rng
from .metrics import evaluate
from .export import write_csv,export_prediction,state_manifest,calibration,fold_ridge_skill
from .model import make_model,TargetModel
from .benefit import refresh_teacher,measure_benefits
from .losses import training_losses,project_sources
from .permissions import make_optimizer,apply_gradients


def proposal_record(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value,dict):
        return {k:proposal_record(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [proposal_record(v) for v in value]
    return value


@torch.no_grad()
def coverage_weights(model,packet,domain,proposal):
    """The F-registered coverage is fixed across all four S candidates."""
    cover=proposal['coverage'];features=torch.cat((packet.get('h0',packet['h']),packet.get('z0',packet['z'])),-1)
    weight=torch.as_tensor(cover['weight'],device=features.device,dtype=features.dtype)
    gate=(features@weight+float(cover['bias'])).sigmoid()
    if cover.get('group',0)==1:
        gate=1-gate
    rows=torch.ones(len(domain),device=domain.device,dtype=torch.bool)
    if cover.get('target_domain') is not None:
        rows &= domain==cover['target_domain']
    if cover.get('excluded_domain') is not None:
        rows &= domain!=cover['excluded_domain']
    output=model(packet,domain)
    if proposal['kind']=='NEED_SPLIT':
        d=proposal['domain'];na=len(model.action_maps[d])
        if model.needs[d].fixed:
            profile=packet.get('h0',packet['h'])@model.fixed_projection
            centers=proposal['centers'].to(profile)
            distance=torch.cdist(profile,centers).square()
        else:
            profile=output.gain_prediction[:,:,:na]
            centers=proposal['centers'].to(profile)
            distance=((profile[:,None]-centers[None]).square()*model.action_weight(d)[None,None,None]).sum((-1,-2))/model.T
        gate=torch.softmax(-distance/model.cfg['model']['center_temperature'],-1)[:,cover.get('group',0)]
        parent=output.need_mass[:,proposal['parent']]
        hard_parent=output.need_mass.argmax(1)==proposal['parent']
    else:
        reads=(output.read_mass*output.need_mass[:,None,:,None,None]).sum(2).mean(1)
        parent=reads[:,proposal['domain'],proposal['parent']]
        hard_parent=parent>0
    return (rows*parent*gate).cpu().numpy(),(rows&hard_parent&(gate>=.5)).cpu().numpy()


class RowStream:
    """A continuous permutation stream; every tail is consumed exactly once."""
    def __init__(self,ids,seed,dataset,namespace,batch):
        self.ids=np.asarray(ids,np.int64);self.seed=seed;self.dataset=dataset
        self.namespace=namespace;self.batch=batch;self.cycle=0;self.cursor=0
        self.order=np.empty(0,np.int64)

    def next(self):
        if self.cursor>=len(self.order):
            self.order=rng(self.seed,self.dataset,self.namespace,self.cycle).permutation(self.ids)
            self.cycle+=1;self.cursor=0
        selected=self.order[self.cursor:self.cursor+self.batch]
        self.cursor+=len(selected)
        return selected

    def state(self):
        return copy.deepcopy(self.__dict__)

    def load(self,state):
        self.__dict__.update(copy.deepcopy(state))


@torch.no_grad()
def predict(model,view,device,native=None,data=None,role=None,chunk=4096):
    model.eval();values=[];extra={}
    for begin in range(0,len(view['ids']),chunk):
        pos=np.arange(begin,min(begin+chunk,len(view['ids'])))
        deployed=packet(view,pos,device)
        if native is not None:
            deployed['h0'],deployed['z0']=deployed['h'],deployed['z']
            encoded=native.encode(tensor(data.x[view['ids'][pos]],device))
            deployed['h'],deployed['z']=encoded.h,encoded.logits
        out=model(deployed,tensor(view['domain'][pos],device))
        values.append(out.logits.detach().cpu().numpy())
        for name in ('reference_logits','need_mass','read_mass','null_mass','gain_prediction'):
            value=getattr(out,name,None)
            if value is not None:
                extra.setdefault(name,[]).append(value.detach().cpu().numpy())
    return np.concatenate(values),{name:np.concatenate(parts) for name,parts in extra.items()}


class Trainer:
    def __init__(self,arm,data,views,prefix,seed,cfg,root,device):
        require_implemented(arm)
        self.arm,self.data,self.views,self.prefix=copy.deepcopy(arm),data,views,prefix
        self.seed,self.cfg,self.root,self.device=seed,cfg,Path(root),device
        self.root.mkdir(parents=True,exist_ok=False)
        self.model=make_model(cfg,arm,prefix['source'].teacher_w,prefix['source'].teacher_b,seed).to(device)
        self.full=arm['train_scope']=='FULL_MODEL_JOINT'
        if self.full:
            self.model.current_native=copy.deepcopy(prefix['native']).to(device).requires_grad_(True)
        self.optimizer=make_optimizer(self.model,cfg)
        self.identity=dict(dataset=data.dataset,seed=seed,method=arm['id'],scope=arm['train_scope'],
                           switches=arm['switches'],protocol=cfg['execution']['protocol'])
        self.main=RowStream(data.roles['F'],seed,data.dataset,40,2048)
        self.owners=[RowStream(data.roles['T0'][data.domain[data.roles['T0']]==s],
                              seed,data.dataset,50+s,256) for s in range(3)]
        self.probe_stream=RowStream(np.arange(len(data.probe)),seed,data.dataset,60,256)
        self.state=dict(epoch=0,step=0,shadow_cursor=0,teacher_version=None,selection_history=[],
                        used_s=[],read_s=[],source_binding=[0,0,0],previous_diagnostic=None,cooldown={})
        self.cache=None;self.epochs=[];self.constraints=[];self.diagnostics=[]
        self.transactions=[];self.proposals=[];self.caps=[];self.selected=None;self.final=None
        self.contrast=[]
        self.cost=dict(training_seconds=0.,probe_seconds=0.,projection_seconds=0.,
                       structure_search_seconds=0.,failed_seconds=0.,diagnostic_seconds=0.,
                       committed_update_seconds=0.)
        if not isinstance(self.model,TargetModel):
            view=self.views['F']
            self.model.initialize_partition(packet(view,np.arange(len(view['ids'])),device),
                                            tensor(view['domain'],device),seed)
        write_json(self.root/'EXECUTION_CONFIG.json',dict(config=cfg,arm=arm,identity=self.identity))
        write_json(self.root/'PARAMETER_AND_MAC_LEDGER.json',self.parameter_ledger())

    def parameter_ledger(self):
        trainable=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        foreign=bool(getattr(self.model,'foreign',False))
        source_calls=sum(len(t.leaves) for t in self.model.sources) if foreign else 0
        width=self.cfg['model']['target_match_width'] if isinstance(self.model,TargetModel) else self.cfg['model']['base_head_width']
        H=self.cfg['model']['native_dim'];T=self.cfg['data']['tasks'];R=self.cfg['model']['need_hidden'];P=self.cfg['prefix']['source_feature_dim']
        mac=dict(target_head_linear=T*((H+T)*width+width))
        if not isinstance(self.model,TargetModel):
            mac['need_trunk_linear']=(H+T)*R
            mac['gain_heads_linear']=R*T*max(len(a) for a in self.model.action_maps) if self.model.has_gain else 0
            mac['local_leaf_linear']=max(len(t.leaves) for t in self.model.needs)*R*T
            mac['source_residual_linear']=source_calls*2*P*self.cfg['model']['source_rank']
            mac['foreign_read_linear']=max(len(a) for a in self.model.action_maps)*P*T if foreign else 0
        return dict(trainable_parameters=trainable,model_parameters=sum(p.numel() for p in self.model.parameters()),
                    frozen_native_parameters=sum(p.numel() for p in self.prefix['native'].parameters()),
                    frozen_source_parameters=sum(p.numel() for p in self.prefix['source'].parameters()),
                    inference_source_calls=source_calls,
                    allocated_leaves=self.leaf_counts(),active_leaves=self.leaf_counts(),
                    adapter_linear_MACs_per_request_upper_by_domain=mac,
                    adapter_linear_MACs_total=sum(mac.values()),
                    native_MACs='UNMEASURED',source_pilot_MACs='UNMEASURED')

    def leaf_counts(self):
        if isinstance(self.model,TargetModel):
            return dict(need=[],source=[])
        return dict(need=[len(t.leaves) for t in self.model.needs],
                    source=[len(t.leaves) for t in self.model.sources])

    def deployed(self,role,ids,main_gradient=False):
        view=self.views[role]
        positions=np.searchsorted(view['ids'],ids)
        result=packet(view,positions,self.device)
        if self.full:
            native=self.model.current_native
            native.train(main_gradient)
            result['h0'],result['z0']=result['h'],result['z']
            with torch.set_grad_enabled(main_gradient):
                encoded=native.encode(tensor(self.data.x[ids],self.device))
                result['h'],result['z']=encoded.h,encoded.logits
        return result

    def refresh(self,epoch,freeze_scale=False):
        if isinstance(self.model,TargetModel):
            return
        start=time.monotonic()
        ids=self.data.probe
        y,_=self.data.labels('F','probe',ids,epoch=epoch)
        p=self.deployed('F',ids)
        self.cache=refresh_teacher(self.model,p,tensor(self.data.domain[ids],self.device),
                    tensor(y,self.device),ids,panels=self.data.folds,
                    version=f'{self.arm["id"]}/epoch{epoch}/step{self.state["step"]}',freeze_scale=freeze_scale)
        self.state['teacher_version']=self.cache.version
        table_arrays={}
        table_metadata=[]
        for d,table in enumerate(self.model.tables):
            for key in ('observed_gain','policy_gain','second_moment','ess','unique_count','observed','prior','prior_refreshes'):
                table_arrays[f'd{d}_{key}']=getattr(table,key).detach().cpu().numpy()
            table_metadata.append(dict(target_domain=d,need_leaves=self.model.needs[d].leaves.copy(),
                       actions=[dict(source=s,source_leaf=leaf) for s,leaf in self.model.action_maps[d]],
                       teacher_version=table.version))
        np.savez_compressed(self.root/f'epoch{epoch:02d}_MEASURED_RELATION.npz',**table_arrays)
        write_json(self.root/f'epoch{epoch:02d}_ACTION_MANIFEST.json',table_metadata)
        self.cost['probe_seconds']+=time.monotonic()-start

    def step(self):
        started=time.monotonic()
        self.model.train();epoch=self.state['epoch'];ids=self.main.next()
        y,_=self.data.labels('F','adapt',ids,method=self.arm['id'],epoch=epoch,step=self.state['step'])
        deployed=self.deployed('F',ids,main_gradient=True)
        probe=None
        if self.cache is not None:
            probe=self.cache.subset(self.probe_stream.next())
        owner=None
        if not isinstance(self.model,TargetModel) and epoch>=3:
            owner_ids=np.concatenate([stream.next() for stream in self.owners])
            yy,_=self.data.labels('T0','owner',owner_ids,method=self.arm['id'],epoch=epoch,step=self.state['step'])
            own_view=self.views['T0'];pos=np.searchsorted(own_view['ids'],owner_ids)
            owner=dict(packet=packet(own_view,pos,self.device),domain=tensor(self.data.domain[owner_ids],self.device),
                       y=tensor(yy,self.device),weights=tensor(self.data.owner_weights,self.device))
        losses=training_losses(self.model,deployed,tensor(self.data.domain[ids],self.device),
                               tensor(y,self.device),probe=probe,owner=owner,epoch=epoch,
                               shadow_action=self.state['shadow_cursor'])
        if not all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v)):
            raise FloatingPointError('Nonfinite adaptation loss')
        diagnostic=apply_gradients(losses,self.model,self.optimizer,epoch,self.cfg)
        self.caps.append(dict(step=self.state['step'],epoch=epoch,**diagnostic))
        if self.arm['switches']['CONTRASTIVE'] and epoch>=5:
            self.contrast.append(dict(step=self.state['step'],epoch=epoch,
                    loss=float(losses['contrast'].detach()),
                    **getattr(self.model,'last_contrast_diagnostic',dict(status='NO_CONTRASTIVE_PAIRS'))))
        self.state['step']+=1;self.state['shadow_cursor']+=1
        synchronize(self.device)
        self.cost['committed_update_seconds']+=time.monotonic()-started
        return {k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v)}

    def owner_factory(self):
        view=self.views['T0']
        for begin in range(0,len(view['ids']),4096):
            pos=np.arange(begin,min(begin+4096,len(view['ids'])))
            ids=view['ids'][pos]
            y,_=self.data.labels('T0','constraint',ids,method=self.arm['id'],epoch=self.state['epoch'])
            yield dict(packet=packet(view,pos,self.device),domain=tensor(view['domain'][pos],self.device),
                       y=tensor(y,self.device),weights=tensor(self.data.owner_weights,self.device))

    def foreign_factory(self):
        view=self.views['F']
        for begin in range(0,len(view['ids']),4096):
            pos=np.arange(begin,min(begin+4096,len(view['ids'])))
            yield dict(packet=packet(view,pos,self.device),domain=tensor(view['domain'][pos],self.device))

    def project(self):
        if isinstance(self.model,TargetModel) or self.state['epoch']<3:
            return
        started=time.monotonic()
        records=project_sources(self.model,self.owner_factory,self.foreign_factory,self.cfg)
        if isinstance(records,dict):
            records=records.get('sources',[records])
        for record in records:
            self.constraints.append(dict(epoch=self.state['epoch'],step=self.state['step'],**record))
        self.cost['projection_seconds']+=time.monotonic()-started

    def snapshot(self):
        used,read=self.data.s_state(self.arm['id'])
        self.state.update(used_s=sorted(used),read_s=sorted(read))
        state=copy.deepcopy(self.state)
        state.update(main=self.main.state(),owners=[s.state() for s in self.owners],
                     probe_stream=self.probe_stream.state(),cache=self.cache,
                     cost=copy.deepcopy(self.cost),constraints=copy.deepcopy(self.constraints),
                     diagnostics=copy.deepcopy(self.diagnostics),transactions=copy.deepcopy(self.transactions),
                     proposals=copy.deepcopy(self.proposals),caps=copy.deepcopy(self.caps),contrast=copy.deepcopy(self.contrast))
        return capture(self.model,self.optimizer,self.identity,state)

    def load_snapshot(self,snapshot):
        self.model,self.optimizer,state=restore(snapshot,self.device,lambda m:make_optimizer(m,self.cfg))
        self.main.load(state.pop('main'))
        for stream,value in zip(self.owners,state.pop('owners')):
            stream.load(value)
        self.probe_stream.load(state.pop('probe_stream'));self.cache=state.pop('cache')
        for field in ('cost','constraints','diagnostics','transactions','proposals','caps','contrast'):
            setattr(self,field,state.pop(field))
        self.state=state
        self.data.restore_s(self.arm['id'],state['used_s'],state['read_s'])
        if self.cache is not None:
            self.cache.to(self.device)

    def fork(self,snapshot,variant,proposal):
        branch=copy.copy(self)
        branch.main=copy.deepcopy(self.main);branch.owners=copy.deepcopy(self.owners)
        branch.probe_stream=copy.deepcopy(self.probe_stream)
        branch.load_snapshot(snapshot)
        if variant!='KEEP':
            branch.structure_metadata=branch.model.add_structure(variant,proposal)
            from .permissions import add_new_parameters
            add_new_parameters(branch.model,branch.optimizer,self.cfg)
            from .benefit import align_cache
            branch.cache=align_cache(branch.cache,branch.model)
        else:
            branch.structure_metadata=dict(added_parameters=0,added_macs=0,capacity_match=True)
        return branch

    def structure_window(self,epoch):
        if isinstance(self.model,TargetModel):
            return
        from .conflict import propose
        from .transaction import run_transaction
        proposal=None
        if epoch in self.cfg['split']['diagnostic_epochs']:
            windows=self.cfg['split']['transaction_epochs']
            current_window=windows.index(epoch) if epoch in windows else -1
            excluded={node for node,last in self.state['cooldown'].items() if current_window<=last}
            proposal,diag=propose(self.model,self.cache,self.data.folds,
                                  self.state.get('previous_diagnostic'),self.cfg,epoch,excluded_nodes=excluded)
            self.diagnostics.append({**diag,'epoch':epoch})
            self.state['previous_diagnostic']=diag
            self.proposals.append(dict(epoch=epoch,proposal=proposal_record(proposal),status='PROPOSED' if proposal else 'NO_SPLIT_PROPOSAL'))
        if epoch not in self.cfg['split']['transaction_epochs']:
            return
        dynamic=self.arm['switches']['DYNAMIC_SPLIT']
        if not dynamic or proposal is None:
            for _ in range(8):
                self.step()
            self.project()
            return
        snapshot=self.snapshot();window=self.cfg['split']['transaction_epochs'].index(epoch)
        def train(branch,nsteps):
            for _ in range(nsteps):
                branch.step()
        def evaluate_arms(branches):
            ids=self.data.consume_s(window,self.arm['id'])
            y,m=self.data.labels('S','transaction',ids,window=window,epoch=epoch,method=self.arm['id'])
            position=np.searchsorted(self.views['S']['ids'],ids)
            view={k:np.asarray(v[position]) if k!='ids' else ids for k,v in self.views['S'].items()}
            frozen=copy.deepcopy(snapshot['model']).to(self.device).eval()
            deployed=packet(view,np.arange(len(ids)),self.device)
            if self.full:
                deployed['h0'],deployed['z0']=deployed['h'],deployed['z']
                with torch.no_grad():
                    current=frozen.current_native.encode(tensor(self.data.x[ids],self.device))
                    deployed['h'],deployed['z']=current.h,current.logits
            cover,hard=coverage_weights(frozen,deployed,tensor(view['domain'],self.device),proposal)
            mass=float(cover.sum());ess=mass*mass/max(float(cover@cover),1e-30)
            supported=int(hard.sum())>=self.cfg['split']['min_child_hard_support'] and ess>=self.cfg['split']['min_child_ess']
            del frozen
            result={}
            for key,branch in branches.items():
                z,_=predict(branch.model,view,self.device,
                           native=branch.model.current_native if branch.full else None,data=self.data)
                result[key]=evaluate(y,z,view['domain'],m)
                ll=np.logaddexp(0.,z)-y*z
                result[key]['local_logloss']=float((ll.mean(1)*cover).sum()/mass) if supported and mass>0 else None
                result[key]['local_mass']=float(cover.mean()) if len(cover) else 0.
                result[key]['local_hard_support']=int(hard.sum());result[key]['local_ess']=ess
                result[key].update(branch.structure_metadata)
            return result
        callbacks=dict(fork=self.fork,train=train,project=lambda arm:arm.project(),
                       evaluate=evaluate_arms,capture=lambda arm:arm.snapshot())
        started=time.monotonic()
        result=run_transaction(snapshot,proposal,callbacks,self.cfg)
        self.load_snapshot(result['snapshot'])
        used,read=self.data.s_state(self.arm['id'])
        self.state.update(used_s=sorted(used),read_s=sorted(read))
        if result['winner']!='SPLIT':
            self.state['cooldown'][proposal['node_id']]=window+1
        self.transactions.extend(result['records'])
        self.cost['structure_search_seconds']+=time.monotonic()-started
        append_jsonl(self.root/'TRANSACTION_DECISIONS.jsonl',dict(epoch=epoch,**result['decision']))

    def run(self):
        started=time.monotonic();best=float('inf')
        steps_per_epoch=(len(self.data.roles['F'])+2047)//2048
        if str(self.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(self.device)
        for epoch in range(1,21):
            ep=time.monotonic();self.state['epoch']=epoch;self.model.set_epoch(epoch)
            if epoch>=2 and (epoch>2 or self.cache is None):
                self.refresh(epoch-1)
            sums={}
            for _ in range(steps_per_epoch):
                for k,v in self.step().items():
                    sums[k]=sums.get(k,0.)+v/steps_per_epoch
            if epoch==1:
                self.refresh(epoch,freeze_scale=True)
            if epoch==2 and not isinstance(self.model,TargetModel):
                ids=self.data.probe
                self.model.initialize_partition(self.deployed('F',ids),tensor(self.data.domain[ids],self.device),self.seed)
            self.project()
            if epoch in self.cfg['split']['diagnostic_epochs']:
                self.structure_window(epoch)
            if isinstance(self.model,TargetModel) and epoch in self.cfg['split']['transaction_epochs']:
                for _ in range(8):
                    self.step()
            y,m=self.data.labels('V','select',method=self.arm['id'],epoch=epoch)
            z,_=predict(self.model,self.views['V'],self.device,
                         native=self.model.current_native if self.full else None,data=self.data)
            metric=evaluate(y,z,self.views['V']['domain'],m)
            synchronize(self.device)
            row=dict(epoch=epoch,main_step=self.state['step'],losses=sums,validation=metric,
                     seconds=time.monotonic()-ep,teacher_version=self.state['teacher_version'])
            self.epochs.append(row);append_jsonl(self.root/'EPOCH_METRICS.jsonl',row)
            if epoch>=5:
                self.state['selection_history'].append(dict(epoch=epoch,macro_logloss=metric['macro_logloss']))
                if metric['macro_logloss']<best-1e-12:
                    best=metric['macro_logloss'];self.selected=self.snapshot()
            print(f'ADAPT {self.data.dataset} seed={self.seed} {self.arm["id"]} epoch={epoch}/20 V_LL={metric["macro_logloss"]:.9f}',flush=True)
        self.cost['training_seconds']=time.monotonic()-started
        self.cost['peak_cuda_bytes']=torch.cuda.max_memory_allocated(self.device) if str(self.device).startswith('cuda') else None
        self.cost['time_accounting']='training_seconds is total adaptation wall time; component times overlap this total'
        self.final=self.snapshot()
        expected=20*steps_per_epoch+24
        if self.state['step']!=expected:
            raise AssertionError(f'Matched main clock differs: {self.state["step"]} != {expected}')
        receipt=dict(identity=self.identity,selected_epoch=self.selected['state']['epoch'],final_epoch=20,
                     main_updates=self.state['step'],expected_main_updates=expected,
                     selected_macro_logloss=best,model_storage='RAM_ONLY',cost=self.cost)
        write_json(self.root/'TRAINING_RECEIPT.json',receipt)
        write_csv(self.root/'SOURCE_CONSTRAINTS.csv',self.constraints)
        write_csv(self.root/'GRADIENT_SKETCH_DIAGNOSTICS.csv',self.diagnostics)
        write_csv(self.root/'TRANSACTION_RESULTS.csv',self.transactions)
        write_csv(self.root/'GRADIENT_PERMISSIONS.csv',self.caps)
        write_csv(self.root/'CONTRASTIVE_DIAGNOSTICS.csv',self.contrast)
        write_json(self.root/'SPLIT_PROPOSALS.json',self.proposals)
        write_json(self.root/'COST_LEDGER.json',self.cost)
        return receipt

    def export_state(self,role,which):
        retained_cost=copy.deepcopy(self.cost)
        snap=getattr(self,which);self.load_snapshot(snap)
        self.cost=retained_cost
        view=self.views[role]
        if role=='Q' and not isinstance(self.model,TargetModel):
            self.register_F_groups(which)
        purpose={'Q':'development','V':'select','E':'final'}[role]
        y,m=self.data.labels(role,purpose,method=self.arm['id'],state=which)
        z,extras=predict(self.model,view,self.device,
                         native=self.model.current_native if self.full else None,data=self.data)
        identity=state_manifest(self.identity,self.state['epoch'],self.state['step'],which)
        result=export_prediction(self.root,role,which,view,y,m,z,extras,identity)
        if not isinstance(self.model,TargetModel):
            self.export_diagnostic(which,role,view,y)
        return result

    @torch.no_grad()
    def latency(self):
        started=time.monotonic();self.model.eval()
        records=[];foreign=bool(getattr(self.model,'foreign',False))
        for batch in (1,256):
            ids=self.views['Q']['ids'][:batch]
            deployed=self.deployed('Q',ids)
            domain=tensor(self.data.domain[ids],self.device)
            raw=tensor(self.data.x[ids],self.device)
            def cached():
                return self.model(deployed,domain)
            def raw_forward():
                if self.full and isinstance(self.model,TargetModel):
                    current=self.model.current_native.encode(raw)
                    p=dict(h=current.h,z=current.logits,phi=current.h.new_zeros(len(raw),3,64))
                else:
                    reference=self.prefix['native'].encode(raw)
                    phi=self.prefix['source'](reference.h) if foreign else reference.h.new_zeros(len(raw),3,64)
                    p=dict(h=reference.h,z=reference.logits,phi=phi)
                    if self.full:
                        current=self.model.current_native.encode(raw)
                        p.update(h0=reference.h,z0=reference.logits,h=current.h,z=current.logits)
                return self.model(p,domain)
            for kind,call in (('cached_adapter',cached),('raw_X_end_to_end',raw_forward)):
                for _ in range(5):
                    call()
                times=[]
                for _ in range(20):
                    synchronize(self.device);begin=time.perf_counter();call();synchronize(self.device)
                    times.append((time.perf_counter()-begin)*1000)
                records.append(dict(kind=kind,batch=len(ids),p50_ms=float(np.percentile(times,50)),
                                    p95_ms=float(np.percentile(times,95)),repetitions=len(times),
                                    shared_hardware=True))
        self.cost['latency_measurement_seconds']=time.monotonic()-started
        write_csv(self.root/'INFERENCE_LATENCY.csv',records)
        write_json(self.root/'COST_LEDGER.json',self.cost)
        return records

    @torch.no_grad()
    def register_F_groups(self,which):
        ids=self.data.probe
        y,_=self.data.labels('F','diagnostic',ids,method=self.arm['id'],state=which,purpose_detail='PRE_Q_GROUP_REGISTRATION')
        deployed=self.deployed('F',ids)
        domain=tensor(self.data.domain[ids],self.device)
        measured=measure_benefits(self.model,deployed,domain,tensor(y,self.device))
        benefits=measured['benefits'].cpu().numpy();q=measured['q'].cpu().numpy()
        arrays=dict(row_id=ids,domain=self.data.domain[ids],benefits=benefits,need_mass=q,
                    folds=self.data.folds,labels=y)
        current_prediction=measured.get('prediction')
        if current_prediction is not None:
            arrays['prediction']=current_prediction.cpu().numpy()
        actions=[[dict(source=int(s),source_leaf=leaf) for s,leaf in row] for row in self.model.action_maps]
        features=torch.cat([deployed['h'],deployed['z']],-1).cpu().numpy()
        folded,fold_prediction=fold_ridge_skill(features,benefits,self.data.folds)
        if fold_prediction is not None:
            arrays['fold_prediction']=fold_prediction
        np.savez_compressed(self.root/f'F_{which}_GAIN_TABLES.npz',**arrays)
        groups=[]
        for d in range(self.model.D):
            chosen=self.data.domain[ids]==d;na=len(actions[d])
            for k in range(len(self.model.needs[d].leaves)):
                weights=q[:,k]*chosen;mass=float(weights.sum());square=float(weights@weights)
                hard=chosen&(q.argmax(1)==k)
                profile=np.einsum('n,nta->ta',weights,benefits[:,:,:na])/max(mass,1e-30)
                panels=[]
                for fold in (0,1):
                    w=weights*(self.data.folds==fold);m=float(w.sum())
                    p=np.einsum('n,nta->ta',w,benefits[:,:,:na])/max(m,1e-30)
                    best=np.concatenate([np.zeros((self.model.T,1)),p],-1).argmax(-1)-1
                    panels.append(dict(fold=fold,hard_support=int((hard&(self.data.folds==fold)).sum()),
                                       soft_mass=m,ess=m*m/max(float(w@w),1e-30),
                                       gain_profile=p.tolist(),best_action=best.tolist()))
                best=np.concatenate([np.zeros((self.model.T,1)),profile],-1).argmax(-1)-1
                groups.append(dict(domain=d,need_leaf=self.model.needs[d].leaves[k],route_index=k,
                    hard_support=int(hard.sum()),soft_mass=mass,ess=mass*mass/max(square,1e-30),
                    mean_gain_profile=profile.tolist(),best_action=best.tolist(),panels=panels,
                    supported=int(hard.sum())>=100 and mass*mass/max(square,1e-30)>=64))
        identity=state_manifest(self.identity,self.state['epoch'],self.state['step'],which)
        write_json(self.root/f'F_{which}_GROUP_REGISTRATION.json',dict(identity=identity,
            status='F_ONLY_DEFINITION_BEFORE_CORRESPONDING_Q_STATE_EXPORT',definition='deployed_need_route_with_F_action_preferences',
            actions=actions,groups=groups,predictor_fold=folded,
            cross_method_comparison='join candidate Q need_mass by row_id onto every comparator prediction',
            cross_seed_alignment='domain, task and F preferred source action; never raw cluster index'))
        student=calibration(benefits,current_prediction.cpu().numpy()*float(self.model.benefit_scale)) if current_prediction is not None else dict(status='NOT_ACTIVE_NO_GAIN_HEAD')
        write_json(self.root/f'F_{which}_GAIN_CALIBRATION.json',dict(current_student=student,current_fold_predictor=folded,
                   state_identity=identity))

    @torch.no_grad()
    def export_diagnostic(self,which,role,view,y):
        started=time.monotonic();observations=[];predictions=[];responsibilities=[]
        for begin in range(0,len(y),4096):
            pos=np.arange(begin,min(begin+4096,len(y)))
            deployed=self.deployed(role,view['ids'][pos])
            measured=measure_benefits(self.model,deployed,tensor(view['domain'][pos],self.device),tensor(y[pos],self.device))
            observations.append(measured['benefits'].cpu().numpy())
            if measured.get('prediction') is not None:
                predictions.append(measured['prediction'].cpu().numpy())
            responsibilities.append(measured['q'].cpu().numpy())
        benefits=np.concatenate(observations);mass=np.concatenate(responsibilities)
        arrays=dict(row_id=view['ids'],domain=view['domain'],benefits=benefits,need_mass=mass)
        status=dict(student=dict(status='NOT_ACTIVE_NO_GAIN_HEAD'))
        if predictions:
            prediction=np.concatenate(predictions);arrays['prediction']=prediction
            status['student']=calibration(benefits,prediction*float(self.model.benefit_scale))
        if self.cache is not None:
            target=self.cache.benefits.cpu().numpy()
            features=torch.cat([self.cache.packet['h'],self.cache.packet['z']],-1).cpu().numpy()
            folded,pred=fold_ridge_skill(features,target,self.data.folds)
            status['previous_teacher_F_predictor_fold_diagnostic']=dict(**folded,teacher_version=self.cache.version)
            if pred is not None:
                np.savez_compressed(self.root/f'{which}_F_FOLD_GAIN_PREDICTIONS.npz',
                                    row_id=self.data.probe,target=target,prediction=pred,folds=self.data.folds)
        np.savez_compressed(self.root/f'{role}_{which}_GAIN_TABLES.npz',**arrays)
        write_json(self.root/f'{role}_{which}_GAIN_CALIBRATION.json',status)
        actions=[dict(target=d,source=s,source_leaf=leaf,task=t)
                 for d,row in enumerate(self.model.action_maps) for s,leaf in row for t in range(self.model.T)]
        write_json(self.root/f'{role}_{which}_ACTION_MANIFEST.json',dict(actions=actions,
                   kind='single_foreign_leaf_relative_to_same_state_OFF',teacher_version=self.state['teacher_version']))
        self.cost['diagnostic_seconds']+=time.monotonic()-started
