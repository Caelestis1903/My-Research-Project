"""Observed frozen raw-X delivery, fair algebraic fusion and cost accounting.

No fit, selection, checkpoint transport, or mutation of resident predictors occurs.
"""
from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.special import expit
import torch
from torch.nn import functional as F
from .artifacts import _json,_sha,_plain
from .state import array_hash,digest,state_hash,capture_rng
from .protocol import RoleView
from .fixture_export import live_model_hashes,export_reader_fixture

REQUIRED=('NATIVE','B_REFIT','H_DIRECT','H_DIRECT_EFFORT','GROUP_CAL',
          'FULL_SOURCE_ALL','RCF','POOLED_TARGET','LOCAL_ACTIVE')


def _module(value):
    return value[0] if isinstance(value,tuple) else value


def _count(module):
    return sum(p.numel() for p in module.parameters())


def _model_groups(bundle,predictions,formations):
    groups={'native':bundle.native,'source':bundle.pilots.source_bank,
            'initial':bundle.pilots.initial_bank}
    groups.update({f'local_{i}':m for i,m in enumerate(bundle.pilots.local_banks)})
    groups.update({f'base_{k}':_module(m) for k,m in bundle.base_models.items()})
    for method,pred in predictions.items():
        models=pred.diagnostics.get('models',pred.diagnostics.get('_models',{}))
        if isinstance(models,dict):groups.update({f'{method}_{k}':_module(m) for k,m in models.items()})
    for method,formed in formations.items():
        groups.update({f'{method}_{fid}':f.module for fid,f in formed.functions.items()})
    return groups


def _observation_fingerprint(bundle,predictions,formations):
    groups=_model_groups(bundle,predictions,formations)
    return dict(models={key:state_hash(m.state_dict()) for key,m in groups.items()},
        modes={key:{n:bool(a.training) for n,a in m.named_modules()} for key,m in groups.items()},
        requires_grad={key:{n:bool(p.requires_grad) for n,p in m.named_parameters()} for key,m in groups.items()},
        gradients={key:state_hash({n:p.grad for n,p in m.named_parameters()}) for key,m in groups.items()},
        readers={method:digest(pred.diagnostics.get('selected_coefficients',{})) for method,pred in predictions.items()},
        normalizers=digest(bundle.normalizers),rng=state_hash(capture_rng()))


def _policy(pred,target,task,group):
    values=pred.diagnostics.get('selected_coefficients')
    if values is None:
        values=getattr(pred.diagnostics.get('predictor'),'coefficients',None)
    if values is None:raise RuntimeError('Selected frozen reader coefficients unavailable')
    result=values.get((target,task,group),values.get(f'{target}_{task}_{group}'))
    if result is None:raise RuntimeError('Selected frozen reader group unavailable')
    return result


class FrozenDeployment:
    """Raw-X compiled predictor with nonlinear boundaries preserved.

    Native/hidden/direct computations remain FP32. Linear feature readouts and
    scalar policy algebra are FP64. All downstream tensors are independent frozen
    copies placed on the native device outside the timed section.
    """
    def __init__(self,bundle,pred,method,*,device=None):
        self.bundle=bundle;self.pred=pred;self.method=method
        self.device=torch.device(device or next(bundle.native.parameters()).device)
        if self.device!=next(bundle.native.parameters()).device:
            raise ValueError('Deployment native device must equal resident native device')
        self.native=bundle.native;self.models={};self.hidden={};self.policies={};self.dependencies={}
        self.direct_models=pred.diagnostics.get('models',pred.diagnostics.get('_models'))
        self.bank=getattr(pred.diagnostics.get('predictor'),'bank',None)
        self.off=bool(getattr(pred.diagnostics.get('predictor'),'off',False) or pred.selection.get('selected')=='OFF')
        self.cut=torch.as_tensor(np.asarray(bundle.normalizers['group_cut']),dtype=torch.float64,device=self.device)
        self.kind='linear' if pred.diagnostics.get('selected_coefficients') is not None else method
        self.eta=bool(pred.diagnostics.get('eta',False))
        if self.kind=='linear' or method in ('B_REFIT','PILOT_PROB'):
            for (t,q),model in bundle.base_models.items():self._model(('base',t,q),_module(model))
        if method in ('H_DIRECT','H_DIRECT_EFFORT','ETA_DIRECT_MATCHED'):
            if not self.direct_models:raise RuntimeError('Selected direct fitted models missing')
            for (t,q),model in self.direct_models.items():self._model(('direct',t,q),_module(model))
            if self.eta:self._source_full()
        elif method=='PILOT_PROB':
            if not self.direct_models:raise RuntimeError('Selected probability gate models missing')
            for (t,q),model in self.direct_models.items():self._model(('gate',t,q),_module(model))
            self._source_full()
        elif method=='LOCAL':self._source_full()
        elif self.kind=='linear' and not self.off:
            if self.bank is None:raise RuntimeError('Selected fitted feature bank missing')
            self._compile_linear()
        elif method not in ('NATIVE','B_REFIT') and self.kind!='linear':
            raise NotImplementedError('No frozen deployment compiler for '+method)
        self.reader_analytical_size=sum(2+len(np.asarray(_policy(pred,t,q,g)['weights'])) for t in range(3) for q in range(2) for g in (0,1)) if self.kind=='linear' and not self.off else 0
        if self.kind!='linear' or self.off:self.cut=None
        # Compiled deployment retains only its needed tensors and native model,
        # not a reference to the research bundle's unused source/bank models.
        self.bundle=None;self.pred=None;self.bank=None;self.direct_models=None

    def _tensor(self,value,dtype=torch.float32):
        return torch.as_tensor(value,dtype=dtype,device=self.device).detach().clone()

    def _model(self,key,model):
        if key in self.models:return
        if not hasattr(model,'hidden') or not hasattr(model,'output'):
            raise NotImplementedError('Frozen head has no supported hidden/output boundary')
        values=dict(w1=self._tensor(model.hidden.weight),b1=self._tensor(model.hidden.bias),
                    w2=self._tensor(model.output.weight),b2=self._tensor(model.output.bias),parameters=_count(model))
        if hasattr(model,'mean'):
            values.update(mean=self._tensor(model.mean),std=self._tensor(model.std))
        self.models[key]=values

    def _source_full(self):
        for s in range(3):
            for q in range(2):self._model(('pilot',s,q),self.bundle.pilots.source_bank.head(s,q))

    def _hidden(self,key,layer):
        if key not in self.hidden:
            self.hidden[key]=dict(weight=self._tensor(layer.weight),bias=self._tensor(layer.bias),
                                  parameters=layer.weight.numel()+layer.bias.numel())
        return key

    def _source_hidden(self,source,task,*,initial=False,local=None,target=None):
        if local is not None:
            bank=self.bundle.pilots.local_banks[local];owner=target
            key=('local',local,owner,task);norm=self.bundle.pilots.local_normalizers[owner,task,local]
        else:
            bank=self.bundle.pilots.initial_bank if initial else self.bundle.pilots.source_bank;owner=source
            key=('initial' if initial else 'source',source,task)
            if initial:
                fit=self.bundle.pilots.hidden('fit',source,task,initial=True,standardized=False)
                fit=fit[self.bundle.views['fit'].domain==source]
                norm=dict(mean=fit.mean(0,dtype=np.float64),std=np.maximum(fit.std(0,dtype=np.float64),1e-3))
            else:norm=self.bundle.pilots.phi_normalizers[source,task]
        return self._hidden(key,bank.head(owner,task).hidden),norm

    def _compile_linear(self):
        fixed=getattr(self.bank,'fixed_method',None)
        for target in range(3):
            for task in range(2):
                for group in (0,1):
                    policy=_policy(self.pred,target,task,group)
                    weights=np.asarray(policy.get('weights',policy.get('w')),np.float64)
                    terms=[];bias=float(policy['beta']);sources=[s for s in range(3) if s!=target]
                    if fixed in ('FULL_SOURCE','FULL_SOURCE_ALL','FULL_TARGET','RANDOM_FEATURE'):
                        for index,source in enumerate(range(2) if fixed=='FULL_TARGET' else sources):
                            w=weights[index*32:(index+1)*32]
                            if not np.any(w):continue
                            key,norm=self._source_hidden(source,task,initial=fixed=='RANDOM_FEATURE',
                                local=source if fixed=='FULL_TARGET' else None,target=target)
                            effective=w/np.asarray(norm['std'],np.float64)
                            bias-=float(np.dot(effective,np.asarray(norm['mean'],np.float64)))
                            terms.append((key,self._tensor(effective,torch.float64)))
                    elif fixed=='PILOT_LOGIT':
                        multipliers={s:float(w) for s,w in zip(sources,weights)}
                        multipliers[target]=-float(weights.sum())
                        for source,multiplier in multipliers.items():
                            if multiplier==0:continue
                            head=self.bundle.pilots.source_bank.head(source,task)
                            key=self._hidden(('source',source,task),head.hidden)
                            effective=multiplier*head.output.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
                            bias+=multiplier*float(head.output.bias.detach().cpu())
                            terms.append((key,self._tensor(effective,torch.float64)))
                    elif hasattr(self.bank,'functions'):
                        for column,w in zip(self.bank.columns[target,task],weights):
                            fn=self.bank.functions[column['function_id']]
                            multiplier=float(w)*float(fn.scale)
                            if multiplier==0:continue
                            key=self._hidden(('delta',fn.function_id),fn.module.hidden)
                            effective=multiplier*fn.module.output.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
                            bias+=multiplier*float(fn.module.output.bias.detach().cpu())
                            terms.append((key,self._tensor(effective,torch.float64)))
                    elif hasattr(self.bank,'projections'):
                        offset=0
                        for source in sources:
                            matrix=self.bank.projections.get((source,task))
                            if matrix is None:continue
                            k=matrix.shape[1];w=weights[offset:offset+k];offset+=k
                            if not np.any(w):continue
                            key,norm=self._source_hidden(source,task)
                            effective=np.asarray(matrix,np.float64)@w/np.asarray(norm['std'],np.float64)
                            bias-=float(np.dot(effective,np.asarray(norm['mean'],np.float64)))
                            terms.append((key,self._tensor(effective,torch.float64)))
                    elif fixed!='GROUP_CAL':
                        raise NotImplementedError('Frozen feature compiler metadata unavailable')
                    self.policies[target,task,group]=dict(alpha=float(policy['alpha']),bias=bias,terms=terms)

    def _head(self,key,x):
        m=self.models[key]
        if 'mean' in m:x=((x-m['mean'])/m['std']).clamp(-10,10)
        out=F.linear(F.relu(F.linear(x,m['w1'],m['b1'])),m['w2'],m['b2'])
        return out.squeeze(-1) if out.shape[-1]==1 else out

    def _direct(self,h,z,domain,*,family='base',eta=False):
        output=torch.empty((len(h),2),device=self.device,dtype=torch.float32)
        for target in range(3):
            pos=domain==target
            if not bool(pos.any()):continue
            for task in range(2):
                parts=[h[pos],z[pos]]
                if eta:parts.extend(self._head(('pilot',s,task),h[pos])[:,None] for s in range(3))
                output[pos,task]=self._head((family,target,task),torch.cat(parts,dim=1))
        return output

    def __call__(self,raw_cpu):
        with torch.inference_mode():
            x=raw_cpu.to(device=self.device,dtype=torch.float32)
            packet=self.native.encode(x);h,z=packet.h,packet.logits;domain=x[:,-1].long()
            if self.method=='NATIVE':probability=torch.sigmoid(z)
            elif self.method=='LOCAL':
                a=z.clone()
                for target in range(3):
                    own=domain==target
                    if bool(own.any()):
                        for q in range(2):a[own,q]+=self._head(('pilot',target,q),h[own])
                probability=torch.sigmoid(a)
            elif self.method in ('H_DIRECT','H_DIRECT_EFFORT','ETA_DIRECT_MATCHED'):
                probability=torch.sigmoid(self._direct(h,z,domain,family='direct',eta=self.eta))
            else:
                base=self._direct(h,z,domain)
                if self.method=='B_REFIT' or (self.kind=='linear' and self.off):probability=torch.sigmoid(base)
                elif self.method=='PILOT_PROB':
                    probability=torch.empty((len(h),2),device=self.device,dtype=torch.float64)
                    for target in range(3):
                        own=domain==target
                        if not bool(own.any()):continue
                        for q in range(2):
                            candidates=torch.stack([base[own,q]]+[z[own,q]+self._head(('pilot',s,q),h[own]) for s in range(3) if s!=target],dim=1)
                            features=torch.cat((h[own],z[own],torch.sigmoid(candidates)),dim=1)
                            gate=self._head(('gate',target,q),features)
                            probability[own,q]=(gate.double().softmax(-1)*candidates.double().sigmoid()).sum(-1)
                else:
                    logits=torch.empty((len(h),2),device=self.device,dtype=torch.float64)
                    for target in range(3):
                        own=domain==target
                        if not bool(own.any()):continue
                        for task in range(2):
                            group=(z[:,task].double()>self.cut[target,task]).long()
                            hidden_cache={}
                            for g in (0,1):
                                pos=own&(group==g)
                                if not bool(pos.any()):continue
                                policy=self.policies[target,task,g]
                                a=policy['alpha']*base[pos,task].double()+policy['bias']
                                for key,weight in policy['terms']:
                                    component=self.hidden[key]
                                    if key not in hidden_cache:
                                        hidden_cache[key]=F.relu(F.linear(h[own],component['weight'],component['bias']))
                                    activation=hidden_cache[key][group[own]==g]
                                    a=a+activation.double()@weight
                                logits[pos,task]=a
                    probability=logits.sigmoid()
            return probability.detach().cpu().numpy()

    def parameter_counts(self):
        native=_count(self.native)
        models=sum(v['parameters'] for v in self.models.values())
        hidden=sum(v['parameters'] for v in self.hidden.values())
        analytical=self.reader_analytical_size
        fused_coefficients=sum(2+sum(w.numel() for _,w in p['terms']) for p in self.policies.values())
        active=[]
        for t in range(3):
            for q in range(2):
                if self.kind=='linear':
                    model_add=self.models['base',t,q]['parameters'];keys=set()
                    maximum=0
                    for g in (0,1):
                        policy=self.policies.get((t,q,g),dict(terms=[]))
                        gkeys={key for key,w in policy['terms']};keys|=gkeys
                        maximum=max(maximum,sum(self.hidden[k]['parameters'] for k in gkeys)+
                            (2+sum(w.numel() for _,w in policy['terms']) if not self.off else 0))
                    extra=model_add+maximum
                elif self.method in ('H_DIRECT','H_DIRECT_EFFORT','ETA_DIRECT_MATCHED'):
                    extra=self.models['direct',t,q]['parameters']+sum(v['parameters'] for k,v in self.models.items() if self.eta and k[0]=='pilot' and k[2]==q)
                elif self.method=='B_REFIT':extra=self.models['base',t,q]['parameters']
                elif self.method=='LOCAL':extra=self.models['pilot',t,q]['parameters']
                elif self.method=='PILOT_PROB':extra=self.models['base',t,q]['parameters']+self.models['gate',t,q]['parameters']+sum(self.models['pilot',s,q]['parameters'] for s in range(3) if s!=t)
                else:extra=0
                active.append(dict(target=t,task=q,additional_active_parameters_max_over_groups=int(extra),shared_native_parameters=native))
        return dict(standalone_deployed_parameters=native+models+hidden+fused_coefficients,
            shared_native_parameters=native,deployed_full_head_parameters=models,
            deployed_hidden_parameters=hidden,reader_analytical_parameters=analytical,
            deployed_fused_linear_scalars=fused_coefficients,off_policy=self.off,
            unused_source_branch_removed=self.off,per_target_task_active=active,
            normalization_buffers=sum(m['mean'].numel()+m['std'].numel() for m in self.models.values() if 'mean' in m),
            group_threshold_buffers=6 if self.kind=='linear' and not self.off else 0,
            native_nonparameter_buffers=sum(v.numel() for v in self.native.buffers()),
            definition='Stored trained/derived scalar weights; buffers separate; native counted once. Per-target/task counts are active additions plus separately listed shared native.')


def _reference_raw(bundle,pred,method,raw,rows):
    """Unfused retained formulas on the same independently encoded raw batch."""
    from .baselines import predict_direct_models,_raw_source,raw_features,predict_mixture_models
    from .readers import LinearPredictor
    role='_delivery_observation';device=next(bundle.native.parameters()).device
    with torch.inference_mode():packet=bundle.native.encode(raw.to(device))
    h,z=packet.h.cpu().numpy(),packet.logits.cpu().numpy();domain=raw[:,-1].numpy().astype(np.int64)
    groups=np.zeros((len(rows),2),np.int64)
    for t in range(3):groups[domain==t]=(z[domain==t]>bundle.normalizers['group_cut'][t]).astype(np.int64)
    context=copy.copy(bundle);context.views=dict(bundle.views)
    context.views[role]=RoleView(np.asarray(rows),domain,None,h,z,groups)
    context.base_logits=dict(bundle.base_logits)
    if method=='NATIVE':return expit(z)
    if method=='LOCAL':
        source=_raw_source(bundle,h);a=z.copy()
        for t in range(3):a[domain==t]+=source[domain==t,t]
        return expit(a)
    models=pred.diagnostics.get('models',pred.diagnostics.get('_models'))
    if method in ('H_DIRECT','H_DIRECT_EFFORT','ETA_DIRECT_MATCHED'):
        eta=pred.diagnostics.get('eta',False);source=_raw_source(bundle,h) if eta else None
        return expit(predict_direct_models(context,models,role,eta=eta,raw_h=h,raw_z=z,raw_source=source))
    base=predict_direct_models(context,bundle.base_models,role,raw_h=h,raw_z=z)
    context.base_logits[role]=base
    if method=='B_REFIT':return expit(base)
    if method=='PILOT_PROB':return predict_mixture_models(context,models,role,raw_h=h,raw_z=z,raw_source=_raw_source(bundle,h))
    predictor=pred.diagnostics.get('predictor');bank=getattr(predictor,'bank',None)
    if bank is None:raise RuntimeError('Unfused reference feature bank missing')
    if getattr(predictor,'off',False):return expit(base)
    content=raw_features(context,bank,role,h,z,domain)
    selected=pred.diagnostics.get('selected_coefficients',predictor.coefficients)
    reference=LinearPredictor(context,bank,selected)
    return reference.predict(role,content,base=base,groups=groups)


def _native_eval_ready(bundle):
    # Native encode invokes the retained module graph. Compiled downstream heads
    # use copied F.linear/ReLU tensors, so unrelated original head modes are not
    # execution guards (the real initial random bank is intentionally train()).
    return not any(module.training for module in bundle.native.modules())


def benchmark_predictor(bundle,pred,method,*,batch_rows=4096):
    """Approved fixed3+10 raw-X benchmark; no labels or predictor fitting."""
    rows=np.asarray(bundle.views['Q_explore'].row_id[:batch_rows],np.int64)
    if len(rows)!=batch_rows or len(np.unique(rows))!=batch_rows:
        return dict(status='NOT_MEASURED',reason='Fewer than4096 distinct natural Q_explore rows')
    raw=torch.from_numpy(np.array(bundle.data.x[rows],dtype=np.float32))
    if not _native_eval_ready(bundle):
        return dict(status='NOT_MEASURED',reason='Retained native graph is in training mode; observational code will not change it')
    native_device=next(bundle.native.parameters()).device
    resident_allocation=int(torch.cuda.memory_allocated(native_device)) if native_device.type=='cuda' else None
    started=time.monotonic();deployment=FrozenDeployment(bundle,pred,method)
    compile_seconds=time.monotonic()-started
    compiled_allocation=int(torch.cuda.memory_allocated(native_device)) if native_device.type=='cuda' else None
    reference=_reference_raw(bundle,pred,method,raw,rows);actual=deployment(raw)
    error=float(np.max(np.abs(np.asarray(actual,np.float64)-np.asarray(reference,np.float64))))
    retained=pred.probabilities.get('Q_explore')
    cached_error=float(np.max(np.abs(np.asarray(actual,np.float64)-np.asarray(retained[:batch_rows],np.float64)))) if retained is not None else None
    if not np.isfinite(actual).all() or error>1e-6 or (cached_error is not None and cached_error>1e-6):
        return dict(status='NOT_MEASURED',reason='Fused/raw reference parity exceeds unchanged1e-6 acceptance',
                    max_abs_probability_error=error,retained_prediction_max_abs_error=cached_error,rows_hash=array_hash(rows),compile_seconds=compile_seconds)
    device=deployment.device
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    for _ in range(3):sync();deployment(raw);sync()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    seconds=[]
    for _ in range(10):
        sync();start=time.perf_counter();deployment(raw);sync();seconds.append(time.perf_counter()-start)
    peak=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None
    return dict(status='MEASURED',batch_size=batch_rows,row_rule='first4096 original-order natural Q_explore row_ids',
        row_ids=rows.tolist(),rows_hash=array_hash(rows),schema_hash=digest(bundle.data.schema),
        repetitions=seconds,warmups=3,measured_repetitions=10,mean_seconds=float(np.mean(seconds)),
        median_seconds=float(np.median(seconds)),std_seconds=float(np.std(seconds,ddof=1)),
        min_seconds=float(min(seconds)),max_seconds=float(max(seconds)),rows_per_second=batch_rows/float(np.mean(seconds)),
        max_abs_probability_error=error,retained_prediction_max_abs_error=cached_error,parity_tolerance=1e-6,compile_seconds=compile_seconds,
        peak_cuda_allocation_bytes=peak,resident_cuda_allocation_before_compile=resident_allocation,
        cuda_allocation_after_compile=compiled_allocation,
        peak_cuda_increment_over_resident=None if peak is None else peak-resident_allocation,device=str(device),native_dtype='float32',linear_readout_dtype='float64',
        threads=torch.get_num_threads(),torch_version=torch.__version__,python_version=platform.python_version(),
        cuda_runtime=torch.version.cuda,transfer_policy='preloaded CPU FP32 raw X to native device and final p back to CPU, both inside timing',
        scope='raw_X_to_native_H_Z_to_required_frozen_components_to_probability',labels_used=False,
        fusion='FP64 source hidden standardization folded into linear readout; formed output scale/reader coefficients folded after ReLU',
        parameters=deployment.parameter_counts())


def _training_costs(bundle,predictions,formations):
    rows=[]
    def add(component,seconds,scope,**extra):
        rows.append(dict(dataset=bundle.dataset,seed=bundle.seed,component=component,
            seconds=None if seconds is None else float(seconds),scope=scope,
            status='MEASURED' if seconds is not None else 'NOT_MEASURED',**extra))
    history=bundle.diagnostics.get('native_history',[])
    add('NATIVE_EPOCHS',sum(x['seconds'] for x in history) if history else None,
        '24 native training+validation epochs; source/pilot/cache excluded',epochs=len(history),shared_prefix=True)
    pilot_records=[];path=Path(bundle.output_root)/'training.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:pilot_records.append(json.loads(line))
            except json.JSONDecodeError:continue
    for component in ('PILOT','LOCAL_PILOT_0','LOCAL_PILOT_1','SOURCE_LABEL_SHUFFLE_PILOT'):
        selected=[r for r in pilot_records if r.get('component')==component]
        # All actual observations are retained, including replayed visits, never
        # mislabeled as one intended trajectory if the log contains duplicates.
        if selected:add(component,sum(r['seconds'] for r in selected),
            'observed pilot-pass log entries; includes repeats if present',passes_observed=len(selected),
            pass_ids=[r['pass_id'] for r in selected],shared_prefix=component!='SOURCE_LABEL_SHUFFLE_PILOT')
    add('PREFIX_WALL',bundle.diagnostics.get('elapsed_prefix_seconds'),
        'inclusive native/cache/source+local pilots; report separately, do not add epoch/pilot subcomponents again',shared_prefix=True,inclusive=True)
    add('B_OOF_AND_REFIT',bundle.diagnostics.get('base_seconds'),
        'all two-fold B and B_refit head fitting+prediction; shared across methods',shared_prefix=True)
    add('DIRECT_FAMILY_COMBINED',bundle.diagnostics.get('direct_seconds'),
        'H_DIRECT+ETA_DIRECT_MATCHED+H_DIRECT_EFFORT combined; per-arm seconds were not recorded',
        methods=['H_DIRECT','ETA_DIRECT_MATCHED','H_DIRECT_EFFORT'],per_arm_allocation='UNAVAILABLE_DO_NOT_SPLIT')
    for method,pred in predictions.items():
        elapsed=pred.diagnostics.get('elapsed_seconds')
        if elapsed is not None:add(method+'_READER_OR_GATE',elapsed,
            'actual method reader/gate fitting+selection+prediction wall time; excludes shared prefix/base')
        candidates=pred.diagnostics.get('candidates',[])
        starts={}
        for candidate in candidates:
            bank=candidate.get('bank') if isinstance(candidate,dict) else None
            ident=getattr(bank,'identity',{})
            if ident.get('method') in ('RRR_K2','UNGROUPED_RRR'):
                starts[ident.get('start',candidate.get('start_id'))]=ident.get('elapsed_seconds')
        for start,seconds in starts.items():add(method+'_FORMATION_'+str(start),seconds,
            'one actual RRR formation start; separate from reader; both starts counted once',start=str(start))
    for method,formed in formations.items():
        fresh=[f for f in formed.functions.values() if not f.metadata.get('reused_from')]
        add(method+'_FORMATION_TERMINAL',sum(f.metadata['elapsed_seconds'] for f in fresh) if fresh else 0.,
            'sum of unique completed slot terminal elapsed times; cumulative per-pass clocks are NOT summed',
            slots=len(fresh),reused_slots=len(formed.functions)-len(fresh),
            effective_adam_steps=sum(f.metadata.get('effective_steps',0) for f in fresh),
            trained_parameter_instances=sum(f.metadata.get('parameter_count',_count(f.module)) for f in fresh))
    return rows


def _training_parameters(bundle,pred,method,formations):
    base=sum(_count(_module(m)) for m in bundle.base_models.values())
    source=_count(bundle.pilots.source_bank)
    local=sum(_count(m) for m in bundle.pilots.local_banks)
    reader=0
    off=bool(getattr(pred.diagnostics.get('predictor'),'off',False) or pred.selection.get('selected')=='OFF')
    if pred.diagnostics.get('selected_coefficients') is not None:
        reader=sum(2+len(np.asarray(_policy(pred,t,q,g)['weights'])) for t in range(3) for q in range(2) for g in (0,1))
    models=pred.diagnostics.get('models',pred.diagnostics.get('_models',{}))
    direct=sum(_count(_module(m)) for m in models.values()) if isinstance(models,dict) else 0
    formed=formations.get(method);new=sum(_count(f.module) for f in formed.functions.values() if not f.metadata.get('reused_from')) if formed else 0
    source_needed=method in ('LOCAL','ETA_DIRECT_MATCHED','PILOT_PROB','PILOT_LOGIT','FULL_SOURCE','FULL_SOURCE_ALL','SPECTRAL_K2','RRR_K2','UNGROUPED_RRR') or bool(formed and method!='LOCAL_ACTIVE')
    local_needed=method=='FULL_TARGET' or method=='LOCAL_ACTIVE'
    base_needed=method not in ('NATIVE','LOCAL','H_DIRECT','H_DIRECT_EFFORT','ETA_DIRECT_MATCHED')
    trained_source=source if source_needed else 0;trained_local=local if local_needed else 0
    inherited=sum(_count(f.module) for f in formed.functions.values() if f.metadata.get('reused_from')) if formed else 0
    additions=new+reader+direct
    if method=='SOURCE_LABEL_SHUFFLE':additions+=source;trained_source=0
    frozen_base=base if base_needed and method!='B_REFIT' else 0
    candidates=[c for c in pred.diagnostics.get('candidates',[]) if isinstance(c,dict) and not c.get('off',False)]
    fitted_reader_instances=sum(sum(2+len(np.asarray(v['weights'])) for v in c.get('coefficients',{}).values()) for c in candidates)
    if method=='B_REFIT':additions=base
    if method=='NATIVE':additions=0
    return dict(trainable_additions_during_method=additions,
        frozen_pretrained_added_parameters_required=frozen_base+trained_source+trained_local+inherited,
        final_trained_stage_parameter_instances=_count(bundle.native)+additions+frozen_base+trained_source+trained_local+inherited,
        reader_optimization_variables_per_candidate=reader,
        reader_fitted_parameter_instances_including_discarded_candidates=fitted_reader_instances,
        selected_reader_variables=0 if off else reader,inherited_formed_parameters=inherited,
        discarded_B_OOF_fitted_parameter_instances=2*base if base_needed else 0,
        shared_prefix_research_parameters=dict(native=_count(bundle.native),source_pilots=source,local_pilots=local,base_refit=base),
        definition='Stage parameter instances: inherited/refit stages can share a lineage, so this is not independent capacity. OOF discarded fits separate. Stored deployment uses fused parameter count separately.')


def align_native_terminal_reporting(bundle,predictions,output_root):
    """Report the actually recorded epoch24 endpoint, preserving selected policy.

    This corrects only the NATIVE PredictionBundle's endpoint data. It neither
    evaluates/reconstructs a model nor changes selected probabilities or selection.
    All companions are input-derived logits/probabilities, never state tensors.
    """
    pred=predictions.get('NATIVE')
    logits=bundle.diagnostics.get('native_terminal_logits')
    if pred is None:return dict(status='NOT_APPLICABLE',reason='NATIVE prediction absent')
    if not logits:return dict(status='MISSING',reason='Actual epoch24 logits unavailable; endpoint cannot be inferred from selected p')
    endpoint_hash=bundle.identity.get('native_epoch24_hash')
    if endpoint_hash is None:raise RuntimeError('Native epoch24 state identity missing for recorded endpoint')
    from .artifacts import _npz
    selected_before={role:array_hash(value) for role,value in pred.probabilities.items()}
    selection_before=digest(pred.selection);identity_before=digest(pred.identity)
    source_before={role:array_hash(value) for role,value in logits.items()}
    terminal={};files={};changed=[]
    for role,z in logits.items():
        if role not in bundle.views:raise ValueError('Native endpoint role missing from bundle views: '+role)
        view=bundle.views[role];z=np.asarray(z)
        if z.shape!=(len(view.row_id),2) or not np.isfinite(z).all():raise ValueError('Invalid actual epoch24 logit data '+role)
        probability=expit(z)  # Preserve actual FP32 native-logit sigmoid path.
        terminal[role]=probability
        prior=pred.terminal_probabilities.get(role)
        if prior is None or not np.array_equal(prior,probability):changed.append(role)
        path=Path(output_root)/'native_epoch24'/f'{role}.npz'
        arrays=dict(row_id=view.row_id,domain=view.domain,native_terminal_logits=z,
                    p=probability,epoch=np.asarray(24,dtype=np.int64),
                    native_state_hash=np.asarray(endpoint_hash))
        if path.exists():
            with np.load(path,allow_pickle=False) as existing:
                if set(existing.files)!=set(arrays) or any(array_hash(existing[k])!=array_hash(np.asarray(v)) for k,v in arrays.items()):
                    raise FileExistsError('Existing native endpoint companion differs: '+str(path))
            files[role]=dict(path=str(path.resolve()),sha256=_sha(path),reused_identical=True)
        else:files[role]=_npz(path,**arrays)
    pred.terminal_probabilities=terminal
    selected_after={role:array_hash(value) for role,value in pred.probabilities.items()}
    unchanged=(selected_before==selected_after and selection_before==digest(pred.selection)
               and identity_before==digest(pred.identity)
               and source_before=={role:array_hash(value) for role,value in logits.items()})
    if not unchanged:raise RuntimeError('Native endpoint reporting correction changed selected/source data')
    receipt=dict(status='ALIGNED_ACTUAL_ENDPOINT',revision='NATIVE_EPOCH24_REPORTING_V1',
        source_field='bundle.diagnostics.native_terminal_logits',native_epoch=24,
        terminal_state_hash=endpoint_hash,selected_native_state_hash=bundle.identity.get('native_hash'),
        changed_roles=changed,role_companions=files,
        selected_probabilities_selection_identity_and_source_logits_unchanged=True,
        selected_probability_hashes=selected_after,
        terminal_probability_hashes={r:array_hash(p) for r,p in terminal.items()},
        no_new_forward_or_training=True,model_files_written=0)
    _json(Path(output_root)/'NATIVE_ENDPOINT_REPORTING_RECEIPT.json',receipt)
    return receipt


def collect_delivery(bundle,predictions,formations,diagnostics,config,output_root=None,benchmark=False,*,allowed_benchmark_methods=None,preserved_measurements=()):
    """Merge identity-bound observations; summaries never erase real timings."""
    from .delivery_observation_preservation import MeasurementArchive,atomic_json,REVISION
    root=Path(output_root or config['paths']['artifact'])/'delivery'/bundle.dataset/f'seed{bundle.seed}'
    root.mkdir(parents=True,exist_ok=True)
    endpoint_reporting=align_native_terminal_reporting(bundle,predictions,root)
    costs=_training_costs(bundle,predictions,formations)
    source=dict(module_sha256=_sha(Path(__file__)),prefix_identity=bundle.identity,
                revision=REVISION,native_execution_identity_preserved=True,retroactive_old_timing_claim=False)
    archive=MeasurementArchive(bundle,predictions,formations,config,root,snapshots=preserved_measurements)
    permitted=set(predictions) if allowed_benchmark_methods is None else set(allowed_benchmark_methods)
    if not permitted<=set(predictions):raise ValueError('Unknown authorized benchmark method')
    result=dict(dataset=bundle.dataset,seed=bundle.seed,costs=costs,parameters={},inference={},
                source=source,benchmark_requested=bool(benchmark),current_call_benchmark_requested=bool(benchmark),
                measurement_preservation_revision=REVISION,required_methods=list(REQUIRED),
                absent_methods=[m for m in REQUIRED if m not in predictions],
                no_new_training=True,model_files_written=0,native_endpoint_reporting=endpoint_reporting,
                measurement_actions={},summary_path=str((root/'COSTS_PARAMETERS_AND_INFERENCE.json').resolve()))
    before=_observation_fingerprint(bundle,predictions,formations) if benchmark else None
    try:
        for method,pred in predictions.items():
            prior=archive.lookup(method)
            try:
                compiled=FrozenDeployment(bundle,pred,method)
                deployed=compiled.parameter_counts();parameters=dict(deployed)
                parameters.update(_training_parameters(bundle,pred,method,formations))
                result['parameters'][method]=parameters
                del compiled
                if prior is not None and prior.get('status')=='MEASURED' and prior.get('parameters')!=_plain(deployed):
                    archive.rejections.append(dict(method=method,reason='Current deployed parameter recipe differs from saved measurement'))
                    prior=None
                if prior is not None and prior['status']=='MEASURED':
                    result['inference'][method]=prior;result['measurement_actions'][method]='REUSED_VERIFIED_MEASURED'
                elif prior is not None and method not in REQUIRED:
                    result['inference'][method]=prior;result['measurement_actions'][method]='PRESERVED_NONREQUIRED_FAILURE'
                elif benchmark and method in permitted:
                    measured=benchmark_predictor(bundle,pred,method)
                    result['inference'][method]=archive.save(method,measured,dict(kind='actual_frozen_benchmark',
                        observed_at_unix=time.time(),delivery_module_sha256=source['module_sha256'],
                        preservation_module_sha256=_sha(Path(__file__).with_name('delivery_observation_preservation.py')),
                        reason='Missing/lost same-identity measurement; original benchmark unchanged'))
                    result['measurement_actions'][method]='BENCHMARKED_MISSING_AUTHORIZED'
                elif prior is not None:
                    result['inference'][method]=prior;result['measurement_actions'][method]='PRESERVED_NOT_MEASURED'
                else:
                    result['inference'][method]=dict(status='NOT_MEASURED',reason='No verified measurement; timing not requested or method not authorized')
                    result['measurement_actions'][method]='NOT_REQUESTED_OR_AUTHORIZED'
            except (NotImplementedError,RuntimeError,ValueError,KeyError) as error:
                result['parameters'].setdefault(method,dict(status='NOT_MEASURED',reason=str(error),**_training_parameters(bundle,pred,method,formations)))
                if prior is not None and prior.get('status')=='MEASURED':
                    result['inference'][method]=prior;result['measurement_actions'][method]='PRESERVED_MEASURED_PARAMETER_OBSERVER_ERROR'
                    result['parameters'][method]['observer_error']=f'{type(error).__name__}: {error}'
                else:
                    failure=dict(status='NOT_MEASURED',reason=f'{type(error).__name__}: {error}')
                    result['inference'][method]=archive.save(method,failure,dict(kind='actual_observer_failure',observed_at_unix=time.time(),delivery_module_sha256=source['module_sha256']))
                    result['measurement_actions'][method]='RECORDED_OBSERVER_FAILURE'
    finally:
        if benchmark:
            after=_observation_fingerprint(bundle,predictions,formations)
            result['immutability']=dict(before_hash=digest(before),after_hash=digest(after),
                parameters_modes_gradients_normalizers_rng_unchanged=before==after)
            if before!=after:raise RuntimeError('Observational delivery altered resident predictor state/mode/RNG')
    result['missing_required_methods']=[m for m in REQUIRED if result['inference'].get(m,{}).get('status')!='MEASURED']
    result['required_measurement_status']='COMPLETE' if not result['missing_required_methods'] else 'INCOMPLETE'
    result['status']='MEASURED' if result['inference'] and all(v['status']=='MEASURED' for v in result['inference'].values()) else 'PARTIAL_OBSERVATIONS'
    result['measurement_source_rejections']=archive.rejections
    atomic_json(root/'COSTS_PARAMETERS_AND_INFERENCE.json',result)
    import io,os,tempfile
    buffer=io.StringIO();keys=list(dict.fromkeys(k for row in costs for k in row));writer=csv.DictWriter(buffer,fieldnames=keys);writer.writeheader()
    for row in costs:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    fd,tmp=tempfile.mkstemp(dir=root,prefix='training_costs.',suffix='.data_tmp')
    try:
        with os.fdopen(fd,'w',newline='') as stream:stream.write(buffer.getvalue())
        os.replace(tmp,root/'training_costs.csv')
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return result
