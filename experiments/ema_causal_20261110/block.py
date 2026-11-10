"""Finalize one live five-seed block before its parameter packets are released."""
import copy
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.aug_history_20261105.model_adapter import build_model
from experiments.aug_history_20261105.registry import ParameterLayout
from experiments.aug_history_20261105.group_registry import build_path_groups
from experiments.ema_mechanism_20261106.panels import load_panels
from experiments.ema_mechanism_20261106.readouts import panel_predictions
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.state import read_json, write_json, sha_file, digest
from experiments.mdmt_history_information_20261103.protocol import metrics as source_metrics
from . import functional as fn
from .bridge import run_bridge, seed_summary, choose, action_order, _mean
from .interventions import build_state, group_masks, vector_hash, matched_states, source_direction_states
from .pairs import exact_transition_table, transition_metrics, sample_pairs, panel_pair_statistics, pair_overlap, source_overlap
from .protocol import WINDOWS, SEEDS
from .state import request, disk_admit
from .trajectory import full_prediction, panel_metrics

FAMILIES = {'S': ('EMBED','STAR_SKIP','EXPERT'), 'N': ('D_ALL','T_ALL','DT_ALL')}
FUSION = ('content','routing','interaction','domain','task')


def _plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_plain(v) for v in value]
    return value


def _descriptor(background='O',group='all',component='full',operation='splice',h=None):
    result=dict(background=background,group=group,component=component,operation=operation)
    if h is not None: result['h']=int(h)
    return result


def _id(descriptor):
    return '/'.join(str(descriptor.get(k,'')) for k in ('background','group','component','operation','h'))


def _mac(metrics,pop='V_A',metric='auc'):
    return _mean([r[metric] for r in metrics[pop]])


def _full_cells(metrics):
    return [dict(auc=metrics[f'd{d}t{t}_auc'],logloss=metrics[f'd{d}t{t}_logloss']) for d in range(3) for t in range(2)]


def _panel_predictions(model,panels,operator=None):
    if operator is None: return {name:panel_predictions(model,p) for name,p in panels.items()}
    device=next(model.parameters()).device;result={}
    with torch.no_grad():
        for name,panel in panels.items():
            natural=panel['natural_x'];valid=panel.get('valid_mask',np.ones(natural.shape[:2],bool))
            p=np.zeros((*natural.shape[:2],2),np.float32);z=np.zeros_like(p)
            value=operator(torch.as_tensor(natural[valid].copy(),device=device).T)
            p[valid]=value.p.cpu().numpy();z[valid]=value.z.cpu().numpy()
            if name=='V_mech': result[name]={'p':p,'z':z}
            else:
                rank=panel['rank_x'];value=operator(torch.as_tensor(rank.reshape(-1,rank.shape[-1]).copy(),device=device).T)
                rp=value.p.reshape(6,512,2).cpu().numpy();rz=value.z.reshape(6,512,2).cpu().numpy()
                result[name]=dict(natural_p=p,natural_z=z,rank_p=np.stack([rp[c,:,c%2] for c in range(6)]),rank_z=np.stack([rz[c,:,c%2] for c in range(6)]))
    return result


def _view(panel,pred,domain,task,population):
    if population in ('P_A','P_B'):
        return panel['rank_y'][2*domain+task],pred['rank_p'][2*domain+task],pred['rank_z'][2*domain+task]
    mask=panel['valid_mask'][domain].copy()
    if population in ('V_A','V_B'): mask &= panel['validation_row_side'][domain]==int(population=='V_B')
    return panel['natural_y'][domain,mask,task],pred['p'][domain,mask,task],pred['z'][domain,mask,task]


def _row_ids(panel,domain,task,population):
    if population in ('P_A','P_B'):return panel['rank_row_ids'][2*domain+task]
    mask=panel['valid_mask'][domain].copy()
    if population in ('V_A','V_B'):mask &= panel['validation_row_side'][domain]==int(population=='V_B')
    return panel['natural_row_ids'][domain,mask]


def _pair_seed(job,population,domain,task):
    return int.from_bytes(hashlib.sha256(f"{job['dataset']}|{job['seed']}|{population}|{domain}|{task}".encode()).digest()[:8],'little')


def _diagnostic_pairs(job,panel,domain,task):
    valid=panel['valid_mask'][domain];y=panel['natural_y'][domain,valid,task]
    pairs=sample_pairs(y,_pair_seed(job,'V_mech',domain,task));rows=_row_ids(panel,domain,task,'V_mech')
    keys=[hashlib.sha256(np.asarray(rows[pair],dtype=np.int64).tobytes()).hexdigest() for pair in pairs]
    chosen=sorted(range(len(pairs)),key=lambda i:keys[i])[:8]
    identities=[dict(pair_id=f"{job['dataset']}:d{domain}:t{task}:"+keys[i],pairset_index=i,domain=domain,task=task,positive_row_id=int(rows[pairs[i,0]]),negative_row_id=int(rows[pairs[i,1]])) for i in chosen]
    return pairs,chosen,identities


def _discovery_bounds(job,panels,reference,population,domain,task):
    discovery='P_A' if population in ('P_A','P_B') else 'V_A'
    key='P_A' if discovery=='P_A' else 'V_mech';panel=panels[key]
    y,p,_=_view(panel,reference[key],domain,task,discovery)
    pairs=sample_pairs(y,_pair_seed(job,discovery,domain,task))
    bounds=np.quantile(np.abs(p[pairs[:,0]].astype('float64')-p[pairs[:,1]].astype('float64')),[.25,.5,.75]).tolist() if len(pairs) else []
    return bounds,dict(discovery_population=discovery,strata_rule='reference probability absolute margin quartiles; discovery frozen across confirmation',strata_rule_hash=digest(dict(discovery=discovery,bounds=bounds)))


class Anchor:
    def __init__(self,out,job,row,mark,bundle,device,tables):
        self.out=out;self.job=job;self.row=row;self.mark=mark;self.bundle=bundle;self.tables=tables
        self.packet=row['state']['selected_packet'] if mark=='tF' else row['packets'][mark]
        self.model=build_model(bundle,job['config']).to(device).requires_grad_(False)
        self.layout=ParameterLayout.from_model(self.model);self.masks=group_masks(build_path_groups(self.model,bundle))
        self.panels=load_panels(out,job['dataset'],job['seed'],bundle)
        self.identity=dict(dataset=job['dataset'],objective=job['objective'],seed=job['seed'],anchor=mark,
                           epoch=self.packet['epoch'],step=self.packet['step'])
        self.base={};self.scalar={}
        for name,vector in (('O',self.packet['online_fp32']),('F',self.packet['F_fp32'])):
            self.base[name]=self.read(vector)
            self.scalar[name]=self.base[name]['metrics']
        existing={r['descriptor_id'] for r in tables['STATE_DESCRIPTORS']
                  if r['seed']==job['seed'] and r['anchor']==mark}
        for name in ('O','F'):
            if name not in existing:
                self.record(name,self.packet['online_fp32' if name=='O' else 'F_fp32'],self.base[name])

    def read(self,vector,operator=None):
        self.layout.copy_to(self.model,vector,self.packet.get('buffers'))
        full,p,z=full_prediction(self.model,self.bundle,self.job['config']['eval_batch_size'],operator=operator)
        pp=_panel_predictions(self.model,self.panels,operator)
        metrics=panel_metrics(self.panels,pp);metrics['V']=_full_cells(full)
        return dict(metrics=metrics,full_metrics=full,p=p,z=z,panels=pp)

    def evaluate(self,name,descriptor=None,vector=None,operator=None,reference='O'):
        if vector is None: vector,_=build_state(self.packet,descriptor,self.masks)
        reused=None
        if operator is None:
            for endpoint,key in (('O','online_fp32'),('F','F_fp32')):
                if torch.equal(vector,self.packet[key]):
                    reused=endpoint;break
        data=self.base[reused] if reused else self.read(vector,operator)
        metadata=dict(descriptor or {},physical_reused_from=reused)
        self.record(name,vector,data,metadata,reference)
        self.scalar[name]=data['metrics']
        return dict(data,state_id=f"{self.job['job_id']}:{self.mark}:{name}")

    def record(self,name,vector,data,descriptor=None,reference='O'):
        state_id=f"{self.job['job_id']}:{self.mark}:{name}"
        common=dict(self.identity,descriptor_id=name,state_id=state_id,state_hash=vector_hash(vector),
                    descriptor=descriptor or {'endpoint':name},reference_id=reference,reference_state_id=self.base[reference].get('state_id',f"{self.job['job_id']}:{self.mark}:{reference}"))
        for population,values in data['metrics'].items():
            for cell,value in enumerate(values):
                self.tables['INTERVENTION_METRICS'].append(dict(common,population=population,domain=cell//2,task=cell%2,**value))
        self.tables['STATE_DESCRIPTORS'].append(dict(common,full_metrics=data['full_metrics']))
        ref=self.base[reference]
        for d in range(3):
            mask=self.bundle.val_x[:,-1]==d
            for t in range(2):
                y=self.bundle.val_y[mask,t];a=ref['p'][mask,t];b=data['p'][mask,t]
                table=exact_transition_table(y,a,b);stats=transition_metrics(table)
                if stats.get('delta_auc') is not None:
                    actual=data['metrics']['V'][2*d+t]['auc']-ref['metrics']['V'][2*d+t]['auc']
                    if abs(actual-stats['delta_auc'])>1e-12: raise ArithmeticError('Population AUC/pair closure failed')
                self.tables['PAIR_TRANSITIONS'].append(dict(common,population='V',domain=d,task=t,counts=table.tolist(),**stats))
        self.panel_records(common,data['panels'],ref['panels'])

    def panel_records(self,common,new,reference):
        for population in ('P_A','P_B','V_mech','V_A','V_B'):
            panel=self.panels[population if population in ('P_A','P_B') else 'V_mech']
            key=population if population in ('P_A','P_B') else 'V_mech'
            for d in range(3):
                for t in range(2):
                    y,a,za=_view(panel,reference[key],d,t,population);_,b,zb=_view(panel,new[key],d,t,population)
                    seed=int.from_bytes(hashlib.sha256(f"{self.job['dataset']}|{self.job['seed']}|{population}|{d}|{t}".encode()).digest()[:8],'little')
                    pairs=sample_pairs(y,seed)
                    row_ids=_row_ids(panel,d,t,population)
                    pair_common=dict(common,pairset_hash=hashlib.sha256(np.asarray(row_ids[pairs],dtype=np.int64).tobytes()).hexdigest(),
                                     pair_seed=seed,pair_rule='Floyd uniform without replacement; limit=32768; dataset/seed/population/domain/task')
                    table=exact_transition_table(y,a,b)
                    self.tables['PAIR_TRANSITIONS'].append(dict(pair_common,population=population,domain=d,task=t,counts=table.tolist(),**transition_metrics(table)))
                    if not len(pairs): continue
                    bounds,strata_identity=_discovery_bounds(self.job,self.panels,reference,population,d,t)
                    stats=panel_pair_statistics(y,a,b,pairs,za,zb,bounds=bounds)
                    stats.update(strata_identity)
                    self.tables['PAIR_STRATA'].append(dict(pair_common,population=population,domain=d,task=t,**stats))
                    _,f,_=_view(panel,self.base['F']['panels'][key],d,t,population)
                    self.tables['PAIR_OVERLAP'].append(dict(pair_common,population=population,domain=d,task=t,compared_state='F',**pair_overlap(a,f,b,pairs)))
                    if population=='V_B' and common['descriptor_id'] in ('F',) or (population=='V_B' and common['descriptor_id'].startswith('D/')):
                        margin0=a[pairs[:,0]]-a[pairs[:,1]];margin1=b[pairs[:,0]]-b[pairs[:,1]]
                        codes=(np.sign(margin0).astype(int)+1)*3+np.sign(margin1).astype(int)+1
                        for code in range(9):
                            selected=sorted(np.flatnonzero(codes==code),key=lambda i:hashlib.sha256(np.asarray(row_ids[pairs[i]],dtype=np.int64).tobytes()).digest())[:8]
                            for i in selected:
                                self.tables['PAIR_CASES'].append(dict(pair_common,population=population,domain=d,task=t,
                                    pair_index=int(i),positive_index=int(pairs[i,0]),negative_index=int(pairs[i,1]),
                                    positive_row_id=int(row_ids[pairs[i,0]]),negative_row_id=int(row_ids[pairs[i,1]]),
                                    transition_code=code,margin_reference=float(margin0[i]),margin_changed=float(margin1[i])))

    def factorial(self,name,states,kind,extra=None):
        values=[self.scalar[s] if isinstance(s,str) else s['metrics'] for s in states]
        for population in values[0]:
            for cell in range(6):
                for metric in ('auc','logloss'):
                    u=[v[population][cell][metric] for v in values]
                    result=dict(self.identity,contrast_id=name,population=population,domain=cell//2,task=cell%2,metric=metric,states=[f"{self.job['job_id']}:{self.mark}:{s}" if isinstance(s,str) else s['state_id'] for s in states])
                    if all(v is not None for v in u):
                        result.update(independent_first=u[1]-u[0],independent_second=u[2]-u[0],deletion_first=u[3]-u[2],deletion_second=u[3]-u[1],interaction=u[3]-u[1]-u[2]+u[0],total=u[3]-u[0])
                    else: result.update(independent_first=None,independent_second=None,deletion_first=None,deletion_second=None,interaction=None,total=None)
                    self.tables[kind].append(dict(result,**(extra or {})))

    def source(self,h,recent):
        origin=self.packet['origins'][h]
        for pop in ('P_A','P_B','V_mech'):
            panel=self.panels[pop]
            for d in range(3):
                for t in range(2):
                    y,past,_=_view(panel,origin['predictions'][pop],d,t,pop)
                    _,online,_=_view(panel,self.base['O']['panels'][pop],d,t,pop)
                    _,repaired,_=_view(panel,recent['panels'][pop],d,t,pop)
                    pairs=sample_pairs(y,int.from_bytes(hashlib.sha256(f"{self.job['dataset']}|{self.job['seed']}|{pop}|{d}|{t}".encode()).digest()[:8],'little'))
                    if not len(pairs): continue
                    self.tables['SOURCE_OVERLAP'].append(dict(self.identity,population=pop,domain=d,task=t,h=h,
                        origin_step=origin['step'],pb_objective_change=self.packet['pb_objective']-origin['pb_objective'],
                        **source_overlap(past,online,repaired,pairs)))

    def ensembles(self):
        q=self.packet['q']
        for kind in ('probability','logit'):
            pred={}
            for pop,p in q.items():
                if kind=='probability': pred[pop]=copy.deepcopy(p)
                else:
                    pred[pop]={}
                    for name,values in p.items():
                        if name.endswith('z') or name=='z':
                            z=values.astype(np.float64);pkey='p' if name=='z' else name[:-1]+'p'
                            pred[pop][name]=z;pred[pop][pkey]=np.exp(-np.logaddexp(0.,-z))
            metrics=panel_metrics(self.panels,pred)
            for population,values in metrics.items():
                for c,v in enumerate(values): self.tables['ENSEMBLE_METRICS'].append(dict(self.identity,readout=kind,population=population,domain=c//2,task=c%2,**v))
            self.panel_records(dict(self.identity,descriptor_id='q_'+kind,state_id=f"{self.job['job_id']}:{self.mark}:q_{kind}",reference_id='O'),pred,self.base['O']['panels'])


def _rows(out,jobs):
    for job in jobs:
        row=request(out,'get',key=job['job_id'])['row']
        if row is None or row['status']!='COMPLETE' or row['epoch']!=50: raise RuntimeError('Five complete live seed packets required')
        yield job,row


def _anchors(out,jobs,prepared,device,tables,marks):
    for job,row in _rows(out,jobs):
        bundle=cached_bundle(prepared['cache_root'],job['dataset'],include_test=False)
        for mark in marks:
            physical='E'+str(row['state']['selected_packet']['epoch'])
            if mark=='tF' and physical in marks:
                for name,items in list(tables.items()):
                    if not isinstance(items,list):continue
                    existing={digest(v) for v in items}
                    aliases=[]
                    for item in items:
                        if item.get('seed')!=job['seed'] or item.get('anchor')!=physical:continue
                        alias=copy.deepcopy(item);alias['anchor']='tF'
                        if 'state_id' in alias:alias['state_id']=alias['state_id'].replace(':'+physical+':',':tF:')
                        if digest(alias) not in existing:aliases.append(alias);existing.add(digest(alias))
                    items.extend(aliases)
                tables['ANCHOR_ALIASES'].append(dict(job_id=job['job_id'],seed=job['seed'],anchor='tF',physical_anchor=physical))
                continue
            yield Anchor(out,job,row,mark,bundle,device,tables)


def _mean_effect(tables,ids,population='V_A',anchors=('E35','E50')):
    rows=tables['INTERVENTION_METRICS'];values={}
    for seed in SEEDS:
        differences=[]
        for anchor in anchors:
            vals=[]
            for name in ids:
                selected=[r['auc'] for r in rows if r['seed']==seed and r['anchor']==anchor and r['descriptor_id']==name and r['population']==population]
                # A later pass may reproduce endpoints; their deterministic metric is unchanged.
                selected=selected[-6:]
                vals.append(_mean(selected) if len(selected)==6 else None)
            differences.append(vals[0]-vals[1] if all(v is not None for v in vals) else None)
        values[seed]=_mean(differences)
    return seed_summary(values)


def _core(anchor):
    states={}
    for g in ('S','N'):
        descriptor=_descriptor(group=g);states[g]=anchor.evaluate('B/'+g,descriptor)
    anchor.factorial('B/SxN',['O','B/S','B/N','F'],'FACTORIAL_EFFECTS')
    _factorial_function(anchor,'B/SxN',[anchor.base['O'],states['S'],states['N'],anchor.base['F']])
    del states
    for h in sorted(anchor.packet['old_R_fp64_by_effective_h']):
        recent=anchor.evaluate(f'C/{h}/recent',_descriptor(component='recent',operation='add',h=h))
        old=anchor.evaluate(f'C/{h}/old',_descriptor(component='old',operation='add',h=h))
        anchor.factorial(f'C/{h}',['O',f'C/{h}/recent',f'C/{h}/old','F'],'TEMPORAL_EFFECTS',dict(group='all',background='O',h=h))
        _factorial_function(anchor,f'C/{h}',[anchor.base['O'],recent,old,anchor.base['F']])
        anchor.source(h,recent)
    anchor.ensembles()


def _selection(tables):
    s=_mean_effect(tables,('F','B/N'));n=_mean_effect(tables,('F','B/S'))
    records=[r for r in tables['FACTORIAL_EFFECTS'] if r['contrast_id']=='B/SxN' and r['anchor'] in ('E35','E50') and r['population']=='V_A' and r['metric']=='auc']
    interaction=seed_summary({seed:_mean([r['interaction'] for r in records if r['seed']==seed]) for seed in SEEDS})
    if any(r['status']!='COMPLETE' for r in (s,n,interaction)): raise RuntimeError('Core five-seed selection dependencies incomplete')
    chosen=[g for g,e in (('S',s),('N',n)) if abs(e['mean'])>=.0002 or abs(interaction['mean'])>=.0002]
    return dict(selected=chosen,deletion={'S':s,'N':n},interaction=interaction,epsilon=.0002,
                candidates=[sub for g in ('S','N') for sub in (FAMILIES[g] if g in chosen else (g,))])


def _fine(anchor,selection):
    for parent in selection['selected']:
        for group in FAMILIES[parent]:
            for background,operation in (('O','splice'),('F','delete')):
                family=anchor.evaluate(f'B2/{group}/{background}',_descriptor(background,group,operation=operation))
                leaves={'D_ALL':[f'D{d}' for d in range(3)],'T_ALL':['T0','T1'],
                        'DT_ALL':[f'DT{d}{t}' for d in range(3) for t in range(2)]}.get(group,[])
                for leaf in leaves:
                    cells=[(d,t) for d in range(3) for t in range(2)
                           if leaf in (f'D{d}',f'T{t}',f'DT{d}{t}')]
                    data=copy.deepcopy(anchor.base[background])
                    for d,t in cells:
                        mask=anchor.bundle.val_x[:,-1]==d
                        for key in ('p','z'): data[key][mask,t]=family[key][mask,t]
                        for pop in ('P_A','P_B','V_mech'):
                            for suffix in ('p','z'):
                                key=suffix if pop=='V_mech' else 'natural_'+suffix
                                data['panels'][pop][key][d,:,t]=family['panels'][pop][key][d,:,t]
                                if pop!='V_mech': data['panels'][pop]['rank_'+suffix][2*d+t]=family['panels'][pop]['rank_'+suffix][2*d+t]
                    data['full_metrics']=source_metrics(anchor.bundle.val_y,data['p'],anchor.bundle.val_x[:,-1])
                    data['metrics']=panel_metrics(anchor.panels,data['panels']);data['metrics']['V']=_full_cells(data['full_metrics'])
                    descriptor=_descriptor(background,leaf,operation=operation)
                    vector,_=build_state(anchor.packet,descriptor,anchor.masks)
                    anchor.record(f'B_LEAF/{leaf}/{background}',vector,data,dict(descriptor,readout='audited_structural_support_recomposition'))


def _time_groups(anchor,groups):
    h=min(WINDOWS[anchor.job['dataset']][1],anchor.packet['step'])
    for group in groups:
        whole,_=build_state(anchor.packet,_descriptor(group=group),anchor.masks)
        removed,_=build_state(anchor.packet,_descriptor('F',group,operation='delete'),anchor.masks)
        # F-background factorial starts from F with this group's full R removed.
        for background,start,end in (('O',anchor.packet['online_fp32'],whole),('F',removed,anchor.packet['F_fp32'])):
            local=dict(anchor.packet,online_fp32=start,F_fp32=end)
            names=[];readouts=[]
            for component in ('zero','recent','old','full'):
                name=f'C2/{group}/{background}/{component}';names.append(name)
                if component=='zero': vector=start
                elif component=='full': vector=end
                else: vector,_=build_state(local,_descriptor(group=group,component=component,operation='add',h=h),anchor.masks)
                readouts.append(anchor.evaluate(name,vector=vector,descriptor=dict(background=background,group=group,component=component,h=h)))
            anchor.factorial('/'.join(names[0].split('/')[:-1]),names,'TEMPORAL_EFFECTS',dict(group=group,background=background,h=h))
            _factorial_function(anchor,'/'.join(names[0].split('/')[:-1]),readouts)


def _contrast_candidates(groups,selection):
    result={}
    for group in dict.fromkeys(('S','N',*selection['candidates'])):
        name=f'B2/{group}/F' if group not in ('S','N') else ('B/N' if group=='S' else 'B/S')
        result[name]=_descriptor('F',group,operation='delete')
    for component in ('recent','old'):
        # Removing recent leaves old, removing old leaves recent.
        result['C_PRIMARY/'+component]=_descriptor('F','all',component,'delete')
    for group in groups:
        for component in ('recent','old'):
            other='old' if component=='recent' else 'recent'
            result[f'C2/{group}/F/{other}']=_descriptor('F',group,component,'delete')
    return result


def _function_contrast(anchor,label,theta0,theta1):
    first=anchor.read(theta0);second=anchor.read(theta1)
    first['state_id']=f"{anchor.job['job_id']}:{anchor.mark}:D/{label}/00"
    second['state_id']=f"{anchor.job['job_id']}:{anchor.mark}:D/{label}/11"
    anchor.base[label+'_0']=first
    for name,vector,data in (('00',theta0,first),('11',theta1,second)):
        anchor.record(f'D/{label}/{name}',vector,data,dict(contrast=label,endpoint=name),reference=label+'_0')
    panel=anchor.panels['V_mech']
    for cut in ('h','fused'):
        output=[first,None,None,second]
        for i,(donor,recipient) in enumerate(((theta1,theta0),(theta0,theta1)),1):
            def operator(x,donor=donor,recipient=recipient,cut=cut):
                anchor.layout.copy_to(anchor.model,donor,anchor.packet.get('buffers'));values=fn.capture_cut(anchor.model,x,cut)
                anchor.layout.copy_to(anchor.model,recipient,anchor.packet.get('buffers'));return fn.mixed_readout(anchor.model,x,cut,values)
            output[i]=anchor.evaluate(f'D/{label}/{cut}/{i}',vector=recipient,operator=operator,
                reference=label+'_0',descriptor=dict(contrast=label,cut=cut,donor='1' if i==1 else '0',recipient='0' if i==1 else '1',donor_hash=vector_hash(donor),recipient_hash=vector_hash(recipient)))
        state_ids=[v['state_id'] for v in output]
        anchor.factorial(f'D/{label}/{cut}',output,'FUNCTIONAL_EFFECTS',dict(contrast=label,cut=cut))
        for d in range(3):
            for t in range(2):
                pairs,chosen,identities=_diagnostic_pairs(anchor.job,panel,d,t)
                if not len(pairs):continue
                margins={space:[] for space in ('probability','logit')}
                for item in output:
                    _,p,z=_view(panel,item['panels']['V_mech'],d,t,'V_mech')
                    for space,values in (('probability',p),('logit',z)):
                        v=values.astype('float64');margins[space].append(v[pairs[:,0]]-v[pairs[:,1]])
                weights=[(np.sign(m)+1)/2 for m in margins['probability']]
                common=dict(anchor.identity,contrast=label,cut=cut,population='V_mech',domain=d,task=t,state_ids=state_ids,pair_count=len(pairs),cut_complete=True,endpoint_definition='unmodified theta0/theta1 actual source forward')
                for space,values in margins.items():
                    m00,m10,m01,m11=values;a=m10-m00;b=m01-m00;interaction=m11-m10-m01+m00
                    residual=m11-m00-a-b-interaction
                    summary=dict(common,space=space,margin_mean=[float(m.mean()) for m in values],representation_mean=float(a.mean()),downstream_mean=float(b.mean()),interaction_mean=float(interaction.mean()),interaction_rms=float(np.sqrt(np.mean(interaction**2))),closure_max=float(np.max(np.abs(residual))),net=float((weights[3]-weights[0]).mean()),representation_net=float((weights[1]-weights[0]).mean()),downstream_net=float((weights[2]-weights[0]).mean()),lost_correction_without_representation=float(np.minimum(np.maximum(weights[3]-weights[0],0),np.maximum(weights[3]-weights[2],0)).mean()),lost_correction_without_downstream=float(np.minimum(np.maximum(weights[3]-weights[0],0),np.maximum(weights[3]-weights[1],0)).mean()))
                    anchor.tables['FUNCTIONAL_MARGIN'].append(summary)
                    for index,identity in zip(chosen,identities):
                        anchor.tables['FUNCTIONAL_PAIR_DETAILS'].append(dict(common,**{k:v for k,v in identity.items() if k not in ('domain','task')},space=space,coverage='8 outcome-independent hash-ordered pairs per cell from fixed diagnostic pairset; population summaries use all diagnostic pairs',margin_00=float(m00[index]),margin_10=float(m10[index]),margin_01=float(m01[index]),margin_11=float(m11[index]),representation=float(a[index]),downstream=float(b[index]),interaction=float(interaction[index]),rank_states=[float(v[index]) for v in weights],net_auc_pair_contribution=float((weights[3][index]-weights[0][index])/len(pairs))))
    xs=[];pairs=[];tasks=[];pair_ids=[]
    for d in range(3):
        valid=panel['valid_mask'][d];x=panel['natural_x'][d,valid]
        for t in range(2):
            sampled,chosen,identities=_diagnostic_pairs(anchor.job,panel,d,t)
            for index,identity in zip(chosen,identities):
                pair=sampled[index];offset=len(xs);xs.extend([x[pair[0]],x[pair[1]]]);pairs.append([offset,offset+1]);tasks.append(t);pair_ids.append(identity)
    x=torch.as_tensor(np.asarray(xs),device=next(anchor.model.parameters()).device).T
    integral=fn.integrate_path(anchor.model,anchor.layout,x,theta0,theta1,{g:anchor.masks[g] for g in ('S','N')},pairs=np.asarray(pairs),task=np.asarray(tasks))
    anchor.tables['PATH_INTEGRALS'].append(dict(anchor.identity,contrast=label,state_ids=[first['state_id'],second['state_id']],diagnostic_rule='same eight outcome-independent hash-ordered pairs per cell as FUNCTIONAL_PAIR_DETAILS',pair_identities=pair_ids,**_plain(integral)))
    for i,identity in enumerate(pair_ids):
        anchor.tables['PATH_CLOSURE'].append(dict(anchor.identity,contrast=label,**identity,state_0=first['state_id'],state_1=second['state_id'],actual_delta_margin=float(integral['actual_delta'][i]),integral=float(integral['integral'][i]),absolute_residual=float(integral['absolute_residual'][i]),relative_residual=float(integral['relative_residual'][i]),nodes=integral['nodes'],sign_unresolved=bool(integral['sign_unresolved'][i]),group_integrals={k:float(v[i]) for k,v in integral['group_integrals'].items()}))
    del anchor.base[label+'_0']
    return first,second


def _fusion(anchor,component,theta0,theta1,label):
    residuals=[]
    def operator(x):
        anchor.layout.copy_to(anchor.model,theta0,anchor.packet.get('buffers'));a=fn.capture_fusion_parts(anchor.model,x)
        anchor.layout.copy_to(anchor.model,theta1,anchor.packet.get('buffers'));b=fn.capture_fusion_parts(anchor.model,x)
        parts,errors=fn.fusion_components(a,b);residuals.extend(errors)
        return fn.fusion_part_readout(anchor.model,x,b,parts[component])
    name=f'fusion/{label}/{component}'
    data=anchor.evaluate(name,vector=theta1,operator=operator,reference='F',descriptor=dict(contrast=label,component=component,background='F'))
    anchor.tables['FUSION_EFFECTS'].append(dict(anchor.identity,contrast=label,component=component,
        metrics=data['metrics'],deletion_V_A=_mac(anchor.base['F']['metrics'])-_mac(data['metrics']),
        algebra_relative_l2_max=max(r['algebra_relative_l2'] for r in residuals),
        source_rounding_max_abs=max(r['source_rounding_max_abs'] for r in residuals)))
    if label=='X_F':
        anchor.layout.copy_to(anchor.model,theta0,anchor.packet.get('buffers'))
        origin=_panel_predictions(anchor.model,anchor.panels)
        for population in ('V_A','V_B'):
            panel=anchor.panels['V_mech']
            for d in range(3):
                for t in range(2):
                    y,a,_=_view(panel,origin['V_mech'],d,t,population)
                    _,b,_=_view(panel,anchor.base['F']['panels']['V_mech'],d,t,population)
                    _,c,_=_view(panel,data['panels']['V_mech'],d,t,population)
                    pairs=sample_pairs(y,int.from_bytes(hashlib.sha256(f"{anchor.job['dataset']}|{anchor.job['seed']}|{population}|{d}|{t}".encode()).digest()[:8],'little'))
                    if not len(pairs): continue
                    rank=lambda p:(np.sign(p[pairs[:,0]]-p[pairs[:,1]])+1)/2
                    gain=rank(b)-rank(a);removal=rank(b)-rank(c)
                    anchor.tables['FUSION_CONNECTION'].append(dict(anchor.identity,contrast=label,component=component,population=population,domain=d,task=t,
                        lost_X_correction=float(np.minimum(np.maximum(gain,0),np.maximum(removal,0)).mean()),
                        X_net=float(gain.mean()),deletion_net=float(removal.mean()),**pair_overlap(a,b,c,pairs)))
    return data


def _save_tables(directory,tables):
    directory.mkdir(parents=True,exist_ok=True);index={}
    for name,rows in tables.items():
        path=directory/(name+'.json.gz');temporary=path.with_suffix('.tmp')
        with gzip.open(temporary,'wt',encoding='utf-8') as stream: json.dump(_plain(rows),stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        temporary.replace(path);index[name]=dict(path=str(path),format='json.gz',schema_version=1,sha256=sha_file(path),rows=len(rows) if isinstance(rows,list) else 1)
    write_json(directory/'ARTIFACT_INDEX.json',index);return index


def finalize_block(out,jobs,device):
    out=Path(out);jobs=sorted(jobs,key=lambda j:j['seed'])
    if [j['seed'] for j in jobs]!=SEEDS or len({(j['dataset'],j['objective']) for j in jobs})!=1: raise ValueError('A complete condition block is required')
    dataset,objective=jobs[0]['dataset'],jobs[0]['objective'];block=dataset+'__'+objective
    directory=out/'blocks'/block;prepared=read_json(out/'PREPARED.json');tables=defaultdict(list);start=time.time()
    marks=('E5','E10','E20','E35','E50','tF');fine_marks=('E10','E35','E50','tF')
    for anchor in _anchors(out,jobs,prepared,device,tables,marks): _core(anchor)
    selection=_selection(tables);tables['GROUP_SELECTION']=selection
    for anchor in _anchors(out,jobs,prepared,device,tables,fine_marks): _fine(anchor,selection)
    effects={g:_mean_effect(tables,('F',f'B2/{g}/F' if g not in ('S','N') else ('B/N' if g=='S' else 'B/S'))) for g in selection['candidates']}
    if any(v['status']!='COMPLETE' for v in effects.values()): raise RuntimeError('Fine group five-seed dependencies incomplete')
    groups=sorted(effects,key=lambda g:(-abs(effects[g]['mean']),g))[:2]
    selection.update(time_groups=groups,time_group_effects=effects)
    for anchor in _anchors(out,jobs,prepared,device,tables,fine_marks): _time_groups(anchor,groups)
    candidates=_contrast_candidates(groups,selection)
    # Evaluate the F-background full-parameter removals used to select the same X.
    for anchor in _anchors(out,jobs,prepared,device,tables,('E35','E50')):
        for component in ('recent','old'):
            descriptor=dict(candidates['C_PRIMARY/'+component],h=min(WINDOWS[dataset][1],anchor.packet['step']))
            anchor.evaluate('C_PRIMARY/'+component,descriptor)
    contrast_effects={name:_mean_effect(tables,('F',name)) for name in candidates}
    if any(v['status']!='COMPLETE' for v in contrast_effects.values()): raise RuntimeError('X selection dependencies incomplete')
    ranked=sorted(candidates,key=lambda name:(-contrast_effects[name]['mean'],name))
    xname=ranked[0] if ranked and contrast_effects[ranked[0]]['mean']>0 else None
    tables['CONTRAST_SELECTION']=dict(chosen=xname,descriptor=candidates.get(xname),effects=contrast_effects,
                                     status='SELECTED' if xname else 'NO_POSITIVE_COMPONENT_CONTRAST')
    for anchor in _anchors(out,jobs,prepared,device,tables,fine_marks):
        _function_contrast(anchor,'O_F',anchor.packet['online_fp32'],anchor.packet['F_fp32'])
        if xname:
            descriptor=dict(candidates[xname]);descriptor['h']=min(WINDOWS[dataset][1],anchor.packet['step'])
            theta0,_=build_state(anchor.packet,descriptor,anchor.masks)
            _function_contrast(anchor,'X_F',theta0,anchor.packet['F_fp32'])
    for anchor in _anchors(out,jobs,prepared,device,tables,('E35','tF')):
        for component in FUSION: _fusion(anchor,component,anchor.packet['online_fp32'],anchor.packet['F_fp32'],'O_F')
    fusion_effects={c:seed_summary({s:_mean([r['deletion_V_A'] for r in tables['FUSION_EFFECTS'] if r['seed']==s and r['anchor']=='E35' and r['component']==c]) for s in SEEDS}) for c in FUSION}
    top=max(FUSION,key=lambda c:fusion_effects[c]['mean'])
    selected_component=top if fusion_effects[top]['mean']>0 else None
    tables['FUSION_SELECTION']=dict(chosen=selected_component,effects=fusion_effects)
    if selected_component and xname:
        for anchor in _anchors(out,jobs,prepared,device,tables,('E35','tF')):
            descriptor=dict(candidates[xname],h=min(WINDOWS[dataset][1],anchor.packet['step']))
            theta0,_=build_state(anchor.packet,descriptor,anchor.masks)
            _fusion(anchor,selected_component,theta0,anchor.packet['F_fp32'],'X_F')
    pulse=[]
    for job in jobs:
        for path in sorted((out/'runs'/job['job_id']/'epochs').glob('E*.json')): pulse.extend(read_json(path).get('pulse',[]))
    bridge=run_bridge(pulse);tables.update(bridge)
    _conditional(out,jobs,prepared,device,tables,selection,candidates.get(xname),groups)
    index=_save_tables(directory,tables);disk_admit(out)
    receipt=dict(status='COMPLETE',block=block,seeds=SEEDS,logical_epochs=250,
                 job_ids=[j['job_id'] for j in jobs],
                 logical_adam_steps=250*jobs[0]['config']['steps_per_epoch'],duration_seconds=time.time()-start,
                 selected_groups=groups,selected_contrast=xname,selected_fusion_component=selected_component,
                 artifact_index=str(directory/'ARTIFACT_INDEX.json'),tables=list(index))
    write_json(directory/'BLOCK_COMPLETE.json',receipt);return receipt


def _candidate_panel(anchor,vector):
    anchor.layout.copy_to(anchor.model,vector,anchor.packet.get('buffers'))
    return panel_metrics(anchor.panels,_panel_predictions(anchor.model,anchor.panels))


def _conditional(out,jobs,prepared,device,tables,selection,xdescriptor,groups):
    h2=tables['H2_TRIGGER'][0]
    if h2['status']=='DEPENDENCY_INCOMPLETE': raise RuntimeError('H2 trigger inputs incomplete')
    if h2['triggered']:
        scope=max(('S','N'),key=lambda g:selection['deletion'][g]['mean'])
        for anchor in _anchors(out,jobs,prepared,device,tables,('E35','E50')):
            fixed=anchor.packet['F_fp32'];h=min(WINDOWS[anchor.job['dataset']][1],anchor.packet['step'])
            matched=anchor.row['state']['matched_done'][anchor.packet['epoch']]
            for group in ('all',scope):
                mask=anchor.masks[group];vectors={};choices={}
                for menu,variants in (('rho',('fast','slow')),('mechanism',('recent','old'))):
                    candidates={'F':fixed};utilities={'F':0.}
                    for variant in variants:
                        if menu=='rho':
                            vector=fixed.clone();vector[mask]=matched[variant][mask]
                        else: vector,_=build_state(anchor.packet,_descriptor('F',group,variant,'delete',h),anchor.masks)
                        candidates[variant]=vector
                        utilities[variant]=_mac(_candidate_panel(anchor,vector),'P_A')-_mac(anchor.base['F']['metrics'],'P_A')
                    choices[menu]=choose(utilities,('F',)+variants);vectors[menu]=candidates[choices[menu]]
                a,b,info=matched_states(fixed,vectors['rho'],vectors['mechanism'])
                values=[]
                for label,vector in (('rho',a),('mechanism',b)):
                    values.append(anchor.evaluate(f'H2/{group}/{label}',vector=vector,reference='F',descriptor=dict(group=group,matched='H_steps_scope_radius',original_choice=choices[label])))
                change=max(float(np.max(np.abs(v['p'].astype('float64')-anchor.base['F']['p']))) for v in values)
                tables['H2_MATCHED_ASSAY'].append(dict(anchor.identity,scope=group,choices=choices,h=h,start_step=anchor.packet['step']-h,
                    **info,metrics={k:v['metrics'] for k,v in zip(('rho','mechanism'),values)},
                    maximum_probability_change=change,status='INFORMATIVE' if info['informative'] and change>np.finfo(np.float32).eps else 'MATCHED_ASSAY_UNINFORMATIVE',
                    V_B_utility={k:_mac(v['metrics'],'V_B')-_mac(anchor.base['F']['metrics'],'V_B') for k,v in zip(('rho','mechanism'),values)},selection_population='P_A'))
    ds,obj=jobs[0]['dataset'],jobs[0]['objective'];h=WINDOWS[ds][1]
    effect=_mean_effect(tables,(f'C/{h}/recent','O'),population='V')
    source=[r for r in tables['SOURCE_OVERLAP'] if r['anchor'] in ('E35','E50') and r['h']==h and r['population']=='V_mech']
    recovery=seed_summary({s:_mean([r['recovered']-r['lost_recent_benefit'] for r in source if r['seed']==s]) for s in SEEDS})
    objective=seed_summary({s:_mean([r['pb_objective_change'] for r in source if r['seed']==s]) for s in SEEDS})
    complete=all(v['status']=='COMPLETE' for v in (effect,recovery,objective))
    trigger=obj=='clean' and complete and effect['mean']>.0002 and recovery['mean']>0 and objective['mean']<0
    tables['SOURCE_ASSAY_DECISION']=dict(triggered=trigger,status='TRIGGERED' if trigger else ('NOT_TRIGGERED' if complete else 'DEPENDENCY_INCOMPLETE'),
        objective_is_clean=obj=='clean',recent_effect=effect,net_recovery=recovery,training_objective_change=objective)
    if not complete: raise RuntimeError('Source assay dependencies incomplete')
    if trigger:
        group=xdescriptor['group'] if xdescriptor and xdescriptor['group']!='all' else groups[0]
        for anchor in _anchors(out,jobs,prepared,device,tables,('E20','E35')):
            packet=anchor.row['state']['source_packets'][anchor.packet['epoch']];c=anchor.job['config']
            states=source_direction_states(anchor.packet['online_fp32'],packet['pretheta'],packet['data_grad'],packet['mhat'],packet['vhat'],anchor.masks[group],lr=c['lr'],eps=c['eps'],weight_decay=c['weight_decay'])
            anchor.layout.copy_to(anchor.model,packet['pretheta'],anchor.packet.get('buffers'));past=_panel_predictions(anchor.model,anchor.panels)
            for variant,(vector,identity) in states.items():
                anchor.layout.copy_to(anchor.model,vector,anchor.packet.get('buffers'));pred=_panel_predictions(anchor.model,anchor.panels)
                metrics=panel_metrics(anchor.panels,pred)
                tables['SOURCE_ASSAY'].append(dict(anchor.identity,scope=group,variant=variant,metrics=metrics,**identity,real_adam_steps=0,state_hash=vector_hash(vector)))
                for pop in ('P_A','P_B','V_mech'):
                    panel=anchor.panels[pop]
                    for d in range(3):
                        for t in range(2):
                            y,a,_=_view(panel,past[pop],d,t,pop);_,o,_=_view(panel,anchor.base['O']['panels'][pop],d,t,pop);_,b,_=_view(panel,pred[pop],d,t,pop)
                            pairs=sample_pairs(y,int.from_bytes(hashlib.sha256(f'{ds}|{anchor.job["seed"]}|{pop}|{d}|{t}'.encode()).digest()[:8],'little'))
                            if len(pairs): tables['SOURCE_ASSAY_OVERLAP'].append(dict(anchor.identity,scope=group,variant=variant,population=pop,domain=d,task=t,**source_overlap(a,o,b,pairs)))
    margin=[r for r in tables['FUNCTIONAL_MARGIN'] if r['contrast']=='O_F' and r['cut']=='h' and r.get('space')=='logit' and r['anchor'] in ('E35','E50')]
    interaction=seed_summary({s:_mean([r['interaction_rms'] for r in margin if r['seed']==s]) for s in SEEDS})
    rank_interaction=seed_summary({s:_mean([r['net']-r['representation_net']-r['downstream_net'] for r in margin if r['seed']==s]) for s in SEEDS})
    complete=all(v['status']=='COMPLETE' for v in (interaction,rank_interaction))
    # The fixed absolute functional precision also used by the registered path check.
    cross=complete and set(selection['selected'])=={'S','N'} and interaction['mean']>1e-5 and rank_interaction['mean']!=0
    tables['STAR_CROSS_DECISION']=dict(triggered=cross,status='TRIGGERED' if cross else ('NOT_TRIGGERED' if complete else 'DEPENDENCY_INCOMPLETE'),
        expanded_S_and_D=set(selection['selected'])=={'S','N'},margin_interaction=interaction,net_rank_interaction=rank_interaction,functional_resolution=1e-5)
    if not complete: raise RuntimeError('STAR trigger dependencies incomplete')
    if cross:
        for anchor in _anchors(out,jobs,prepared,device,tables,('E35','tF')):
            a,b=anchor.packet['online_fp32'],anchor.packet['F_fp32']
            def operator(x): return fn.star_cross_readout(anchor.model,anchor.layout,x,a,b,mode='delete',background='F')
            value=anchor.evaluate('STAR_CROSS/delete',vector=b,operator=operator,reference='F',descriptor=dict(background='F',component='delta_shared_times_delta_slot'))
            tables['STAR_CROSS_EFFECTS'].append(dict(anchor.identity,metrics=value['metrics'],state_id='STAR_CROSS/delete'))


def engineering_costs(out,dataset,device):
    """One real source step and population-sized causal readouts; no science run."""
    from .trajectory import create_session, train_batch,rounding_observation
    from experiments.overnight_ab_20261103.state import tensor_bytes
    prepared=read_json(Path(out)/'PREPARED.json')
    job=next(j for j in prepared['jobs'] if j['dataset']==dataset and j['objective']=='clean' and j['seed']==2022)
    bundle=cached_bundle(prepared['cache_root'],dataset,include_test=False)
    panels=load_panels(out,dataset,2022,bundle)
    s=create_session(job,bundle,device,panels)
    n=job['config']['batch_size'];x=torch.as_tensor(bundle.train_x[:n].copy(),device=device).T;y=torch.as_tensor(bundle.train_y[:n].copy(),device=device)
    train_batch(s,x,y)
    rounding=rounding_observation(s)
    assert len(rounding)==18 and all('delta_auc' in row and 'probability_max_abs' in row for row in rounding)
    a=s['layout'].flatten(s['model'].parameters()).cpu();b=s['bank'].get('F').detach().cpu().clone()
    packet=dict(online_fp32=a,F_fp32=b,R_fp64=b.double()-a.double(),old_R_fp64_by_effective_h={1:torch.zeros_like(a,dtype=torch.float64)},buffers={},epoch=0,step=1)
    row={'packets':{'ENGINEERING':packet}};tables=defaultdict(list)
    start=time.perf_counter();anchor=Anchor(out,job,row,'ENGINEERING',bundle,device,tables)
    endpoint_seconds=time.perf_counter()-start
    start=time.perf_counter();anchor.evaluate('B/S',_descriptor(group='S'));core_seconds=time.perf_counter()-start
    def cut_operator(x):
        anchor.layout.copy_to(anchor.model,a);values=fn.capture_cut(anchor.model,x,'h')
        anchor.layout.copy_to(anchor.model,b);return fn.mixed_readout(anchor.model,x,'h',values)
    start=time.perf_counter();anchor.evaluate('D/engineering/h',vector=b,operator=cut_operator)
    cut_seconds=time.perf_counter()-start
    start=time.perf_counter();_fusion(anchor,'content',a,b,'O_F');fusion_seconds=time.perf_counter()-start
    panel=panels['V_mech'];rows=[];pairs=[];tasks=[]
    for d in range(3):
        valid=panel['valid_mask'][d]
        for t in range(2):
            sampled=sample_pairs(panel['natural_y'][d,valid,t],2022+10*d+t,8)
            xx=panel['natural_x'][d,valid]
            for pair in sampled:
                i=len(rows);rows.extend(xx[pair]);pairs.append([i,i+1]);tasks.append(t)
    x=torch.as_tensor(np.asarray(rows),device=device).T
    start=time.perf_counter();integral=fn.integrate_path(anchor.model,anchor.layout,x,a,b,{g:anchor.masks[g] for g in ('S','N')},np.asarray(pairs),np.asarray(tasks));jvp_seconds=time.perf_counter()-start
    _function_contrast(anchor,'ENGINEERING',a,b)
    details=tables['FUNCTIONAL_PAIR_DETAILS'];paths=tables['PATH_CLOSURE']
    assert {r['pair_id'] for r in details}=={r['pair_id'] for r in paths}
    assert all(len(r['state_ids'])==4 and all(r['state_ids']) for r in details)
    assert all(abs(r['margin_11']-r['margin_00']-r['representation']-r['downstream']-r['interaction'])<1e-10 for r in details)
    assert all(all(v!='actual_readout' for v in r['states']) for r in tables['FUNCTIONAL_EFFECTS'])
    receipt=dict(status='PASS',role='ENGINEERING_ONLY',dataset=dataset,seed=2022,rounding_audit=rounding,functional_pair_identity_checked=True,
        real_source_steps=1,validation_rows=len(bundle.val_y),endpoint_pair_seconds=endpoint_seconds,
        core_state_and_exact_pair_seconds=core_seconds,cut_mixed_seconds=cut_seconds,
        fusion_delete_seconds=fusion_seconds,jvp_seconds=jvp_seconds,jvp_nodes=integral['nodes'],
        jvp_unresolved_pairs=int(integral['sign_unresolved'].sum()),parameter_payload_bytes=tensor_bytes(s),
        maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        table_rows={k:len(v) for k,v in tables.items()},
        forecast_counts_per_seed=dict(core_states=60,fine_states_max=48,group_time_states_max=64,
                                     endpoint_reloads_and_selection_max=68,mixed_cuts_max=32,
                                     fusion_deletes_max=12,path_integrals_max=8,conditional_states_max=14),
        worst_case_causal_seconds_per_seed=254*core_seconds+32*cut_seconds+12*fusion_seconds+8*jvp_seconds)
    write_json(Path(out)/f'BLOCK_SMOKE_{dataset.upper()}.json',receipt)
    return receipt


def _factorial_function(anchor,name,states):
    """Score/margin interactions are evaluated separately from AUC threshold effects."""
    for population in ('P_A','P_B','V_mech','V_A','V_B'):
        panel=anchor.panels[population if population in ('P_A','P_B') else 'V_mech']
        key=population if population in ('P_A','P_B') else 'V_mech'
        for d in range(3):
            for t in range(2):
                values=[_view(panel,s['panels'][key],d,t,population) for s in states]
                y=values[0][0]
                pairs=sample_pairs(y,int.from_bytes(hashlib.sha256(f"{anchor.job['dataset']}|{anchor.job['seed']}|{population}|{d}|{t}".encode()).digest()[:8],'little'))
                if not len(pairs):continue
                for space,index in (('probability',1),('logit',2)):
                    scores=[v[index].astype('float64') for v in values]
                    interaction=scores[3]-scores[1]-scores[2]+scores[0]
                    margins=[v[pairs[:,0]]-v[pairs[:,1]] for v in scores]
                    terms=[margins[1]-margins[0],margins[2]-margins[0],margins[3]-margins[1]-margins[2]+margins[0]]
                    residual=margins[3]-margins[0]-sum(terms)
                    anchor.tables['FACTORIAL_SCORE_MARGIN'].append(dict(anchor.identity,contrast_id=name,population=population,domain=d,task=t,space=space,
                        pair_count=len(pairs),score_interaction_mean=float(interaction.mean()),score_interaction_rms=float(np.sqrt(np.mean(interaction**2))),
                        margin_first_mean=float(terms[0].mean()),margin_second_mean=float(terms[1].mean()),margin_interaction_mean=float(terms[2].mean()),
                        margin_interaction_rms=float(np.sqrt(np.mean(terms[2]**2))),closure_max=float(np.max(np.abs(residual)))))
