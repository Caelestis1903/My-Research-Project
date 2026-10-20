"""All preregistered transfer candidates against declared, fixed references."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t

OUT=Path(__file__).resolve().parent/'transfer'

def stats(v):
    v=np.asarray(v,dtype=float);n=len(v);sd=v.std(ddof=1);mean=v.mean();half=t.ppf(.975,n-1)*sd/np.sqrt(n)
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,positive=int((v>0).sum()),negative=int((v<0).sum()))

def analyze():
    f=pd.read_csv(OUT/'run_task_macro_metrics.csv');f=f[f.intervention=='none'].copy()
    pairs=[];summaries=[];missing=[];definitions=[]
    recipe=['study','population','scope','dataset','optimizer','method','suite','variant','decay_mode','schedule','fraction','backbone','noise_interaction','stage_group','budget','lr']
    # Preserve historical identity even where pandas would interpret a string as missing.
    f[recipe]=f[recipe].fillna('UNSPECIFIED')
    def same_tasks(reference,candidate):
        taskset=set(candidate.task)
        if len(taskset)==2 and -1 in taskset:
            single=next(task for task in taskset if task!=-1)
            reference=reference[reference.task==single].copy()
            single_macro=reference.copy();single_macro['task']=-1
            return pd.concat([reference,single_macro],ignore_index=True)
        return reference
    def pair(candidate,reference,name,meaning):
        if candidate.empty or reference.empty:
            missing.append(dict(comparison=name,reason='candidate/reference absent'));return
        a=candidate.set_index(['seed','checkpoint','task']);b=reference.set_index(['seed','checkpoint','task'])
        if a.index.duplicated().any() or b.index.duplicated().any():raise ValueError('ambiguous recipe '+name)
        if set(a.index)!=set(b.index) or set(a.index.get_level_values('seed'))!={2022,2023,2024,2025}:
            missing.append(dict(comparison=name,reason='incomplete paired four-seed cell set'));return
        definitions.append(dict(comparison=name,meaning=meaning,
            candidate_run_ids=sorted(candidate.run_id.unique()),reference_run_ids=sorted(reference.run_id.unique())))
        for ix in a.index:
            row=dict(comparison=name,meaning=meaning,seed=ix[0],checkpoint=ix[1],task=ix[2],
                candidate_run_id=a.loc[ix,'run_id'],reference_run_id=b.loc[ix,'run_id'])
            row.update({m:float(a.loc[ix,m]-b.loc[ix,m]) for m in ('macro_auc','macro_logloss','support_p_cell_auc','exposure_p_cell_auc','eval_p_cell_auc')})
            row.update(reference_auc=float(b.loc[ix,'macro_auc']),candidate_auc=float(a.loc[ix,'macro_auc']),
                reference_logloss=float(b.loc[ix,'macro_logloss']),candidate_logloss=float(a.loc[ix,'macro_logloss']))
            pairs.append(row)
        g=pd.DataFrame(pairs)[lambda z:z.comparison==name]
        for (checkpoint,task),gg in g.groupby(['checkpoint','task']):
            for metric in ('macro_auc','macro_logloss','support_p_cell_auc','exposure_p_cell_auc','eval_p_cell_auc'):
                summaries.append(dict(comparison=name,meaning=meaning,checkpoint=checkpoint,task=task,metric=metric,reference_auc=gg.reference_auc.mean(),candidate_auc=gg.candidate_auc.mean(),reference_logloss=gg.reference_logloss.mean(),candidate_logloss=gg.candidate_logloss.mean(),**stats(gg[metric])))
    for key,candidate in f.groupby(recipe,sort=False):
        c=dict(zip(recipe,key));base=f.copy()
        for k in ('study','population','scope','dataset','optimizer','suite','noise_interaction','lr'):
            base=base[base[k]==c[k]]
        prefix='|'.join(str(c[k]) for k in recipe)
        if c['study']=='N19':
            references=['fixed'] if c['suite']=='natural' else ['fixed'] if c['suite']=='stability' else ['audit_fixed'] if c['suite']=='noise' else ['mask_independent','fixed']
            if c['suite']=='natural' and c['method']=='rank_protected':references+=['rank_uniform']
            if c['suite']=='natural' and c['method']=='cell_codebook':references+=['domain_codebook']
            if c['suite']=='natural' and c['method'] in ('domain_mix','axis_mix','full_mix','interaction','reliability_interaction'):
                references+=['full_mix','interaction']
            if c['suite']=='natural' and c['method'] in ('utility_router','regret_router','coalition_router','entropy_router'):
                references+=['entropy_router','full_mix']
            if c['suite']=='noise' and c['noise_interaction']==0:references+=['naive','clean_oracle']
            if c['suite']=='stability' and c['method']=='task_dro':references+=['task_mask']
            if c['suite']=='noise' and c['method']=='audit_joint' and c['noise_interaction']==0:references+=['nested_axis']
            if c['suite']=='noise' and c['method']=='joint_axis':references+=['ind_channel','nested_axis']
            if c['suite']=='noise' and c['method']=='nested_axis':references+=['nested_ind']
            if c['suite']=='noise' and c['method']=='profile_joint':references+=['profile_axis','audit_joint']
            if c['suite']=='sampling' and c['method']=='mask_residual_cv':references+=['mask_antithetic_cv','mask_independent_cv']
            if c['suite']=='sampling' and c['method'] in ('mask_copula','mask_copula_cv'):references+=['mask_antithetic' if c['method']=='mask_copula' else 'mask_antithetic_cv']
            for ref in dict.fromkeys(references):
                if ref==c['method']:continue
                # Sampling all-label fixed reference is the native population suite.
                refbase=base
                if c['suite']=='sampling' and ref=='fixed':
                    refbase=f[(f.study==c['study'])&(f.population==c['population'])&(f.scope==c['scope'])&
                        (f.dataset==c['dataset'])&(f.optimizer==c['optimizer'])&(f.suite=='natural')]
                elif c['suite']=='stability':refbase=base[base.backbone==c['backbone']]
                reference=refbase[refbase.method==ref]
                if c['suite']=='natural' and ref=='fixed':reference=reference[reference.backbone=='m3oe']
                pair(candidate,reference,prefix+' minus '+ref,
                    'complete historical-method comparison; loss/architecture/information differences retained in definition')
        else:
            # Each ablation is compared to the same model/task/optimizer native G0.
            native=base[(base.stage_group=='G0')&(base.method==c['method'])&(base.variant=='native')&
                (base.decay_mode=='standard')&(base.fraction==1.)]
            if c['stage_group'] not in ('G0','G0M','G0MC'):
                pair(candidate,native,prefix+' minus same-model native',
                    'same model/task native reference; declared decay/structure/support/budget intervention')
            if c['stage_group']=='G0':
                fixed=base[(base.stage_group=='G0')&(base.method=='fixed')&(base.variant=='native')&(base.decay_mode=='standard')]
                fixed=same_tasks(fixed,candidate)
                if c['method']!='fixed':pair(candidate,fixed,prefix+' minus fixed same task',
                    'competitive architecture reference, target task(s) matched; parameter counts may differ')
            if c['stage_group']=='G0M':
                private=base[(base.stage_group=='G0MC')&(base.method==c['method'].replace('_joint','_private_joint'))]
                pair(candidate,private,prefix+' minus two-private joint',
                    'joint versus two-private backbone under common mean-task loss; capacity is recorded, not assumed equal')
            if c['stage_group'] in ('G1','G2','G3'):
                for control in ('fixed','ple'):
                    reference=base[(base.stage_group=='G0')&(base.method==control)&(base.variant=='native')&
                        (base.decay_mode=='standard')&(base.fraction==1.)]
                    reference=same_tasks(reference,candidate)
                    pair(candidate,reference,prefix+' minus '+control+' same task',
                        'competitive remedy reference; same target task, no equal-capacity or equal-architecture causal claim')
            if c['stage_group']=='G1' and c['decay_mode'] in ('both_wd0','local_adamw'):
                wd0=base[(base.stage_group=='G1')&(base.method==c['method'])&(base.variant==c['variant'])&
                    (base.decay_mode=='wd0')&(base.fraction==1.)]
                pair(candidate,wd0,prefix+' minus global WD0',
                    'local modulation-group recipe versus global zero decay, same model/task and optimizer family')
                aw=f
                for field in ('study','population','scope','dataset','suite','noise_interaction','lr'):
                    aw=aw[aw[field]==c[field]]
                aw=aw[(aw.stage_group=='G0')&(aw.method==c['method'])&(aw.variant=='native')&
                    (aw.decay_mode=='standard')&(aw.optimizer=='adamw')&(aw.fraction==1.)]
                pair(candidate,aw,prefix+' minus global AdamW',
                    'local recipe versus declared global nominal AdamW; effective float32 shrinkage evaluated separately')
            if c['stage_group']=='G4' and c['schedule']=='matched_updates':
                normal=base[(base.stage_group=='G4')&(base.method==c['method'])&(base.variant==c['variant'])&
                    (base.decay_mode==c['decay_mode'])&(base.fraction==c['fraction'])&(base.schedule=='normal_epochs')]
                pair(candidate,normal,prefix+' minus same-support normal epochs',
                    'same nested support and model, matched-update replay versus ordinary epochs; no pure unique-N effect')
        # Both optimizers, where actually run, retained as a separate comparison.
        if c['optimizer']=='adamw':
            ref=f
            for k in recipe:
                if k!='optimizer':ref=ref[ref[k]==c[k]]
            pair(candidate,ref[ref.optimizer=='adam'],prefix+' minus Adam',
                'nominal AdamW versus Adam at original dtype/kernel/WD; no claim of nonzero effective shrinkage')
    # Population/budget controls preserve the exact old recipe run identifier.
    for study in ('N19','CM19'):
        studyrows=f[f.study==study]
        scopes=['natural_full','natural_equal_N','natural_matched_exposure']
        artificial='artificial_fixed2022' if study=='CM19' else 'historical'
        for candidate_scope,reference_scope in [(x,artificial) for x in scopes]+[
                ('natural_full','natural_matched_exposure'),('natural_equal_N','natural_matched_exposure')]:
            candidates=studyrows[studyrows.scope==candidate_scope]
            for old_id in candidates.historical_run_id.unique():
                # Old IDs include seed. Join recipe via the ID with only its seed removed.
                import re
                signature=re.sub(r'(__seed|__s)\d+',r'\1SEED',old_id)
                names=studyrows.historical_run_id.map(lambda s:re.sub(r'(__seed|__s)\d+',r'\1SEED',s))
                aa=candidates[candidates.historical_run_id.isin(studyrows.loc[names==signature,'historical_run_id'])]
                bb=studyrows[(studyrows.scope==reference_scope)&(names==signature)&(studyrows.dataset=='movielens')]
                name=study+'|'+candidate_scope+' minus '+reference_scope+'|'+signature
                if any(d['comparison']==name for d in definitions) or any(d['comparison']==name for d in missing):continue
                pair(aa,bb,name,'population/support/exposure comparison; multiple population attributes may change, not pure ratio causation')
    pd.DataFrame(pairs).to_csv(OUT/'paired_seed_deltas.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'paired_summary.csv',index=False)
    (OUT/'comparison_definitions.json').write_text(json.dumps(definitions,indent=2)+'\n')
    (OUT/'missing_or_incomplete_comparisons.json').write_text(json.dumps(missing,indent=2)+'\n')
    print('Comparisons',len(definitions),'incomplete/absent',len(missing))

if __name__=='__main__':analyze()
