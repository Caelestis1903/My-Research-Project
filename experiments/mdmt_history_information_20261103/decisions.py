"""Frozen validation-only stage decisions and their exact requested identities."""
import json
from pathlib import Path
from statistics import mean
from .protocol import config, job, load_protocol, digest

PARAMS = ('lambda_mix','lambda_scale','tau','kappa','lr','weight_decay_multiplier')
SIMPLE = ['SCALE','A1','U_MIX','M','H_NORM','T_NORM','C_MIX','H_REAL','T0','H_U','H_MIX','T_U','TM']
HARM = [f'd{d}t{t}_auc' for d in range(3) for t in range(2)]+['sample_weighted_cell_auc','pooled_task_mean_auc']
STAGES = ['D0','D1','REPLICATE','D2','CONFIRM']


def recipe(c):
    out = {k:c[k] for k in PARAMS if k in c}
    base = load_protocol()['datasets'][c['dataset']]['weight_decay']
    out['weight_decay_multiplier'] = c['weight_decay']/base
    return dict(arm=c['arm'], **out)


def make_job(d, r, seed, stage):
    return job(d, r['arm'], seed, stage, **{k:v for k,v in r.items() if k!='arm'})


def blocked_recipe(dataset, r, blocked, seed=2022):
    c=config(r['arm'],dataset,**{k:v for k,v in r.items() if k!='arm'})
    return ((dataset,r['arm']) in blocked and c==config(r['arm'],dataset)) or (dataset,digest(c),seed) in blocked


def has_results(results,d,r,seeds):
    c=config(r['arm'],d,**{k:v for k,v in r.items() if k!='arm'})
    return all(any(x['job']['dataset']==d and x['job']['seed']==s and x['job']['config']==c for x in results) for s in seeds)


def sig(c):
    return digest(c)


def lookup(results, d, r, seed):
    c = config(r['arm'], d, **{k:v for k,v in r.items() if k!='arm'})
    matches = [x for x in results if x['job']['dataset']==d and x['job']['seed']==seed and x['job']['config']==c]
    assert len(matches)==1, ('Missing or duplicate completed identity',d,r,seed,len(matches))
    return matches[0]['selected_validation']['fixed_ema']


def value(results,d,r,seeds=(2022,)):
    return mean(lookup(results,d,r,s)['macro_auc'] for s in seeds)


def difference(results,d,a,b,seeds):
    return [lookup(results,d,a,s)['macro_auc']-lookup(results,d,b,s)['macro_auc'] for s in seeds]


def protection(results,d,a,baselines,seeds):
    harms=[]; ll=[]
    for b in baselines:
        for metric in HARM:
            ds=[lookup(results,d,a,s)[metric]-lookup(results,d,b,s)[metric] for s in seeds]
            if mean(ds)<=-.002 and all(x<0 for x in ds): harms.append(dict(baseline=b,metric=metric,deltas=ds))
        ds=[lookup(results,d,a,s)['macro_logloss']-lookup(results,d,b,s)['macro_logloss'] for s in seeds]
        if mean(ds)>=.002 and all(x>0 for x in ds):ll.append(dict(baseline=b,deltas=ds))
    return dict(AUC_TRADEOFF=harms,AUC_WITH_LL_TRADEOFF=ll)


def unique(rs, dataset=None):
    seen=set(); out=[]
    for r in rs:
        k=digest(config(r['arm'],dataset,**{n:v for n,v in r.items() if n!='arm'})) if dataset else digest(r)
        if k not in seen:seen.add(k);out.append(r)
    return out


def best(results,d,rs,seeds=(2022,),tol=1e-6,order=None):
    top=max(value(results,d,r,seeds) for r in rs)
    tied=[r for r in rs if top-value(results,d,r,seeds)<=tol]
    if order:tied.sort(key=lambda r:order.index(r['arm']))
    return tied[0]


def grid(d,arm):
    p=load_protocol();name='A0' if arm=='A0f' else arm
    return [dict(arm=arm,**r) for r in p['grids'][name]]


def read_decision(attempt,stage):
    return json.loads((Path(attempt)/'decisions'/f'{stage}.json').read_text())


def stage_jobs(attempt,stage):
    p=load_protocol(); jobs=[]
    if stage=='D0':
        for d in p['datasets']:
            jobs += [job(d,a,2022,stage) for a in p['stages']['D0']['arms']]
        return jobs
    prior=read_decision(attempt,STAGES[STAGES.index(stage)-1])
    for d,v in prior['datasets'].items():
        if v.get('next_stage')!=stage:continue
        if stage=='D1':
            for group in v['groups']:jobs += [make_job(d,r,2022,stage) for r in group['configs']]
        elif stage=='REPLICATE':jobs += [make_job(d,r,2024,stage) for r in v['winners']]
        elif stage=='D2':jobs += [make_job(d,r,s,stage) for r in v['controls'] for s in (2022,2024)]
        else:jobs += [make_job(d,r,s,stage) for r in v['confirmation_arms'] for s in (2023,2025,2026)]
    return list({j['key']:j for j in jobs}.values())


def matched_controls(family,main,simple):
    tau=main.get('tau',.1);k=main.get('kappa',.25)
    if family=='H':return [dict(arm='H_REAL',kappa=k),dict(arm='C_MIX',tau=tau,kappa=k),dict(arm='H_U',tau=tau,kappa=k),dict(arm='H_NORM',tau=tau,kappa=k),dict(arm='M',lambda_mix=.1,tau=tau),dict(arm='SCALE',lambda_scale=.1)]
    if family=='T':return [dict(arm='M',lambda_mix=.1,tau=tau),dict(arm='T0',kappa=k),dict(arm='A0f'),dict(arm='T_U',lambda_mix=.1,tau=tau,kappa=k),dict(arm='T_NORM',lambda_mix=.1,tau=tau,kappa=k),dict(arm='SCALE',lambda_scale=.1)]
    return [dict(arm='U_MIX',lambda_mix=main['lambda_mix'],tau=tau),dict(arm='M',lambda_mix=simple['lambda_mix'],tau=simple['tau']),dict(arm='SCALE',lambda_scale=main['lambda_mix']),dict(arm='SCALE',lambda_scale=simple['lambda_mix']),dict(arm='A0f'),dict(arm='A1')]


def decide(attempt,stage,results,blocked=()):
    """Only selected_validation is accessed here; test metrics never enter selection."""
    p=load_protocol(); out=dict(stage=stage,source='validation_fixed_ema_selected',datasets={})
    previous=read_decision(attempt,STAGES[STAGES.index(stage)-1]) if stage!='D0' else None
    for d in p['datasets']:
        if previous and previous['datasets'][d].get('next_stage')!=stage:
            out['datasets'][d]=dict(previous['datasets'][d],stage_status='NOT_TRIGGERED');continue
        a0=dict(arm='A0f')
        if stage=='D0':
            available=[dict(arm=a) for a in p['stages']['D0']['arms'] if not blocked_recipe(d,dict(arm=a),blocked)]
            prot={r['arm']:protection(results,d,r,[a0],(2022,)) for r in available}
            signal={}
            for r in available:
                a=r['arm'];gain=value(results,d,r)-value(results,d,a0); near=False
                if a in ('H_MIX','TM'):
                    neighbors=['H_REAL','C_MIX'] if a=='H_MIX' else ['M','T0']
                    ns=[x for x in available if x['arm'] in neighbors]
                    near=len(ns)==2 and gain>=-.0005 and value(results,d,r)-max(value(results,d,n) for n in ns)>=.0003
                signal[a]=(gain>0 or near) and not prot[a]['AUC_TRADEOFF']
            families={f:[r for r in available if r['arm'] in members and signal[r['arm']]] for f,members in [('M',['M']),('H',['H_MIX','H_REAL','C_MIX']),('T',['TM','T0'])]}
            unavailable_families=[]
            for family,neighbors in [('H',['H_REAL','C_MIX']),('T',['M','T0'])]:
                if families[family] and not any(has_results(results,d,dict(arm=a),(2022,)) for a in neighbors):
                    families[family]=[];unavailable_families.append(family)
            scores={f:max(value(results,d,r) for r in rs) for f,rs in families.items() if rs}
            v=dict(status='IMPLEMENTATION_BLOCKED' if unavailable_families else 'NO_GAIN',unavailable_families=unavailable_families,next_stage=None,protections=prot,signals=signal,family_scores=scores,cost_blocked_default_arms=[list(x) for x in sorted(blocked) if x[0]==d],chosen=a0)
            if scores:
                top=max(scores.values());family=next(f for f in ['M','H','T'] if f in scores and top-scores[f]<=.0002)
                if family=='H':simple=best(results,d,[r for r in [dict(arm='H_REAL'),dict(arm='C_MIX')] if has_results(results,d,r,(2022,))],tol=.0002)['arm']; focal='H_MIX'
                elif family=='T':simple=best(results,d,[r for r in [dict(arm='M'),dict(arm='T0')] if has_results(results,d,r,(2022,))],tol=.0002)['arm'];focal='TM'
                else:simple='U_MIX';focal='M'
                v.update(status='COMPLETED',next_stage='D1',family=family,deferred_families=[f for f in scores if f!=family],groups=[dict(role='main',configs=grid(d,focal)),dict(role='simple',configs=grid(d,simple)),dict(role='baseline',configs=grid(d,'A0f'))])
        elif stage=='D1':
            old=previous['datasets'][d];winners=[];details=[]
            for group in old['groups']:
                measured=[r for r in group['configs'] if not blocked_recipe(d,r,blocked)]
                ps=[protection(results,d,r,[a0],(2022,)) for r in measured]
                allowed=[r for r,h in zip(measured,ps) if not h['AUC_TRADEOFF']]
                winner=best(results,d,allowed or measured) if measured else None
                winners.append(winner);details.append(dict(group=group,protections=ps,winner=winner,eligible=bool(allowed),cost_blocked=[r for r in group['configs'] if blocked_recipe(d,r,blocked)]))
            v=dict(status='COMPLETED' if all(winners) else 'IMPLEMENTATION_BLOCKED',next_stage='REPLICATE' if all(winners) else None,family=old['family'],winners=winners,groups=details)
        elif stage=='REPLICATE':
            old=previous['datasets'][d];main,simple,a0b=old['winners'];baselines=[a0,a0b];passes=[]
            if not has_results(results,d,a0b,(2022,2024)):
                out['datasets'][d]=dict(old,status='IMPLEMENTATION_BLOCKED',next_stage=None,reason='Chosen baseline missing a terminally blocked replicate');continue
            for r in (main,simple):
                if not has_results(results,d,r,(2022,2024)):
                    passes.append(False);continue
                ds=[difference(results,d,r,b,(2022,2024)) for b in baselines]
                protected=protection(results,d,r,baselines,(2022,2024))
                passes.append(all(mean(x)>=.0005 and min(x)>=-.0002 for x in ds) and not protected['AUC_TRADEOFF'])
            v=dict(status='COMPLETED' if any(passes) else ('NO_GAIN' if all(has_results(results,d,r,(2022,2024)) for r in (main,simple)) else 'IMPLEMENTATION_BLOCKED'),next_stage='D2' if any(passes) else None,family=old['family'],winners=old['winners'],development_pass=passes,chosen=a0)
            if any(passes):v['controls']=matched_controls(old['family'],main,simple)
        elif stage=='D2':
            old=previous['datasets'][d];main,simple,a0b=old['winners'];baselines=[a0,a0b];pool=[r for r in unique([main,simple]+old['controls'],d) if has_results(results,d,r,(2022,2024))];eligible=[];details=[]
            for r in pool:
                diffs=[difference(results,d,r,b,(2022,2024)) for b in baselines]
                protected=protection(results,d,r,baselines,(2022,2024))
                basic=all(mean(x)>=.0005 and min(x)>0 for x in diffs) and not protected['AUC_TRADEOFF']
                joint=True;comparisons=[]
                if r['arm'] in ('H_MIX','TM'):
                    names=['H_REAL','C_MIX','H_U','H_NORM'] if r['arm']=='H_MIX' else ['M','T0','T_U','T_NORM']
                    for name in names:
                        b=next(c for c in old['controls'] if c['arm']==name)
                        if not has_results(results,d,b,(2022,2024)):
                            comparisons.append(dict(control=b,status='COST_BLOCKED' if (d,b['arm']) in blocked else 'IMPLEMENTATION_BLOCKED'));joint=False;continue
                        ds=difference(results,d,r,b,(2022,2024));comparisons.append(dict(control=b,deltas=ds));joint &= mean(ds)>=.0003 and min(ds)>0
                if basic and joint:eligible.append(r)
                details.append(dict(recipe=r,basic=basic,joint=joint,baseline_deltas=diffs,protections=protected,components=comparisons))
            chosen=best(results,d,eligible,(2022,2024),.0003,SIMPLE) if eligible else a0
            v=dict(status='COMPLETED' if eligible else 'NO_GAIN',next_stage='CONFIRM' if eligible else None,family=old['family'],winners=old['winners'],controls=old['controls'],chosen=chosen,candidates=details)
            if eligible:
                # Conditional M has no privileged status over an equally useful U_MIX.
                others=[r for r in pool if config(r['arm'],d,**{k:v for k,v in r.items() if k!='arm'})!=config(chosen['arm'],d,**{k:v for k,v in chosen.items() if k!='arm'}) and r['arm']!='A0f' and not protection(results,d,r,baselines,(2022,2024))['AUC_TRADEOFF']]
                comparator=best(results,d,others,(2022,2024),.0003,SIMPLE) if others else None
                v['comparator']=comparator;v['confirmation_arms']=unique([a0,a0b,chosen]+([comparator] if comparator else []),d)
        else:
            old=previous['datasets'][d]
            for r in old['confirmation_arms']:
                for s in (2022,2023,2024,2025,2026):
                    if not has_results(results,d,r,(s,)) and not blocked_recipe(d,r,blocked,s):raise AssertionError('Confirmation incomplete')
            complete=all(has_results(results,d,r,(2022,2023,2024,2025,2026)) for r in old['confirmation_arms'])
            v=dict(old,status='COMPLETED' if complete else 'IMPLEMENTATION_BLOCKED',next_stage=None,confirmation_complete=complete)
        out['datasets'][d]=v
    return out
