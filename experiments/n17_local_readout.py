"""Primary-orchestrator numerical summaries for the frozen N3 estimands."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import n17_common as c

def collect(directories,out):
    records=[];seen=set();identities={}
    for directory in map(Path,directories):
        receipt=json.loads((directory/'receipt.json').read_text());assert receipt['status']=='COMPLETE'
        conf=json.loads((directory/'config.json').read_text());frame=pd.read_csv(directory/'macro_readouts.csv')
        identity_key=(conf['optimizer'],conf['seed'],conf['epoch'],conf['anchor_arm'])
        identity={key:conf[key] for key in ('checkpoint_sha','source_sha','geometry_source_sha','base_sources','draw_stream','batch_size','lr')}
        if identity_key in identities:assert identities[identity_key]==identity,'incompatible anchor or supplementary draw bank'
        identities[identity_key]=identity
        for split,part in frame.groupby('split'):
            baseline=part[part.case=='unupdated_anchor'].iloc[0];part=part[part.draw>=0]
            for draw,drawpart in part.groupby('draw'):
                pivot=drawpart.set_index('case');sb=pivot.loc['native_SB'];sn=pivot.loc['native_SN']
                for case,row in pivot.iterrows():
                    for metric in ('auc','logloss','train_J'):
                        # J is common to both validation subsets: retain just one copy.
                        if metric=='train_J' and split=='selection':continue
                        common=dict(seed=int(conf['seed']),epoch=int(conf['epoch']),anchor_arm=conf['anchor_arm'],optimizer=conf['optimizer'],draw=int(draw),split='train' if metric=='train_J' else split,metric=metric)
                        block=str(row.block)
                        for label,value in [('vs_SB',row[metric]-sb[metric]),('vs_SN',row[metric]-sn[metric]),('vs_anchor',row[metric]-baseline[metric]),('absolute',row[metric])]:
                            key=tuple(common.values())+(case,label)
                            if key in seen:raise AssertionError('duplicate draw/estimand; prefix is not new data')
                            seen.add(key);records.append(dict(**common,block=block,case=case,estimand=label,value=float(value),direction_undefined=bool(row.direction_undefined)))
                for block in drawpart.block.unique():
                    if block=='ALL' or block=='P':continue
                    names=[block+'_full',block+'_norm',block+'_direction']
                    if not all(name in pivot.index for name in names):continue
                    full,norm,direction=[pivot.loc[name] for name in names]
                    for metric in ('auc','logloss','train_J'):
                        if metric=='train_J' and split=='selection':continue
                        common=dict(seed=int(conf['seed']),epoch=int(conf['epoch']),anchor_arm=conf['anchor_arm'],optimizer=conf['optimizer'],draw=int(draw),split='train' if metric=='train_J' else split,metric=metric,block=block,case=block+'_factorial')
                        for label,value in [('direction_residual',full[metric]-norm[metric]),('norm_direction_interaction',full[metric]-norm[metric]-direction[metric]+sb[metric])]:records.append(dict(**common,estimand=label,value=float(value),direction_undefined=False))
    out.mkdir(parents=True,exist_ok=False);frame=pd.DataFrame(records);frame.to_csv(out/'draw_estimands.csv',index=False)
    keys=['optimizer','epoch','anchor_arm','block','case','estimand','split','metric','seed'];seedrows=[]
    for key,part in frame.groupby(keys):
        assert not part.draw.duplicated().any();n=len(part);seedrows.append(dict(zip(keys,key),draws=n,mean=float(part.value.mean()),mc_se=float(part.value.std(ddof=1)/np.sqrt(n)) if n>1 else None,direction_undefined=int(part.direction_undefined.sum())))
    seeds=pd.DataFrame(seedrows);seeds.to_csv(out/'seed_estimands.csv',index=False);summaries=[]
    for key,part in seeds.groupby(keys[:-1]):
        n=len(part);x=part['mean'].to_numpy();mu=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n)) if n>1 else None
        summaries.append(dict(zip(keys[:-1],key),n=n,seeds=part.seed.to_list(),mean=mu,positive=int(sum(x>0)),negative=int(sum(x<0)),seed_t95_low=float(mu-t.ppf(.975,n-1)*se) if n>1 else None,seed_t95_high=float(mu+t.ppf(.975,n-1)*se) if n>1 else None,mc_se=float(np.sqrt(np.square(part.mc_se).sum())/n),minimum_draws=int(part.draws.min()),maximum_draws=int(part.draws.max()),undefined_draws=int(part.direction_undefined.sum())))
    pd.DataFrame(summaries).to_csv(out/'conditional_summary.csv',index=False);c.dump(out/'receipt.json',dict(status='COMPLETE',source_sha=c.sha(__file__),sources=list(map(str,directories)),training_seed_unit=True,draws_are_not_training_seeds=True,test_read=False))
    return pd.DataFrame(summaries)

def nominate(summary,out):
    blocks=['C_embedding','C_shared_STAR','C_skip','C_star_mlp','C_experts'];candidates=[]
    for epoch in (10,40):
        for index,block in enumerate(blocks):
            entries={};eligible=True
            for anchor in ('SN','SB'):
                for effect in ('full','direction'):
                    cond=(summary.epoch==epoch)&(summary.anchor_arm==anchor)&(summary.block==block)&(summary.split=='diagnostic')&(summary.metric=='auc')
                    cond&=((summary.case==block+'_full')&(summary.estimand=='vs_SB')) if effect=='full' else ((summary.case==block+'_factorial')&(summary.estimand=='direction_residual'))
                    part=summary[cond]
                    if len(part)!=1:eligible=False;continue
                    row=part.iloc[0].to_dict();entries[anchor+'_'+effect]=row
                    if row['n']!=4 or row['undefined_draws']>0 or row['minimum_draws']<32:eligible=False
            if eligible:
                full_mean=np.mean([entries[anchor+'_full']['mean'] for anchor in ('SN','SB')]);direction_mean=np.mean([entries[anchor+'_direction']['mean'] for anchor in ('SN','SB')]);score=abs(full_mean);need64=any(v['mc_se']>.0001/4 for v in entries.values())
                candidates.append(dict(epoch=epoch,block=block,block_order=index,score=score,need64=need64,expected_full_sign=int(np.sign(full_mean)),expected_direction_sign=int(np.sign(direction_mean)),entries=entries))
    candidates.sort(key=lambda x:(-x['score'],x['epoch'],x['block_order']))
    choice=candidates[0] if candidates else None
    c.dump(out/'candidate_frozen.json',dict(status='FROZEN',candidate=choice,all_development_candidates=candidates,delta_local=.0001,test_read=False,selection_rule='N3_DESIGN_ADDENDUM; no reselection after precision supplement'))
    print('N3 candidate',None if choice is None else {k:choice[k] for k in ('block','epoch','need64','expected_full_sign','expected_direction_sign')})

def confirm(summary,candidate_file,out):
    frozen=json.loads(candidate_file.read_text());assert frozen['status']=='FROZEN';candidate=frozen['candidate']
    if candidate is None:
        c.dump(out/'confirmation_gate.json',dict(status='NO_DEFINED_CANDIDATE',N4_trigger=False));return
    block=candidate['block'];epoch=candidate['epoch'];entries={};passed=True;need64=False;precision_failed=False
    for anchor in ('SN','SB'):
        for effect in ('full','direction'):
            cond=(summary.epoch==epoch)&(summary.anchor_arm==anchor)&(summary.block==block)&(summary.split=='diagnostic')&(summary.metric=='auc')
            cond&=((summary.case==block+'_full')&(summary.estimand=='vs_SB')) if effect=='full' else ((summary.case==block+'_factorial')&(summary.estimand=='direction_residual'))
            part=summary[cond];assert len(part)==1;row=part.iloc[0].to_dict();assert row['n']==4 and sorted(row['seeds'])==list(range(2036,2040))
            assert row['minimum_draws']>=32 and row['maximum_draws']<=64
            precision=row['mc_se']<=.0001/4;expected=candidate['expected_'+effect+'_sign'];same_sign=row['positive'] if expected>0 else row['negative'] if expected<0 else 0
            effect_pass=abs(row['mean'])>=.0001 and int(np.sign(row['mean']))==expected and same_sign>=3 and row['undefined_draws']==0
            passed&=bool(effect_pass and precision);need64|=bool(not precision and row['maximum_draws']<=32);precision_failed|=bool(not precision and row['maximum_draws']>=64)
            entries[anchor+'_'+effect]=dict(**row,expected_sign=expected,effect_rule_pass=bool(effect_pass),precision_pass=bool(precision))
    resource=json.loads((c.OUT/'N1_diagnostic_gates.json').read_text());important=resource['optimizers'][resource['N2_optimizer']]['important_residual']
    c.dump(out/'confirmation_gate.json',dict(status='NEED_PRECISION_SUPPLEMENT' if need64 else 'INCONCLUSIVE_PRECISION' if precision_failed else 'FROZEN',
        candidate_sha=c.sha(candidate_file),candidate=candidate,entries=entries,delta_local=.0001,need64=need64,
        independent_local_rule_pass=bool(passed),N1_important_residual=important,N4_trigger=bool(passed and important),test_read=False,
        scope='resource trigger; independent training seeds, conditional Monte Carlo precision; not a method-benefit claim'))
    print('N3 independent local gate',bool(passed),'need64',need64,'N4 trigger',bool(passed and important))

def main():
    p=argparse.ArgumentParser();p.add_argument('--directories',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--nominate',action='store_true');p.add_argument('--confirm',type=Path);a=p.parse_args()
    assert not (a.nominate and a.confirm)
    summary=collect(a.directories,a.out)
    if a.nominate:nominate(summary,a.out)
    if a.confirm:confirm(summary,a.confirm,a.out)

if __name__=='__main__':main()
