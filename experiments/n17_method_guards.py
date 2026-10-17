"""Numerical benefit/protection checks; these alone do not establish a new method."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import n17_common as c
from n17_n1_control import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoints',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    frame=pd.read_csv(a.endpoints);bounds=[];decisions=[]
    for (group,readout,budget,estimand),part in frame.groupby(['group','readout','budget','estimand']):
        budget=int(budget)
        lookup=part.set_index(['seed','metric']);seeds=sorted(part.seed.unique());n=len(seeds);assert n>=2
        main=part[part.metric=='macro_auc'].sort_values('seed');assert main.seed.tolist()==seeds
        stat=paired(main.difference);critical=float(t.ppf(1-.05/7,n-1));protected=[]
        for domain in range(3):
            for task in range(2):
                metric=f'd{domain}t{task}_auc';x=np.array([lookup.loc[(seed,metric),'difference'] for seed in seeds],dtype=float)
                mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n));lower=mean-critical*se;good=lower>=-.003;protected.append(good)
                bounds.append(dict(group=group,readout=readout,budget=budget,metric=metric,n=n,mean=mean,one_sided_bound=lower,
                    bound_direction='lower',threshold=-.003,passes=bool(good),alpha=.05/7,seed_values=x.tolist()))
        loss=part[part.metric=='train_p_cell_logloss'].sort_values('seed');assert loss.seed.tolist()==seeds and (loss.SN>0).all()
        relative=((loss.SB-loss.SN)/loss.SN).to_numpy();mean=float(relative.mean());se=float(relative.std(ddof=1)/np.sqrt(n));upper=mean+critical*se;good=upper<=.01;protected.append(good)
        bounds.append(dict(group=group,readout=readout,budget=budget,metric='train_p_cell_logloss_relative',n=n,mean=mean,
            one_sided_bound=upper,bound_direction='upper',threshold=.01,passes=bool(good),alpha=.05/7,seed_values=relative.tolist()))
        benefit=stat['mean']>=.003 and stat['ci95'][0]>0
        decisions.append(dict(group=group,readout=readout,budget=budget,estimand=estimand,candidate='SB-labelled member',baseline='SN-labelled member',
            macro_auc=stat,practical_and_positive_CI=bool(benefit),seven_simultaneous_protections_pass=bool(all(protected)),
            numerical_benefit_and_protections_pass=bool(benefit and all(protected) and n>=8),independent_seed_count=n,
            new_method_claim_authorized=False,scope='orientation fixed in comparison manifest; report all readouts, do not select a favorable one; novelty, frozen primary choice and cost require separate scientific assessment'))
    a.out.mkdir(parents=True,exist_ok=True);pd.DataFrame(bounds).to_csv(a.out/'seven_Bonferroni_protection_bounds.csv',index=False)
    c.dump(a.out/'numerical_benefit_guard_summary.json',dict(status='COMPLETE',results=decisions,source_sha=c.sha(__file__),endpoint_sha=c.sha(a.endpoints),
        relative_loss_definition='paired seed ratio (LL_candidate-LL_baseline)/LL_baseline, then mean/t bound',confidence_scope='seven one-sided Bonferroni t bounds; training seed is unit; finite small-n approximation'))


if __name__=='__main__':main()
