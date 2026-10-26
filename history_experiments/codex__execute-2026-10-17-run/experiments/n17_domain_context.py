"""Descriptive size/performance context and opposite trade-off patterns; no regression."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import n17_common as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparisons',type=Path,required=True);p.add_argument('--performance',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.performance/'receipt.json').read_text());assert receipt['status']=='COMPLETE' and receipt['comparison_sha']==c.sha(a.comparisons)
    pairs=json.loads(a.comparisons.read_text())['comparisons'];lookup={(x['group'],int(x['seed']),int(x.get('budget',100))):x for x in pairs};raw=pd.read_csv(a.performance/'paired_seed_endpoints.csv')
    context=[];patterns=[]
    for (group,seed,budget,readout,estimand),part in raw.groupby(['group','seed','budget','readout','estimand']):
        pair=lookup[(group,int(seed),int(budget))];conf=json.loads((Path(pair['SN'])/'config.json').read_text());other=json.loads((Path(pair['SB'])/'config.json').read_text())
        counts={int(d):int(n) for d,n in conf['data']['train_domain_counts'].items()};assert conf['data']['split_row_id_hashes']==other['data']['split_row_id_hashes']
        large=max(counts,key=counts.get);smaller=[d for d in counts if d!=large];assert len(smaller)==2 and counts[large]>max(counts[d] for d in smaller)
        values=part.set_index('metric');domain_differences={d:float(values.loc[f'd{d}_auc','difference']) for d in counts}
        for d,n in counts.items():
            for task in (0,1):
                for metric in ('auc','logloss'):
                    value=values.loc[f'd{d}t{task}_{metric}'];population=conf['data']['cell_stats']['test'][f'd{d}t{task}']
                    context.append(dict(group=group,seed=int(seed),budget=int(budget),readout=readout,estimand=estimand,dataset=conf['data']['dataset_key'],domain=d,task=task,metric=metric,
                        train_domain_n=n,test_cell_n=int(population['count']),test_positive_rate=float(population['positive_rate']),largest_train_domain=d==large,
                        SN_reference=float(value.SN),SB_candidate=float(value.SB),difference=float(value.difference),reference_q_arm=conf['arm'],candidate_q_arm=other['arm']))
        patterns.append(dict(group=group,seed=int(seed),budget=int(budget),readout=readout,estimand=estimand,largest_domain=large,
            largest_delta=domain_differences[large],smaller_0=smaller[0],smaller_0_delta=domain_differences[smaller[0]],smaller_1=smaller[1],smaller_1_delta=domain_differences[smaller[1]],
            both_smaller_gain_larger_loss=all(domain_differences[d]>0 for d in smaller) and domain_differences[large]<0,
            either_smaller_gain_larger_loss=any(domain_differences[d]>0 for d in smaller) and domain_differences[large]<0,
            both_smaller_loss_larger_gain=all(domain_differences[d]<0 for d in smaller) and domain_differences[large]>0,
            practical_both_smaller_gain_larger_loss=all(domain_differences[d]>=.003 for d in smaller) and domain_differences[large]<=-.003,
            practical_both_smaller_loss_larger_gain=all(domain_differences[d]<=-.003 for d in smaller) and domain_differences[large]>=.003))
    a.out.mkdir(parents=True,exist_ok=True);pd.DataFrame(context).to_csv(a.out/'domain_task_baseline_population_context.csv',index=False);frame=pd.DataFrame(patterns);frame.to_csv(a.out/'seed_domain_tradeoff_patterns.csv',index=False)
    flags=['both_smaller_gain_larger_loss','either_smaller_gain_larger_loss','both_smaller_loss_larger_gain','practical_both_smaller_gain_larger_loss','practical_both_smaller_loss_larger_gain'];rows=[]
    for key,part in frame.groupby(['group','budget','readout','estimand']):
        rows.append(dict(zip(['group','budget','readout','estimand'],[int(x) if isinstance(x,np.integer) else x for x in key]),n_seeds=int(part.seed.nunique()),
            **{flag+'_seeds':int(part[flag].sum()) for flag in flags}))
    c.dump(a.out/'domain_tradeoff_pattern_counts.json',dict(status='COMPLETE',results=rows,source_sha=c.sha(__file__),comparison_sha=c.sha(a.comparisons),
        scope='descriptive signs of two-task within-domain mean AUC differences, not independent domain replicates; strict both-smaller and flexible either-smaller distinguished; practical versions use .003; no causal size/difficulty attribution or new multiplicity claim; see paired-seed intervals in performance tables'))


if __name__=='__main__':main()
