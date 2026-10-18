"""Generate numeric research tables without making scientific judgments."""
import argparse
import pandas as pd
from n19_data import OUT

def table(frame,columns):
    lines=['|'+ '|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        values=[]
        for c in columns:
            x=row[c]
            values.append('' if pd.isna(x) else (f'{x:.6f}' if isinstance(x,float) and c not in ['n_seeds','seed','epochs','parameters'] else str(x)))
        lines.append('|'+ '|'.join(values)+'|')
    return '\n'.join(lines)

def main(phase):
    source=OUT/(phase+'_readout');means=pd.read_csv(source/'method_means.csv');pairs=pd.read_csv(source/'paired_comparisons.csv');cost=pd.read_csv(source/'resources.csv')
    lines=[f'# {phase} numeric tables','',
        'Generated from complete-run raw evidence. These tables make no novelty or effectiveness verdict. All deltas are candidate minus control; higher AUC and lower LogLoss favor the candidate. Four-seed t intervals are descriptive; populations and cells are not independent replications.','']
    for (suite,dataset),g in means.groupby(['suite','dataset']):
        lines.extend([f'## {suite} / {dataset}','',table(g[g.stage=='selected'],['method','n_seeds','macro_auc','macro_logloss','train_weighted_auc','train_weighted_logloss','evaluation_weighted_auc','evaluation_weighted_logloss','worst_cell_auc','worst_cell_logloss']),'',
            'Final checkpoint:','',table(g[g.stage=='final'],['method','n_seeds','macro_auc','macro_logloss','train_weighted_auc','train_weighted_logloss']),''])
        p=pairs[(pairs.suite==suite)&(pairs.dataset==dataset)&(pairs.stage=='selected')&(pairs.metric.isin(['macro_auc','macro_logloss']))]
        lines.extend(['Selected paired comparisons:','',table(p,['method','control','metric','n_seeds','mean','ci95_low','ci95_high','positive_seeds','negative_seeds']),''])
    lines.extend(['## Operational training resources','',
        'Wall time includes training, validation and checkpointing after run initialization, under up to five concurrent jobs per GPU. It is not isolated latency. Reserved memory is per-process PyTorch peak.',''])
    keys=['suite','dataset','method'];c=cost.groupby(keys).agg(n_seeds=('seed','nunique'),epochs=('epochs','mean'),parameters=('parameters','mean'),supernet_parameters=('model_parameters','mean'),wall_seconds=('wall_seconds','mean'),peak_reserved_mib=('peak_reserved_mib','mean'),p6_labels_per_visit=('p6_label_observations_per_row_visit','mean')).reset_index()
    lines.append('P6 labels/visit is instrumented sampling usage; the natural fixed reference uses both labels. Blank cells are outside this cost comparison, not zero-label training.\n')
    lines.append(table(c,keys+['n_seeds','epochs','parameters','supernet_parameters','wall_seconds','peak_reserved_mib','p6_labels_per_visit']))
    target=source/'NUMERIC_TABLES.md';target.write_text('\n'.join(lines)+'\n');print(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','test'],default='test');a=p.parse_args();main(a.phase)
