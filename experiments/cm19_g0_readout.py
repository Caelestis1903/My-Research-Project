"""Descriptive G0 evidence from the entire completed development scope; no test access."""
import json
import pandas as pd
from cm19_train import OUT, dump


def main():
    root=OUT/'development_g0_readout'
    receipt=json.loads((root/'collection_receipt.json').read_text())
    assert receipt['stage']=='G0' and receipt['independent_runs']==160 and not receipt['test_metrics_used']
    cc=pd.read_csv(root/'seed_cell_metrics.csv')
    cc=cc[cc.intervention=='none']
    native=cc[cc.model=='fixed']
    match=cc[cc.model!='fixed'].merge(native,on=['dataset','optimizer','seed','checkpoint','domain','task'],suffixes=('_model','_fixed'),validate='many_to_one')
    match['delta_auc']=match.auc_model-match.auc_fixed
    match['delta_logloss']=match.logloss_model-match.logloss_fixed
    keys=['dataset','model_model','optimizer','checkpoint','domain','task']
    match[keys+['seed','delta_auc','delta_logloss']].rename(columns={'model_model':'model'}).to_csv(root/'task_matched_fixed_reference_seed_cells.csv',index=False)
    stats=match.groupby(keys).agg(delta_auc=('delta_auc','mean'),auc_positive_seeds=('delta_auc',lambda x:int((x>0).sum())),auc_negative_seeds=('delta_auc',lambda x:int((x<0).sum())),delta_logloss=('delta_logloss','mean'),loss_positive_seeds=('delta_logloss',lambda x:int((x>0).sum())),loss_negative_seeds=('delta_logloss',lambda x:int((x<0).sum())),n_seeds=('seed','nunique')).reset_index().rename(columns={'model_model':'model'})
    assert stats.n_seeds.eq(4).all()
    stats.to_csv(root/'task_matched_fixed_reference_cell_summary.csv',index=False)
    paths=pd.read_csv(root/'path_diagnostics.csv.gz')
    keys=['dataset','model','optimizer','label','probe','intervention']
    paths.groupby(keys).agg(mean_delta_p=('delta_p','mean'),max_delta_p=('max_delta','max'),cells_exact_zero=('delta_p',lambda x:int((x==0).sum())),seed_cell_rows=('delta_p','size'),mean_delta_auc_neutral_minus_natural=('delta_auc','mean'),mean_delta_logloss_neutral_minus_natural=('delta_logloss','mean')).reset_index().to_csv(root/'native_probe_summary.csv',index=False)
    paths.groupby(keys+['seed']).agg(mean_delta_p=('delta_p','mean'),max_delta_p=('max_delta','max'),cells_exact_zero=('delta_p',lambda x:int((x==0).sum())),cell_rows=('delta_p','size')).reset_index().to_csv(root/'native_probe_seed_summary.csv',index=False)
    mm=pd.read_csv(root/'seed_macro_metrics.csv')
    keys=['dataset','model','optimizer','seed']
    stages=mm[mm.checkpoint=='final'].merge(mm[mm.checkpoint=='selected'],on=keys,suffixes=('_final','_selected'),validate='one_to_one')
    stages['delta_auc_final_minus_selected']=stages.macro_auc_final-stages.macro_auc_selected
    stages['delta_logloss_final_minus_selected']=stages.macro_logloss_final-stages.macro_logloss_selected
    stages[keys+['delta_auc_final_minus_selected','delta_logloss_final_minus_selected']].to_csv(root/'training_stage_seed_deltas.csv',index=False)
    stages.groupby(keys[:-1]).agg(auc_delta=('delta_auc_final_minus_selected','mean'),auc_positive_seeds=('delta_auc_final_minus_selected',lambda x:int((x>0).sum())),loss_delta=('delta_logloss_final_minus_selected','mean'),loss_positive_seeds=('delta_logloss_final_minus_selected',lambda x:int((x>0).sum()))).reset_index().to_csv(root/'training_stage_summary.csv',index=False)
    dump(root/'descriptive_evidence_receipt.json',dict(training_scope='all 160 G0 runs',split='development',test_used=False,matched_cell_rows=len(match),task_matching='same task, physical domain, optimization seed, optimizer and checkpoint rule; different model capacities retained',probe_rows='cells within a seed are not independent replications'))
    print('Wrote descriptive G0 evidence; no test access or new training.')


if __name__=='__main__':main()
