"""Historical cell baseline, label population, and paired-seed context for N0."""
import json
import numpy as np
import pandas as pd
import n17_common as c
from n17_n1_control import paired


def main():
    out=c.OUT/'N0';manifest=json.loads((out/'readout_manifest.json').read_text())
    source=pd.read_csv(manifest['source_table']);population=pd.read_csv(out/'split_populations.csv')
    rows=[]
    selected=source[source.readout.isin(['common_epoch10','common_epoch20','common_epoch50','selected_macro_auc'])]
    for key,part in selected.groupby(['group','seed','readout','domain','task']):
        group,seed,readout,domain,task=key;record=dict(zip(['group','seed','readout','domain','task'],key))
        context=population[(population.group==group)&(population.seed==seed)&(population.domain==domain)].set_index('split')
        for split in ('train','selection','diagnostic','test'):
            record[split+'_n']=int(context.loc[split,'n']);record[split+'_positive']=int(context.loc[split,f't{task}_pos'])
        rate=record['test_positive']/record['test_n'];entropy=-rate*np.log(max(rate,1e-15))-(1-rate)*np.log(max(1-rate,1e-15))
        record.update(test_label_rate=rate,test_label_entropy=entropy)
        for item in part.itertuples():
            for arm in ('SN','SB'):record[arm+'_'+item.metric]=float(getattr(item,arm))
            record['delta_'+item.metric]=float(item.difference)
        rows.append(record)
    frame=pd.DataFrame(rows);frame.to_csv(out/'historical_cell_population_context.csv',index=False);summaries=[]
    for key,part in frame.groupby(['group','readout','domain','task']):
        record=dict(zip(['group','readout','domain','task'],[x.item() if isinstance(x,np.generic) else x for x in key]));record['n_seeds']=int(part.seed.nunique())
        for col in ('train_n','selection_n','diagnostic_n','test_n','test_label_rate','test_label_entropy'):
            record[col+'_min']=float(part[col].min());record[col+'_max']=float(part[col].max())
        for metric in ('auc','logloss'):
            for arm in ('SN','SB'):record[arm+'_'+metric+'_mean']=float(part[arm+'_'+metric].mean())
            stat=paired(part.sort_values('seed')['delta_'+metric]);record.update({f'delta_{metric}_{name}':stat[name] for name in ('mean','ci95','positive','negative','raw')})
        summaries.append(record)
    c.dump(out/'historical_cell_population_summary.json',summaries)
    c.dump(out/'cell_context_receipt.json',dict(status='COMPLETE_HISTORICAL_ONLY',rows=len(frame),summaries=len(summaries),
        source_sha=c.sha(__file__),source_table_sha=c.sha(manifest['source_table']),population_sha=c.sha(out/'split_populations.csv'),
        scope='SN is a reference model, not intrinsic domain difficulty; label entropy is descriptive; old ML seed varies population and optimization together; four training seeds, no domain causal regression'))
    # Compare precisely the same stored predictions/epochs, before calibration.
    historical=source[source.metric=='logloss'].melt(id_vars=['group','seed','readout','domain','task'],value_vars=['SN','SB'],var_name='arm',value_name='historical_LL')
    calibration=pd.read_csv(out/'calibration_cell_metrics.csv');calibration=calibration[calibration.split=='test']
    match=calibration.merge(historical,on=['group','seed','readout','domain','task','arm'],validate='one_to_one');assert len(match)==len(calibration)
    diagnostic=json.loads((out/'calibration_diagnostics.json').read_text())
    diagnostic['historical_test_LL_identity']=dict(compared_cells=len(match),max_absolute_residual=float((match.logloss_before-match.historical_LL).abs().max()),
        prediction_probability_clipped_cells=int(calibration.clipped.sum()),scope='original historical metric table versus before-calibration recomputation; no new model test')
    c.dump(out/'calibration_diagnostics.json',diagnostic)
    print('historical cell contexts',len(frame),'summaries',len(summaries))


if __name__=='__main__':main()
