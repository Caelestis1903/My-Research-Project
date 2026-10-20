"""Recalculate recorded N17 seed arrays and preserve finite-state mechanism evidence."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t

OLD=Path('/home/xzf/Code/phase0-3-20261017-git/research_plans/2026-10-17/execution_20261017')
OUT=Path(__file__).resolve().parent/'n17_history'

def analyze():
    OUT.mkdir(exist_ok=True)
    names=['FINAL_STAGE_MACRO_AUC.csv','FINAL_SELECTED_ENDPOINTS.csv','FINAL_SELECTED_CELL_METRICS.csv',
        'N5_winners_frozen.json','N1_winners_frozen.json','FINAL_performance/paired_seed_endpoints.csv',
        'N2_fixedpoint_summary/raw_moments.csv',
        'N2_fixedpoint_summary/trace_contrasts.csv','N2_fixedpoint_summary/mean_mapping_contrasts.csv',
        'N2_fixedpoint_summary/zero_sampled_gradient_blocks.csv',
        'N3_development_summary/conditional_summary.csv','N3_confirmation_summary/conditional_summary.csv',
        'N3_confirmation_summary/confirmation_gate.json']
    index=[];audit=[]
    for name in names:
        source=OLD/name;target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        index.append(dict(source=str(source),copy=str(target),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            evidence='recorded table/seed array, not a new evaluation of old predictions or checkpoints'))
        if name in ('FINAL_STAGE_MACRO_AUC.csv','FINAL_SELECTED_ENDPOINTS.csv','FINAL_SELECTED_CELL_METRICS.csv'):

            f=pd.read_csv(source)
            for r in f.itertuples():
                v=np.asarray(ast.literal_eval(r.raw),dtype=float);assert len(v)==r.n
                mean=v.mean();se=v.std(ddof=1)/np.sqrt(len(v));ci=np.asarray(ast.literal_eval(r.ci95))
                half=t.ppf(.975,len(v)-1)*se
                assert abs(mean-r.mean)<1e-12 and abs(se-r.se)<1e-12
                assert np.allclose(ci,np.array([mean-half,mean+half]),rtol=1e-12,atol=1e-12)
                assert int((v>0).sum())==r.positive and int((v<0).sum())==r.negative
                audit.append(dict(table=name,group=r.group,readout=r.readout,metric=r.metric,n=r.n,
                    recomputed_mean=mean,mean_residual=mean-r.mean,recomputed_se=se,
                    ci_low=mean-half,ci_high=mean+half,positive=int((v>0).sum()),negative=int((v<0).sum())))
    pd.DataFrame(audit).to_csv(OUT/'recorded_seed_array_recalculation.csv',index=False)
    paired=pd.read_csv(OLD/'FINAL_performance/paired_seed_endpoints.csv')
    assert np.allclose(paired.SB-paired.SN,paired.difference,rtol=1e-10,atol=1e-10)
    summaries=[]
    for key,g in paired.groupby(['group','readout','metric','estimand']):
        assert not g.seed.duplicated().any() and set(g.seed)==set(range(2032,2040))
        v=g.difference;mean=v.mean();se=v.std(ddof=1)/np.sqrt(len(v));half=t.ppf(.975,len(v)-1)*se
        summaries.append(dict(zip(['group','readout','metric','estimand'],key),n=len(v),
            SN_mean=g.SN.mean(),SB_mean=g.SB.mean(),mean=mean,se=se,ci95_low=mean-half,ci95_high=mean+half,
            positive=int((v>0).sum()),negative=int((v<0).sum()),
            population='old artificial d2_large' if key[0].startswith('N5') else 'old natural KuaiRand'))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'all_endpoint_seed_summary.csv',index=False)
    iid=paired[paired.group=='N1_common_adam'].set_index(['seed','readout','metric'])
    quota=paired[paired.group=='N2_quota_adam'].set_index(['seed','readout','metric'])
    assert set(iid.index)==set(quota.index)
    controls=[]
    for ix in iid.index:
        controls.append(dict(seed=ix[0],readout=ix[1],metric=ix[2],
            SN_quota_minus_IID=quota.loc[ix,'SN']-iid.loc[ix,'SN'],
            SB_quota_minus_IID=quota.loc[ix,'SB']-iid.loc[ix,'SB'],
            quota_by_q_interaction=quota.loc[ix,'difference']-iid.loc[ix,'difference']))
    pd.DataFrame(controls).to_csv(OUT/'quota_by_q_paired_seed_controls.csv',index=False)


    g=pd.read_csv(OLD/'N2_fixedpoint_summary/raw_moments.csv');a=g[(g.block=='ALL')&(g.variant=='IID')].copy()
    a['count_fraction_recomputed']=a.between_trace_prediction/a.sum_trace_prediction
    a.to_csv(OUT/'N2_ALL_IID_count_context.csv',index=False)
    traces=pd.read_csv(OLD/'N2_fixedpoint_summary/trace_contrasts.csv')
    traces[traces.block=='ALL'].to_csv(OUT/'N2_ALL_trace_contrasts.csv',index=False)
    m=pd.read_csv(OLD/'N2_fixedpoint_summary/mean_mapping_contrasts.csv');m=m[m.block=='ALL'].copy()
    m['Adam_gap_over_marginal_RMS_sum']=m.mean_update_difference_norm/m.mean_update_difference_mc_rms_upper_bound
    m['lagged_gap_over_native_gap']=m.fixed_denominator_mean_update_difference_norm/m.mean_update_difference_norm
    m.to_csv(OUT/'N2_ALL_mean_mapping_context.csv',index=False)
    gate=json.loads((OLD/'N3_confirmation_summary/confirmation_gate.json').read_text())
    pd.DataFrame([dict(entry=name,**row) for name,row in gate['entries'].items()]).to_csv(OUT/'N3_frozen_independent_confirmation_entries.csv',index=False)
    pd.DataFrame(index).to_csv(OUT/'source_index.csv',index=False)
    (OUT/'status.json').write_text(json.dumps(dict(historical_record_reconstruction_complete=True,
        recorded_seed_array_rows_recalculated=len(audit),N4_trigger=gate['N4_trigger'],
        old_formal_seeds=list(range(2032,2040)),old_diagnostic_seeds=list(range(2032,2036)),
        independent_N3_confirmation_seeds=list(range(2036,2040)),
        not_seed_paired_with_new_population=True,raw_old_large_artifacts_reloaded=False),indent=2)+'\n')
    print('Recalculated recorded seed arrays',len(audit))
    print('ALL IID count fraction range',a.count_fraction_recomputed.min(),a.count_fraction_recomputed.max())
    print('ALL Adam gap/RMS range',m.Adam_gap_over_marginal_RMS_sum.min(),m.Adam_gap_over_marginal_RMS_sum.max())

if __name__=='__main__':analyze()
