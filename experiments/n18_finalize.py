"""Primary scientific summaries and the minimum checks that bind the conclusions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from n18_common import OUT,dump

def main():
    cells=[];costs=[];receipts=[]
    for seed in range(2022,2026):
        source=OUT/'measurements'/f'seed{seed}';corrected=OUT/'measurements_functional'/f'seed{seed}'
        for p in [source/'receipt.json',corrected/'receipt.json',OUT/'P3_training'/f'seed{seed}'/'receipt.json']:
            r=json.loads(p.read_text());assert r['status']=='COMPLETE' and not r['test_access'];receipts.append(str(p.relative_to(OUT)))
        gram=np.load(corrected/'P4_gram.npz');assert not np.intersect1d(gram['A_row_ids'],gram['B_row_ids']).size
        samples=json.loads((corrected/'P6_samples.json').read_text())
        assert not ({r['row_id'] for r in samples['design']}&{r['row_id'] for r in samples['heldout']})
        fit=np.load(source/'features_fit.npz');sel=np.load(source/'features_selection.npz');diag=np.load(source/'features_diagnostic.npz')
        assert not np.intersect1d(sel['row_ids'],diag['row_ids']).size
        pred=np.load(OUT/'P3_training'/f'seed{seed}'/'predictions.npz')
        assert np.array_equal(pred['diagnostic_centered_repair'],pred['diagnostic_ordinary_one_cell'])
        metrics=pd.read_csv(OUT/'P3_training'/f'seed{seed}'/'metrics.csv');metrics=metrics[metrics.split=='diagnostic']
        assert np.isfinite(metrics[['auc','logloss','teacher_logit_mse']]).all().all()
        ref=metrics[metrics.model=='teacher'][['domain','task','auc','logloss']]
        joined=metrics.merge(ref,on=['domain','task'],suffixes=['','_ref'])
        joined['delta_auc']=joined.auc-joined.auc_ref;joined['delta_logloss']=joined.logloss-joined.logloss_ref;cells.append(joined)
        config=json.loads((source/'config.json').read_text());head=json.loads((OUT/'P3_training'/f'seed{seed}'/'config.json').read_text())
        trunk=config['parameter_count']-config['module_costs']['tower']
        for name,v in head['models'].items():
            stored=v['stored_head_scalars']+head['mean_std_storage']
            head_bytes=v['stored_head_scalars']*8+head['mean_std_storage']*4
            costs.append(dict(seed=seed,model=name,teacher_parameters=config['parameter_count'],common_trunk_parameters=trunk,
                coefficient_dof=v['coefficient_dof'],head_storage_scalars=v['stored_head_scalars'],normalizer_scalars=head['mean_std_storage'],
                total_storage_scalars=trunk+stored,whole_model_fraction_saved=1-(trunk+stored)/config['parameter_count'],
                head_and_normalizer_bytes=head_bytes,total_numeric_bytes=trunk*4+head_bytes,
                teacher_numeric_bytes=config['parameter_count']*4,
                numeric_byte_fraction_saved=1-(trunk*4+head_bytes)/(config['parameter_count']*4)))
    pd.concat(cells).to_csv(OUT/'readouts/P3_cell_deltas.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'readouts/P3_deployment_cost.csv',index=False)
    p1=json.loads((OUT/'theory_data/P1_support.json').read_text())
    assert [r['min_absolute_moment'] for r in p1['support_oracles']]==[1,0]
    p3=json.loads((OUT/'theory_data/P3_certificates.json').read_text())
    for r in p3['interval_examples']:
        if 'formula_feasible' in r:assert r['formula_feasible']==r['lp_feasible']==r['certificate']['feasible']
        if not r['certificate']['feasible']:assert r['certificate']['cycle_cost']<0
    p2=pd.read_csv(OUT/'theory_data/P2_mask_bounds.csv')
    assert ((p2.oracle_positive_rate>=p2.logic_lower)&(p2.oracle_positive_rate<=p2.logic_upper)).all()
    assert np.allclose(p2.logic_upper,p2.ordinary_same_information_lp_upper)
    p2['naive_width']=p2.naive_upper-p2.naive_lower;p2['logic_width']=p2.logic_upper-p2.logic_lower
    p2['known_MAR_IPW_estimate']=np.where(p2.mechanism=='MAR',p2.naive_lower/.4,np.nan)
    p2.to_csv(OUT/'readouts/P2_information_summary.csv',index=False)
    p6=pd.read_csv(OUT/'readouts/P6_variance.csv');assert p6.mean_preservation_error.max()<1e-12
    assert np.isfinite(p6[['covariance_trace','mean_direction_variance']]).all().all()
    assert (p6[['covariance_trace','mean_direction_variance']]>=0).all().all()
    dump(OUT/'SCIENTIFIC_ACCEPTANCE.json',dict(status='PASS',scope='only identities necessary for stated conclusions',
        verified_receipts=receipts,checks=['exact support residual','LP/cycle agreement','same-information bound equality',
          'heldout oracle inside identification envelope','disjoint measurement and evaluation rows',
          'equivalent repair predictions','centered finite covariance','importance mean preservation','finite cell metrics'],
        independent_review=False,test_access=False))
    # One standalone scientific artifact, not a new experimental comparison.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    means=p2.groupby('mechanism')[['naive_width','logic_width']].mean()
    xx=np.arange(2);axes[0,0].bar(xx-.18,means.naive_width,.36,label='No label logic')
    axes[0,0].bar(xx+.18,means.logic_width,.36,label='Known nested labels')
    axes[0,0].set(xticks=xx,xticklabels=means.index,title='P2: identification interval width',ylabel='Width (probability)');axes[0,0].legend(fontsize=8)
    cc=pd.concat(cells);rep=cc[cc.model=='centered_repair']
    for t in [0,1]:
        g=rep[rep.task==t].groupby('domain').delta_auc
        axes[0,1].errorbar(np.arange(3)+.08*(2*t-1),g.mean(),yerr=g.std(),fmt='o',label=f'Task {t}')
    axes[0,1].axhline(0,color='black',lw=.7);axes[0,1].axhline(-.003,color='red',ls=':',lw=.8)
    axes[0,1].set(xticks=[0,1,2],title='P3: repair vs frozen teacher',xlabel='Physical domain',ylabel='AUC delta (mean +/- seed SD)');axes[0,1].legend(fontsize=8)
    p4=pd.read_csv(OUT/'readouts/P4_interaction.csv')
    for block in ['all','shared']:
        g=p4[(p4.method=='full_6cell')&(p4.block==block)]
        axes[1,0].plot(g.seed,g.interaction_A_B_cosine,'o-',label=block)
    axes[1,0].set(xticks=list(range(2022,2026)),title='P4: independent train-batch interaction',ylabel='Cosine A vs B');axes[1,0].legend(fontsize=8)
    held=p6[p6.block=='heldout'];ratios=[]
    for s,g in held.groupby('seed'):
        g=g.set_index('method');ratios.append([s,100*(1-g.loc['direction_IS','mean_direction_variance']/g.loc['norm_IS','mean_direction_variance'])])
    ratios=np.array(ratios);axes[1,1].bar(ratios[:,0],ratios[:,1]);axes[1,1].axhline(10,color='red',ls=':',label='Frozen 10% gate')
    axes[1,1].set(xticks=list(range(2022,2026)),title='P6: direction IS vs norm IS',ylabel='Heldout projected variance reduction (%)');axes[1,1].legend(fontsize=8)
    fig.savefig(OUT/'RESEARCH_RESULTS.png',dpi=180);fig.savefig(OUT/'RESEARCH_RESULTS.pdf');plt.close(fig)

if __name__=='__main__':main()
