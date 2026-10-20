"""Primary N2/N3 analysis; finite checkpoints, MC error and seed units separated."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_core import interval

ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'n17_mechanisms'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def safe_ratio(a,b):return float(a/b) if b else np.nan

def analyze():
    manifest=json.loads((ART/'diagnostic_manifest.json').read_text())
    extra=json.loads((ART/'extra_precision_manifest.json').read_text())['configurations']
    configs=manifest['configurations']+extra
    assert len(manifest['configurations'])==48
    for cfg in configs:
        receipt=json.loads((Path(cfg['out'])/'extension_receipt.json').read_text())
        assert receipt['status']=='COMPLETE' and receipt['test_read'] is False
    OUT.mkdir(exist_ok=True)
    geometry=[];mapping=[];sgd=[];comparisons=[];local=[];localcells=[];sources=[]
    for cfg in configs:
        folder=Path(cfg['out']);native=json.loads((folder/'config.json').read_text())
        assert native['checkpoint_sha']==sha(cfg['checkpoint']) and native['test_read'] is False
        sources.append(dict(id=cfg['id'],kind=cfg['kind'],out=str(folder),
            receipt_sha256=sha(folder/'extension_receipt.json'),checkpoint_sha256=sha(cfg['checkpoint'])))
        if cfg['kind']=='n2':
            g=pd.DataFrame(json.loads((folder/'geometry.json').read_text()))
            g['count_prediction_fraction']=g.between_trace_prediction/g.sum_trace_prediction
            g['MC_over_prediction']=g.empirical_covariance_trace/g.sum_trace_prediction
            g['mean_error_over_RMS_SE']=g.mean_bias_norm/g.mc_rms_se
            g['diagnostic_id']=cfg['id'];geometry.append(g)
            m=pd.DataFrame(json.loads((folder/'adam_mapping.json').read_text()))
            m['diagnostic_id']=cfg['id'];mapping.append(m)
            s=pd.DataFrame(json.loads((folder/'same_state_sgd_mapping.json').read_text()))
            s['diagnostic_id']=cfg['id'];sgd.append(s)
            groups={key:[] for key in g.block.unique()};order=native['parameter_order']
            for record in order:
                ix=list(range(record['start'],record['end']));groups[record['block']].extend(ix)
                if record['block']=='C':
                    name=record['name']
                    sub='C_embedding' if name.startswith('embedding.') else 'C_shared_STAR' if name in ('shared_weight','shared_bias') else 'C_skip' if name.startswith('skip_conn.') else 'C_star_mlp' if name.startswith('star_mlp.') else 'C_experts' if name.startswith('expert.') else None
                    assert sub is not None;groups[sub].extend(ix)
            groups['ALL']=list(range(order[-1]['end']))
            for variant in ('IID','quota'):
                with np.load(folder/f'SN_{variant}_moments.npz') as z:sn={k:z[k] for k in ('mean','mean_update','mean_linear_update')}
                with np.load(folder/f'SB_{variant}_moments.npz') as z:sb={k:z[k] for k in ('mean','mean_update','mean_linear_update')}
                for block,indices in groups.items():
                    ix=np.asarray(indices,dtype=int)
                    a=g[(g.arm=='SN')&(g.variant==variant)&(g.block==block)].iloc[0]
                    b=g[(g.arm=='SB')&(g.variant==variant)&(g.block==block)].iloc[0]
                    ma=m[(m.arm=='SN')&(m.variant==variant)&(m.block==block)].iloc[0]
                    mb=m[(m.arm=='SB')&(m.variant==variant)&(m.block==block)].iloc[0]
                    grad_gap=float(np.linalg.norm(sb['mean'][ix]-sn['mean'][ix]))
                    update_gap=float(np.linalg.norm(sb['mean_update'][ix]-sn['mean_update'][ix]))
                    fixed_d_gap=float(np.linalg.norm(sb['mean_linear_update'][ix]-sn['mean_linear_update'][ix]))
                    comparisons.append(dict(seed=cfg['seed'],anchor_arm=cfg['anchor_arm'],epoch=cfg['epoch'],variant=variant,block=block,
                        SB_over_SN_gradient_trace=safe_ratio(b.empirical_covariance_trace,a.empirical_covariance_trace),
                        same_target_gradient_mean_difference_norm=grad_gap,
                        gradient_gap_over_RMS_sum=safe_ratio(grad_gap,a.mc_rms_se+b.mc_rms_se),
                        same_state_Adam_mean_update_difference_norm=update_gap,
                        Adam_gap_over_RMS_sum=safe_ratio(update_gap,ma.update_mc_rms_se+mb.update_mc_rms_se),
                        same_previous_v_denominator_mean_update_difference_norm=fixed_d_gap,
                        same_state_SGD_mean_update_difference_norm=cfg['lr']*grad_gap,
                        interpretation='vector norms have positive finite-MC floor; RMS sum is an error scale, not a CI or historical-state necessity test'))
        else:
            f=pd.read_csv(folder/'raw_contrasts.csv');f['chunk']=cfg['id'];local.append(f)
            f=pd.read_csv(folder/'cell_readouts.csv');f['bank']=cfg['bank'];f['chunk']=cfg['id'];localcells.append(f)
    pd.concat(geometry,ignore_index=True).to_csv(OUT/'N2_geometry.csv',index=False)
    pd.concat(mapping,ignore_index=True).to_csv(OUT/'N2_Adam_mapping.csv',index=False)
    pd.concat(sgd,ignore_index=True).to_csv(OUT/'N2_SGD_and_lagged_mapping.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'N2_paired_estimator_mean_maps.csv',index=False)
    raw=pd.concat(local,ignore_index=True)
    keys=['seed','anchor_arm','epoch','bank','draw','split','case']
    assert not raw.duplicated(keys).any()
    raw.to_csv(OUT/'N3_raw_paired_draws.csv',index=False)
    pd.concat(localcells,ignore_index=True).to_csv(OUT/'N3_all_cell_readouts.csv',index=False)
    metrics=[m+'_'+ref for m in ('auc','logloss','train_J') for ref in ('minus_SB','minus_SN','minus_anchor')]
    # Norm x direction interaction is paired within a draw, never a difference
    # of independent SEs. All three cases share the same native SB reference.
    interactions=[]
    for key,g in raw.groupby(['seed','anchor_arm','epoch','bank','draw','split']):
        g=g.set_index('case')
        for block in ('C','C_embedding','C_shared_STAR','C_skip','C_star_mlp','C_experts'):
            cases=[block+'_'+suffix for suffix in ('full','norm','direction')]
            if not set(cases)<=set(g.index):continue
            if g.loc[cases,'direction_undefined'].astype(bool).any():continue
            interactions.append(dict(zip(['seed','anchor_arm','epoch','bank','draw','split'],key),case=block+'_norm_direction_interaction',block=block,
                **{metric:float(g.loc[cases[0],metric]-g.loc[cases[1],metric]-g.loc[cases[2],metric]) for metric in metrics if metric.endswith('minus_SB')}))
    interaction=pd.DataFrame(interactions);interaction.to_csv(OUT/'N3_paired_norm_direction_interactions.csv',index=False)
    valid=raw[~raw.direction_undefined.astype(bool)].copy()
    combined=pd.concat([valid,interaction],ignore_index=True)
    summaries=[]
    for key,g in combined.groupby(['seed','anchor_arm','epoch','bank','split','case','block']):
        for metric in metrics:
            values=g[metric].dropna().to_numpy()
            if not len(values):continue
            summaries.append(dict(zip(['seed','anchor_arm','epoch','bank','split','case','block'],key),metric=metric,
                draws=len(values),mean=float(values.mean()),mcse=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else np.nan))
    banks=pd.DataFrame(summaries);banks.to_csv(OUT/'N3_bank_MC_summary.csv',index=False)
    seeds=[]
    for key,g in banks.groupby(['seed','anchor_arm','epoch','split','case','block','metric']):
        # Preserve undefined/missing cases rather than converting them to zero.
        # With all draws defined both banks have equal registered counts.
        weights=g.draws/g.draws.sum()
        seeds.append(dict(zip(['seed','anchor_arm','epoch','split','case','block','metric'],key),
            mean=float(np.sum(weights*g['mean'])),mcse=float(np.sqrt(np.sum(np.square(weights*g.mcse)))),
            valid_draws=int(g.draws.sum()),independent_banks=g.bank.nunique(),
            both_banks_present=set(g.bank)=={0,1},defined_counts_equal=g.draws.nunique()==1))
    seeds=pd.DataFrame(seeds);seeds.to_csv(OUT/'N3_seed_anchor_summary.csv',index=False)
    across=[];incomplete=[]
    for key,g in seeds.groupby(['anchor_arm','epoch','split','case','block','metric']):
        if set(g.seed)!={2022,2023,2024,2025} or not g.both_banks_present.all():
            incomplete.append(dict(zip(['anchor_arm','epoch','split','case','block','metric'],key),
                available_seeds=sorted(g.seed.tolist()),reason='undefined/missing local direction or bank; no formal four-seed mean'))
            continue
        across.append(dict(zip(['anchor_arm','epoch','split','case','block','metric'],key),**interval(g['mean']),
            inference='four optimizer seeds on fixed population; anchors/draws/banks are not independent population replicates'))
    pd.DataFrame(across).to_csv(OUT/'N3_four_seed_summary.csv',index=False)
    (OUT/'N3_incomplete_cases.json').write_text(json.dumps(incomplete,indent=2)+'\n')
    pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(complete=True,base_tasks=48,extra_precision_tasks=len(extra),
        test_read=False,historical_optimizer_history_necessity_identified=False,
        moment_mapping='before parameter rounding for N2; actual native rounded updates for N3'),indent=2)+'\n')
    print('N17 mechanisms complete:',len(configs),'tasks; four-seed summaries',len(across))

if __name__=='__main__':analyze()
