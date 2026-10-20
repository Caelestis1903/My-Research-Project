"""Analyze the complete 16-anchor N2 geometry substage; no N3 or test access."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'n17_n2_geometry'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ratio(a,b):return a/b if b else np.nan

def analyze():
    configs=[c for c in json.loads((ART/'diagnostic_manifest.json').read_text())['configurations'] if c['kind']=='n2']
    assert len(configs)==16
    geometry=[];maps=[];linear=[];sources=[]
    for c in configs:
        p=Path(c['out']);r=json.loads((p/'extension_receipt.json').read_text())
        assert r['status']=='COMPLETE' and r['test_read'] is False
        native=json.loads((p/'config.json').read_text())
        assert native['k']==512 and native['test_read'] is False and native['checkpoint_sha']==sha(c['checkpoint'])
        assert (native['seed'],native['anchor_arm'],native['epoch'])==(c['seed'],c['anchor_arm'],c['epoch'])
        for filename,bag in [('geometry.json',geometry),('adam_mapping.json',maps),('same_state_sgd_mapping.json',linear)]:
            f=pd.DataFrame(json.loads((p/filename).read_text()));f['diagnostic_id']=c['id'];bag.append(f)
        for filename in ['extension_receipt.json','config.json','geometry.json','adam_mapping.json','same_state_sgd_mapping.json']:
            sources.append(dict(diagnostic_id=c['id'],path=str(p/filename),sha256=sha(p/filename)))
    OUT.mkdir(exist_ok=True)
    g=pd.concat(geometry,ignore_index=True);m=pd.concat(maps,ignore_index=True);s=pd.concat(linear,ignore_index=True)
    keys=['seed','anchor_arm','epoch','arm','variant','block','diagnostic_id']
    joint=g.merge(m[keys+['mean_update_norm','update_mc_rms_se','jensen_residual_norm']],on=keys,validate='one_to_one')
    joint=joint.merge(s[keys+['lagged_common_denominator_update_covariance_trace','sgd_update_covariance_trace']],on=keys,validate='one_to_one')
    joint['Adam_update_trace']=joint.k*joint.update_mc_rms_se**2
    joint['count_fraction_of_prediction']=joint.between_trace_prediction/joint.sum_trace_prediction
    joint['empirical_over_estimated_prediction']=joint.empirical_covariance_trace/joint.sum_trace_prediction
    joint['mean_error_over_RMS_SE']=joint.mean_bias_norm/joint.mc_rms_se
    joint.to_csv(OUT/'joint_geometry_and_update_traces.csv',index=False)
    shares=joint[(joint.variant=='IID')&joint.block.isin(['ALL','C','P','E','R'])].copy()
    totals=shares[shares.block=='ALL'][['diagnostic_id','arm','empirical_covariance_trace','Adam_update_trace']].rename(columns={'empirical_covariance_trace':'all_gradient_trace','Adam_update_trace':'all_update_trace'})
    shares=shares.merge(totals,on=['diagnostic_id','arm'],validate='many_to_one')
    shares['gradient_share']=shares.empirical_covariance_trace/shares.all_gradient_trace
    shares['update_share']=shares.Adam_update_trace/shares.all_update_trace
    shares.to_csv(OUT/'partition_trace_shares.csv',index=False)
    pairs=[]
    for key,f in joint.groupby(['seed','anchor_arm','epoch','variant','block']):
        f=f.set_index('arm');a=f.loc['SB'];b=f.loc['SN']
        pairs.append(dict(zip(['seed','anchor_arm','epoch','variant','block'],key),coordinates=int(a.coordinates),
            gradient_trace_ratio=ratio(a.empirical_covariance_trace,b.empirical_covariance_trace),
            Adam_update_trace_ratio=ratio(a.Adam_update_trace,b.Adam_update_trace),
            previous_v_linear_trace_ratio=ratio(a.lagged_common_denominator_update_covariance_trace,b.lagged_common_denominator_update_covariance_trace),
            SGD_affine_trace_ratio=ratio(a.sgd_update_covariance_trace,b.sgd_update_covariance_trace)))
    pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'SB_over_SN_trace_ratios.csv',index=False)
    summary=[]
    for key,f in pairs.groupby(['anchor_arm','epoch','variant','block']):
        assert set(f.seed)=={2022,2023,2024,2025}
        for metric in ['gradient_trace_ratio','Adam_update_trace_ratio','previous_v_linear_trace_ratio','SGD_affine_trace_ratio']:
            v=f[metric];valid=v[np.isfinite(v)]
            summary.append(dict(zip(['anchor_arm','epoch','variant','block'],key),metric=metric,n_seeds=4,n_defined=len(valid),
                mean=valid.mean() if len(valid)==4 else np.nan,min=valid.min() if len(valid)==4 else np.nan,max=valid.max() if len(valid)==4 else np.nan,
                above_one=int((valid>1).sum()),below_one=int((valid<1).sum()),
                inference='descriptive optimizer-seed range at one anchor/epoch; not independent population or ratio MC confidence interval'))
    pd.DataFrame(summary).to_csv(OUT/'four_seed_trace_ratio_summary.csv',index=False)
    pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(scope='N2 geometry and update covariance only',complete_anchors=16,k_per_estimator=512,
        test_read=False,N3_closed=False,within_prediction='estimated using same MC domain-centered batch gradients; not independent analytic validation',
        covariance_scope='full-coordinate diagonal moments and trace; no dense cross-coordinate covariance matrix',
        update_scope='before parameter storage rounding; SGD affine and previous-v transforms reuse gradient samples'),indent=2)+'\n')
    print(pd.DataFrame(summary).query("block=='ALL'").round(6).to_string(index=False))

if __name__=='__main__':analyze()
