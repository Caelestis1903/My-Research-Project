"""Primary collection of full-coordinate, fixed-state core diagnostics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ART=Path('/home_new/xzf_natural_movielens_reassessment')
OUT=Path(__file__).resolve().parent/'core_mechanism'

def ratio(a,b):return float(a/b) if b else None

def analyze():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ART/'core_manifest.json').read_text());runs={r['run_id']:r for r in manifest['runs']}
    covariance=[];blocks=[];oracles=[];adam=[];risk=[];missing=[];sources=[]
    for cfg in manifest['offline_diagnostics']:
        path=Path(cfg['output'])
        if not path.exists():missing.append(str(path));continue
        x=json.loads(path.read_text());parent=runs[cfg['run_id']]
        assert x['full_coordinate'] and x['mc_independent_batches']==512 and x['batch_size']==40960
        assert x['SN']['samples']==x['SB']['samples']==512
        assert x['checkpoint_sha256']==hashlib.sha256((Path(cfg['run_dir'])/cfg['checkpoint']).read_bytes()).hexdigest()
        base={k:parent[k] for k in ('run_id','population','budget','seed','arm')};base['checkpoint']=cfg['checkpoint']
        sn=x['SN'];sb=x['SB']
        covariance.append(dict(base,SN_trace=sn['covariance_trace'],SB_trace=sb['covariance_trace'],
            SB_over_SN_trace=ratio(sb['covariance_trace'],sn['covariance_trace']),
            SN_mean_squared_norm=sn['mean_squared_norm'],SB_mean_squared_norm=sb['mean_squared_norm'],
            same_target_mean_difference_norm=x['same_target_mc_mean_difference_norm'],
            SN_mean_RMS_SE=sn['mean_standard_error_norm'],SB_mean_RMS_SE=sb['mean_standard_error_norm'],
            mean_difference_over_RMS_sum=ratio(x['same_target_mc_mean_difference_norm'],sn['mean_standard_error_norm']+sb['mean_standard_error_norm']),
            uncertainty='RMS is a vector MC error scale, not confidence interval; trace MCSE not saved'))
        for name in sn['block_covariance_trace']:
            a=sn['block_covariance_trace'][name];b=sb['block_covariance_trace'][name]
            blocks.append(dict(base,partition=name,SN_trace=a,SB_trace=b,SB_over_SN_trace=ratio(b,a),
                scope='all original coordinates in named partition; main optimizer excludes A_arch'))
        finite=x['finite_stratified_pool_oracle']
        for arm in ('SN','SB'):
            identity=finite['identity'][arm];mc=finite['monte_carlo'][arm]
            oracles.append(dict(base,estimator=arm,finite_rows=96,rows_per_domain=32,
                expected_gradient_error_norm=identity['full_coordinate_error_norm'],
                exact_finite_batch_trace=identity['exact_batch_covariance_trace'],
                MC_trace=mc['covariance_trace'],MC_over_exact=ratio(mc['covariance_trace'],identity['exact_batch_covariance_trace']),
                MC_error_vs_exact_mean_norm=mc['mc_error_norm_vs_exact_oracle'],MC_mean_RMS_SE=mc['mean_standard_error_norm']))
        grid=x['adam_same_baseline_grid']
        if grid['applicable']:
            for row in grid['grid']:adam.append(dict(base,**row,scope=grid['definition']))
        jac=x['prediction_risk_jacobian'];assert jac['prediction_max_abs_difference']==0
        for block,squared in jac['block_difference_squared_norm'].items():
            risk.append(dict(base,partition=block,block_gradient_difference_squared_norm=squared,
                functional_risk_gradient_norm=jac['all_coordinate_gradient_norm'],
                training_derivative_norm=jac['training_derivative_norm'],
                gradient_difference_norm=jac['difference_norm'],prediction_max_abs_difference=0,
                warning='equal function values; reattached derivative is a separate diagnostic, not the actual training field'))
        sources.append(dict(base,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_sha256=x['checkpoint_sha256']))
    for name,rows in [('covariance',covariance),('block_covariance',blocks),('finite_pool_identity',oracles),
            ('same_baseline_adam_grid',adam),('functional_vs_training_derivative',risk),('source_index',sources)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'analysis_status.json').write_text(json.dumps(dict(planned=160,completed=len(sources),missing=missing,
        complete=len(sources)==160,formal_training_runs=0,
        interpretation_limits=['finite96-row identity is not full-population covariance',
            'same-baseline grid is a synthetic history intervention, not necessity of historical Adam state',
            'gradient trace is not actual optimizer update variance or generalization performance',
            'shared original model function and reattached functional derivative must not be conflated']),indent=2)+'\n')
    print('Core diagnostics',len(sources),'/160')

if __name__=='__main__':analyze()
