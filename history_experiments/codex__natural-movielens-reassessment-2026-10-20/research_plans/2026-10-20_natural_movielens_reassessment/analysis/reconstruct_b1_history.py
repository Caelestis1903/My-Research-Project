"""Recorded B1 reconstruction: keep primary and sensitivity rules, stages and identities."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_core import interval

OLD=Path('/home/xzf/Code/phase0-3-git/research_plans/2026-10-15/execution_20261015/B1_reconstruction_20261016')
OUT=Path(__file__).resolve().parent/'b1_history'

def analyze():
    OUT.mkdir(exist_ok=True);sources=[]
    for name in ['analysis/selected_timing_decomposition.csv','analysis/same_epoch_differences.csv',
            'interpretation/stage_summary.csv','interpretation/selected_final_training_loss.csv','analysis_scope_gate.json']:
        source=OLD/name;data=source.read_bytes();target=OUT/(name.replace('/','__')+('.gz' if len(data)>1000000 else ''))
        saved=gzip.compress(data,mtime=0) if target.suffix=='.gz' else data;target.write_bytes(saved)
        sources.append(dict(source=str(source),copy=str(target),source_sha256=hashlib.sha256(data).hexdigest(),
            archive_sha256=hashlib.sha256(saved).hexdigest(),evidence='historical recorded CSV; no old predictions/checkpoints reloaded'))
    f=pd.read_csv(OLD/'analysis/selected_timing_decomposition.csv')
    assert np.allclose(f.DH-f.F,f.delta,rtol=1e-12,atol=1e-12)
    residual1=f.delta-f.policy_at_F-f.timing_on_DH;residual2=f.delta-f.policy_at_DH-f.timing_on_F
    assert residual1.abs().max()<1e-12 and residual2.abs().max()<1e-12
    rows=[]
    for key,g in f.groupby(['split','rule','metric','quantity']):
        assert set(g.seed)=={2022,2023,2024,2025} and not g.seed.duplicated().any()
        rows.append(dict(zip(['split','rule','metric','quantity'],key),F_mean=g.F.mean(),DH_mean=g.DH.mean(),**interval(g.delta)))
    pd.DataFrame(rows).to_csv(OUT/'selected_rule_seed_summary.csv',index=False)
    epoch=pd.read_csv(OLD/'analysis/same_epoch_differences.csv')
    assert np.allclose(epoch.DH-epoch.F,epoch.delta,rtol=1e-12,atol=1e-12)
    epoch['native_epoch_window']=epoch.epoch//10
    seed=epoch.groupby(['seed','split','metric','quantity','native_epoch_window']).delta.mean().reset_index()
    seed.to_csv(OUT/'paired_seed_stage_means.csv',index=False)
    stage=[]
    for key,g in seed.groupby(['split','metric','quantity','native_epoch_window']):
        stage.append(dict(zip(['split','metric','quantity','native_epoch_window'],key),
            completed_epoch_start=10*key[-1]+1,completed_epoch_end=10*(key[-1]+1),**interval(g.delta)))
    pd.DataFrame(stage).to_csv(OUT/'four_seed_stage_summary.csv',index=False)
    r1=f[f.rule=='R1'].set_index(['seed','split','metric','quantity'])
    r4=f[f.rule=='R4'].set_index(['seed','split','metric','quantity'])
    assert set(r1.index)==set(r4.index)
    assert r1[['F_epoch','DH_epoch']].equals(r4[['F_epoch','DH_epoch']])
    pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
    (OUT/'status.json').write_text(json.dumps(dict(recorded_B1_reconstruction_complete=True,formal_runs=8,
        identity='RECONSTRUCTION_NOT_HISTORICAL_EXACT',population='natural KuaiRand',target_physical_domain=1,
        selection_rules=['R0','R1','R2','R3','R4'],primary_rule='R0',
        R1_R4_selected_models_identical=True,paired_selected_records=len(f),same_epoch_records=len(epoch),
        timing_identity_maxabs=float(max(residual1.abs().max(),residual2.abs().max())),
        timing_is_arithmetic_not_same_theta_causality=True,new_training=0),indent=2)+'\n')
    print('B1 historical rule summaries',len(rows),'stage summaries',len(stage))
    print(pd.DataFrame(rows).query("split=='test' and quantity in ['macro','non_target','target']").to_string(index=False))

if __name__=='__main__':analyze()
