"""Frozen P1 gate arithmetic. Primary reviews before authorizing a branch."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v2_20261016 import OUT,dump,sha
from v2_analyze_20261016 import summarize


def main():
    for opt in ('adam','sgd'):
        for seed in (2022,2023,2024,2025):
            for arm in ('SN','SB'):
                r=OUT/'runs'/'kuairand'/opt/f'seed{seed}'/arm/'attempt1'
                assert (r/'run_manifest.json').exists(),str(r)+' incomplete'
    summary=pd.read_csv(OUT/'analysis'/'paired_seed_summary.csv')
    z=summary.query("dataset=='kuairand' and variant=='iid' and split=='validation' and checkpoint=='selected' and contrast=='SB-SN' and metric=='auc_macro'")
    adam=z[z.optimizer=='adam'].iloc[0].to_dict();sgd=z[z.optimizer=='sgd'].iloc[0].to_dict()
    cal=json.loads((OUT/'sgd_calibration_decision.json').read_text())
    adam_negative=adam['evidence']=='STRONG' and adam['mean']<0
    sgd_negative=sgd['evidence']=='STRONG' and sgd['mean']<0
    if adam_negative and sgd_negative:
        case='B';branch='P2-B';reason='Four-seed stable negative sampling effect also exists without Adam; necessity disproved in tested SGD regime, not equal effect magnitude.'
    elif adam_negative and not sgd_negative and abs(sgd['mean'])<=.5*abs(adam['mean']) and cal['optimization_adequate']:
        case='A';branch='P2-A';reason='Operational attenuation rule met. Adam necessity remains a candidate because optimizer regime and finite LR calibration are alternative explanations.'
    else:
        case='C';branch='P3/P7 boundary plus preregistered minimum exposure diagnostic';reason='Operational A/B not established; do not turn a wide/null contrast into equivalence or a mechanism.'
    additional_non_adam=case=='C' and sgd_negative
    if additional_non_adam:
        branch+='; additionally P2-B SGD and P7 SGD required by plan section6'
    interaction=np.array([sgd[f'seed{s}']-adam[f'seed{s}'] for s in (2022,2023,2024,2025)])
    dump(OUT/'p1_decision.json',dict(status='ARITHMETIC_COMPLETE_PENDING_PRIMARY_BRANCH_REVIEW',operational_case=case,
        next_branch=branch,reason=reason,adam_primary=adam,sgd_primary=sgd,paired_optimizer_by_sampling_interaction=summarize(interaction),
        calibration=cal,global_stop=False,source_sha256=sha(__file__),
        additional_non_adam_controls_required=additional_non_adam,
        protocol_scope='fixed fusion / single optimizer; historical Stage2D learned fusion / two optimizers kept separate'))
    print(json.dumps(dict(operational_case=case,next_branch=branch,adam_mean=adam['mean'],sgd_mean=sgd['mean'],interaction=summarize(interaction))))

if __name__=='__main__':main()
