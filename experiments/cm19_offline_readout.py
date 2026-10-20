"""Collect the complete planned raw local-state and cost diagnostics."""
import json
import pandas as pd
from cm19_train import OUT, dump


def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    local=[];moments=[];costs=[];seen=set();missing=[]
    for c in manifest['runs']:
        rr=OUT/'runs'/c['run_id']
        fields={k:c[k] for k in ['stage','dataset','model','variant','optimizer','decay_mode','seed']}
        if c['stage']=='G0' and c['model'] in ['fixed','star0','star1']:
            if not (rr/'local_mechanism_receipt.json').exists():missing.append(c['run_id']+'/local_mechanism')
            else:
                local.append(pd.read_csv(rr/'offline_local_counterfactual.csv').assign(**fields))
                moments.append(pd.read_csv(rr/'offline_moment_statistics.csv').assign(**fields))
        key=(c['model'],c['variant'],c['dataset'])
        if c['seed']==2022 and c['optimizer']=='adam' and key not in seen:
            seen.add(key)
            if not (rr/'cost_profile.json').exists():missing.append(c['run_id']+'/cost_profile')
            else:costs.append({**fields,**json.loads((rr/'cost_profile.json').read_text())})
    if missing:raise RuntimeError(f'{len(missing)} planned raw diagnostics still missing; wait for the existing queue.')
    assert len(local)==48 and len(costs)==78,(len(local),len(costs))
    target=OUT/'offline_readout';target.mkdir(exist_ok=True)
    pd.concat(local,ignore_index=True).to_csv(target/'common_state_counterfactuals.csv',index=False)
    pd.concat(moments,ignore_index=True).to_csv(target/'common_state_moments.csv.gz',index=False,compression='gzip')
    pd.DataFrame(costs).to_csv(target/'model_cost_profiles.csv',index=False)
    dump(target/'collection_receipt.json',dict(local_runs=len(local),cost_profiles=len(costs),planned_tasks=126,complete=True,test_used=False,local_scope='One hypothetical step, common weights/moments and eval-mode train-meta gradient, only factor groups change.',cost_scope='Supported dense FLOPs lower bound; shared-host CPU model-only latency. See COST_MEASUREMENT.md.'))
    print('Collected all126 planned offline raw diagnostics.')


if __name__=='__main__':main()
