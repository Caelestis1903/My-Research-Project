"""Apply the already reviewed Q crossover criterion to each frozen role.

Temporary role aliases are used only to call the existing criterion; published
rows retain their actual role. No model, policy, slot, or request is selected.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analyze_core_four_seed import ACTIVE, REPORT, read, write, require_complete_export_inventory
from experiments.mdmt_rcf_20261026.campaign import _registered_crossovers


def main():
    inventory=read(ACTIVE/'FOUR_SEED_ACCEPTANCE.json');index=read(ACTIVE/'ARTIFACT_INDEX.json')
    if inventory['status']!='PASS' or not inventory['analysis_authorized']:raise PermissionError('Four seeds required')
    require_complete_export_inventory(inventory,index)
    rows=[];summaries=[];keys=defaultdict(dict)
    for item in index['prefixes']:
        root=Path(item['root']);diag=read(root/'RCF'/'mechanisms.json')
        with (root/'groups.csv').open(newline='') as stream:groups=list(csv.DictReader(stream))
        for role in ('Q_explore','validation','test_exposed','Q_legacy_confirm_exposed'):
            temporary={k:[dict(r,role='Q_explore' if r.get('role')==role else r.get('role'))
                          for r in diag[k] if r.get('role') in ('P_form',role)]
                       for k in ('crossovers','group_profiles')}
            controls={}
            for method in ('POOLED_TARGET','LOCAL_ACTIVE'):
                controls[method]=[dict(target=int(r['domain']),task=int(r['task']),group=int(r['group']),
                    logloss=float(r['logloss']) if r['logloss'] else None)
                    for r in groups if r['role']==role and r['method']==method]
            result=_registered_crossovers(dict(diagnostics={'RCF':temporary},group_metrics=controls))
            for row in result:rows.append(dict(row,role=role,dataset=item['dataset'],seed=item['seed']))
            selected={r['profile_id'] for r in result if r['supported_vs_controls']}
            keys[item['dataset'],role][int(item['seed'])]=selected
            summaries.append(dict(dataset=item['dataset'],seed=item['seed'],role=role,
                                  supported_crossing_pairs=len(result),pairs_exceeding_all_controls=len(selected)))
    repeated=[]
    for (dataset,role),seeds in keys.items():
        if set(seeds)!={5201,5202,5203,5204}:raise RuntimeError('Four crossover training seeds required')
        common=set.intersection(*seeds.values())
        repeated.append(dict(dataset=dataset,role=role,n_training_seeds=4,
                             repeated_profiles=sorted(common),count=len(common)))
    write('CROSSOVER_four_seed_per_seed.csv',rows)
    write('CROSSOVER_four_seed_counts.csv',summaries)
    receipt=dict(status='PASS',same_registered_Q_criterion=True,minimum_training_seeds=4,
                 actual_roles_preserved_in_output=True,repeated=repeated,
                 selection_note='Outer-role diagnosis only; no change to frozen Q decision or predictions')
    (REPORT/'CROSSOVER_FOUR_SEED_ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
