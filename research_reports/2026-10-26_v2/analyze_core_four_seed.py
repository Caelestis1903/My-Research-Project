"""Aggregate completed, frozen four-seed exports; no model or training access."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import numpy as np

REPO = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
ACTIVE = REPO/'artifacts/mdmt_rcf_20261026/v2_four_seed'
T95 = 3.182446305284263


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    rows = list(rows)
    path = REPORT/'tables'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def stats(values):
    values=np.asarray(values,float)
    if len(values)!=4 or not np.isfinite(values).all():
        raise RuntimeError('Exactly four finite, independent training seeds required')
    mean=float(values.mean());sd=float(values.std(ddof=1));half=T95*sd/2
    return dict(n_training_seeds=4,mean=mean,sample_sd=sd,t95_low=mean-half,t95_high=mean+half,
                negative_seeds=int((values<0).sum()),positive_seeds=int((values>0).sum()))


def require_complete_export_inventory(acceptance,index):
    expected={(c['dataset'],int(s),c['method']) for c in acceptance['conditions'] for s in c['training_seeds']}
    actual={(p['dataset'],int(p['seed']),method) for p in index['prefixes'] for method in p['prediction_methods']}
    if actual!=expected:
        raise PermissionError('Frozen exports must cover every completed four-seed condition: '+
                              str(dict(missing=sorted(expected-actual),extra=sorted(actual-expected))))


def export_role_audit_tables():
    audit=read(ACTIVE/'DATA_ROLE_MANIFEST.json');cells=[];counts=[]
    for dataset,value in audit.items():
        for role,detail in value['role_details'].items():
            for row in detail['cells']:
                cells.append(dict(dataset=dataset,role=role,**row,
                    positive_rate=row['positive']/row['n'] if row['n'] else None))
            domains={int(c['domain']):c for c in detail['cells'] if int(c['task'])==0}
            total=sum(c['n'] for c in domains.values())
            counts.extend(dict(dataset=dataset,role=role,domain=d,n=c['n'],domain_share=c['n']/total)
                          for d,c in domains.items())
    write('DATA_ROLE_six_cells.csv',cells);write('DATA_ROLE_domain_counts.csv',counts)


def main():
    acceptance=read(ACTIVE/'FOUR_SEED_ACCEPTANCE.json')
    if acceptance.get('status')!='PASS' or not acceptance.get('analysis_authorized'):
        raise PermissionError('Four-seed completion evidence is required before analysis')
    frozen=read(ACTIVE/'FROZEN_MANIFEST.json')
    if frozen['four_seed_acceptance_sha256']!=sha(ACTIVE/'FOUR_SEED_ACCEPTANCE.json'):
        raise PermissionError('Seed inventory changed after freeze')
    index=read(ACTIVE/'ARTIFACT_INDEX.json')
    require_complete_export_inventory(acceptance,index)
    export_role_audit_tables()
    config=read(ACTIVE/'effective_config.json')
    cohorts={s:'development' for s in config['seeds']['development']}
    cohorts.update({s:'locked' for s in config['seeds']['locked']})
    expected={'development':set(config['seeds']['development']),'locked':set(config['seeds']['locked'])}
    if any(len(s)!=4 for s in expected.values()):
        raise RuntimeError('The registered cohorts must each contain four seeds')
    metrics=[];groups=[];formations=[];selections=[];profiles=[];retention=[];drift=[];native=[];manifests=[];bootstraps=[]
    scalar_names=('auc','logloss','natural_logloss','pooled_auc')
    for item in index['prefixes']:
        root=Path(item['root']);dataset=item['dataset'];seed=int(item['seed']);cohort=cohorts[seed]
        ok=read(root/'EXPORT_ACCEPTANCE.json')
        if ok['status']!='PASS':
            raise RuntimeError(f'Export acceptance not passed: {root}')
        prefix=read(ACTIVE/'prefix'/dataset/f'seed{seed}'/'PREFIX_COMPLETE.json')
        native.append(dict(dataset=dataset,seed=seed,cohort=cohort,**prefix))
        for row in csv_rows(root/'metrics.csv'):
            row.update(cohort=cohort)
            metrics.append(row)
        for row in csv_rows(root/'groups.csv'):
            row.update(cohort=cohort)
            groups.append(row)
        for row in read(root/'BOOTSTRAP.json'):
            bootstraps.append(dict(dataset=dataset,seed=seed,cohort=cohort,**row))
        manifests.append(dict(dataset=dataset,seed=seed,cohort=cohort,root=str(root),
                              index_sha256=sha(root/'ARTIFACT_INDEX.json'),
                              acceptance_sha256=sha(root/'EXPORT_ACCEPTANCE.json'),
                              indexed_files=len(item['files']),prediction_methods=item['prediction_methods']))
        for method in item['prediction_methods']:
            selection=read(root/method/'selection.json')
            selections.append(dict(dataset=dataset,seed=seed,cohort=cohort,method=method,
                                   **selection['selection'],reader_identity=selection['identity']))
        for method in item['formation_methods']:
            common=dict(dataset=dataset,seed=seed,cohort=cohort,method=method)
            formed=read(root/method/'formation.json')
            for row in formed['records']:
                if 'projection_c' in row or row.get('effective_k')==0:
                    formations.append(dict(common,**row))
            mechanism=read(root/method/'mechanisms.json')
            profiles.extend(dict(common,**r) for r in mechanism.get('group_profiles',[]))
            retention.extend(dict(common,**r) for r in mechanism.get('source_retention',[]))
            drift.append(dict(common,**mechanism.get('oof_refit_drift',{})))
    keyed=defaultdict(dict)
    for row in metrics:
        key=tuple(row.get(k,'') for k in ['cohort','dataset','method','role','training_stage','level','domain','task'])
        seed=int(row['seed'])
        if seed in keyed[key]:
            raise RuntimeError(f'Duplicate seed metric: {key}/{seed}')
        keyed[key][seed]=row
    aggregate=[]
    for key,entries in keyed.items():
        if set(entries)!=expected[key[0]]:
            raise RuntimeError(f'Incomplete four-seed metric condition: {key}')
        common=dict(zip(['cohort','dataset','method','role','training_stage','level','domain','task'],key))
        for metric in scalar_names:
            if all(row.get(metric,'') not in ('',None) for row in entries.values()):
                aggregate.append(dict(common,metric=metric,**stats([entries[s][metric] for s in sorted(entries)])))
    paired=[];paired_summary=[]
    references=['H_DIRECT','H_DIRECT_EFFORT','FULL_SOURCE_ALL','POOLED_TARGET','LOCAL_ACTIVE','K1_WIDE','SOURCE_INIT_RANDOM','PILOT_PROB','NATIVE','LOCAL','GROUP_CAL','FULL_TARGET','RANDOM_FEATURE']
    outer_roles={'Q_explore','validation','test_exposed','Q_legacy_confirm_exposed'}
    for key,entries in keyed.items():
        cohort,dataset,method,role,stage,level,domain,task=key
        if role not in outer_roles:
            continue
        if stage!='selected' and method!='NATIVE':
            continue
        for ref in references:
            other=keyed.get((cohort,dataset,ref,role,stage,level,domain,task))
            if ref==method or not other:
                continue
            if set(other)!=expected[cohort]:
                raise RuntimeError('Incomplete paired reference')
            common=dict(cohort=cohort,dataset=dataset,candidate=method,reference=ref,role=role,
                        training_stage=stage,level=level,domain=domain,task=task)
            for metric in scalar_names:
                if not all(entries[s].get(metric,'') not in ('',None) and other[s].get(metric,'') not in ('',None) for s in entries):
                    continue
                values=[float(entries[s][metric])-float(other[s][metric]) for s in sorted(entries)]
                paired_summary.append(dict(common,metric=metric,**stats(values)))
                paired.extend(dict(common,metric=metric,seed=s,delta=v) for s,v in zip(sorted(entries),values))
    # Worst cells are computed per seed before averaging, distinct from a worst mean cell.
    worst=defaultdict(list)
    for key,entries in keyed.items():
        if key[5]!='cell':
            continue
        for seed,row in entries.items():
            worst[key[:5]+(seed,)].append(row)
    worst_rows=[]
    for key,cells in worst.items():
        if len(cells)!=6:
            raise RuntimeError('Six cells are required for worst-cell statistics')
        cohort,dataset,method,role,stage,seed=key
        worst_rows.append(dict(cohort=cohort,dataset=dataset,method=method,role=role,training_stage=stage,
                               seed=seed,worst_cell_auc=min(float(c['auc']) for c in cells),
                               worst_cell_logloss=max(float(c['logloss']) for c in cells)))
    write('CORE_four_seed_per_seed_metrics.csv',metrics)
    write('CORE_four_seed_metric_summary.csv',aggregate)
    write('CORE_four_seed_paired_differences.csv',paired)
    write('CORE_four_seed_paired_summary.csv',paired_summary)
    q_pairs={('PILOT_PROB','NATIVE'),('PILOT_PROB','H_DIRECT'),('RCF','NATIVE'),
             ('RCF','H_DIRECT'),('RCF','POOLED_TARGET'),('RCF','LOCAL_ACTIVE')}
    q_domain_task=[]
    for row in paired_summary:
        if row['role']=='Q_explore' and row['level'] in ('domain','task') and \
           row['metric'] in ('auc','logloss') and (row['candidate'],row['reference']) in q_pairs:
            q_domain_task.append(dict(row,index=row[row['level']]))
    write('Q_four_seed_main_contrasts_domain_task.csv',q_domain_task)
    write('CORE_four_seed_group_metrics.csv',groups)
    write('CORE_four_seed_worst_cells.csv',worst_rows)
    write('CORE_four_seed_selections.csv',selections)
    write('CORE_four_seed_formation_constraints.csv',formations)
    write('CORE_four_seed_group_function_profiles.csv',profiles)
    write('CORE_four_seed_source_retention.csv',retention)
    write('CORE_four_seed_oof_refit_drift.csv',drift)
    write('CORE_four_seed_native_identity_cost.csv',native)
    write('CORE_four_seed_within_seed_bootstrap.csv',bootstraps)
    # Seed, not native-logit group or bootstrap replicate, is the repeat unit.
    group_keys=['cohort','dataset','method','role','domain','task','group']
    group_entries=defaultdict(dict)
    for row in groups:
        group_entries[tuple(row[k] for k in group_keys)][int(row['seed'])]=row
    group_summary=[]
    for key,entries in group_entries.items():
        if set(entries)!=expected[key[0]]:raise RuntimeError('Incomplete four-seed group condition')
        for metric in ('auc','logloss'):
            if all(row.get(metric,'') not in ('',None) for row in entries.values()):
                group_summary.append(dict(zip(group_keys,key),metric=metric,
                    **stats([entries[s][metric] for s in sorted(entries)])))
    write('CORE_four_seed_group_metric_summary.csv',group_summary)
    native_stage=[]
    for key,selected in keyed.items():
        if key[2]!='NATIVE' or key[4]!='selected':continue
        terminal=keyed.get(key[:4]+('terminal',)+key[5:])
        if terminal is None:raise RuntimeError('Actual native epoch24 endpoint missing')
        common=dict(zip(['cohort','dataset','method','role','training_stage','level','domain','task'],key))
        for metric in scalar_names:
            if all(selected[s].get(metric,'') not in ('',None) and terminal[s].get(metric,'') not in ('',None) for s in selected):
                values=[float(terminal[s][metric])-float(selected[s][metric]) for s in sorted(selected)]
                native_stage.append(dict(common,metric=metric,contrast='epoch24_minus_validation_selected',**stats(values)))
    write('CORE_four_seed_native_epoch24_minus_selected.csv',native_stage)
    receipt=dict(status='PASS',minimum_training_seeds=4,cohorts={k:sorted(v) for k,v in expected.items()},
                  n_prefix_exports=len(manifests),prefixes=manifests,metric_rows=len(metrics),
                  paired_summary_rows=len(paired_summary),frozen_manifest_sha256=sha(ACTIVE/'FROZEN_MANIFEST.json'),
                  seed_inventory_sha256=sha(ACTIVE/'FOUR_SEED_ACCEPTANCE.json'),
                  metric_provenance='exported metrics.csv from exact prediction arrays; export verifier recomputed every indexed prediction file',
                  interval='descriptive paired Student t, df3; no fold/task/bootstrap counted as seeds',
                  exposed_data=True)
    (REPORT/'CORE_FOUR_SEED_ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',prefixes=len(manifests),metric_rows=len(metrics),paired_summary_rows=len(paired_summary))))


if __name__=='__main__':
    main()
