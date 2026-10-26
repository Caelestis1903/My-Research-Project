"""Tabulate observed cost metadata only; no models, fitting, or timing calls."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np
from analyze_core_four_seed import require_complete_export_inventory

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
ACTIVE = ROOT/'artifacts/mdmt_rcf_20261026/v2_four_seed'
REQUIRED = {'NATIVE','B_REFIT','H_DIRECT','H_DIRECT_EFFORT','GROUP_CAL',
            'FULL_SOURCE_ALL','RCF','POOLED_TARGET','LOCAL_ACTIVE'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (REPORT/'tables'/name).open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=keys);out.writeheader()
        for row in rows:
            out.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                          for k,v in row.items()})


def main():
    four=read(ACTIVE/'FOUR_SEED_ACCEPTANCE.json')
    if four['status']!='PASS' or not four['analysis_authorized']:
        raise PermissionError('Four training seeds required before cost report')
    index=read(ACTIVE/'ARTIFACT_INDEX.json')
    require_complete_export_inventory(four,index)
    costs=[];parameters=[];timings=[];repetitions=[];active=[];candidates=[];provenance=[]
    widths={'GROUP_CAL':0,'PILOT_LOGIT':2,'FULL_SOURCE':64,'FULL_SOURCE_ALL':64,
            'FULL_TARGET':64,'RANDOM_FEATURE':64,'RCF':4,'POOLED_TARGET':4,
            'LOCAL_ACTIVE':4,'K1_WIDE':2,'SOURCE_INIT_RANDOM':4}
    for item in index['prefixes']:
        dataset=item['dataset'];seed=int(item['seed']);base=dict(dataset=dataset,seed=seed)
        path=ACTIVE/'delivery'/dataset/f'seed{seed}'/'COSTS_PARAMETERS_AND_INFERENCE.json'
        d=read(path)
        missing=REQUIRED-set(d['inference'])
        failed={m:d['inference'][m] for m in REQUIRED&set(d['inference'])
                if d['inference'][m]['status']!='MEASURED'}
        if missing or failed:
            raise RuntimeError(f'Required measured cost evidence incomplete: {dataset}/{seed}/{missing}/{failed}')
        if not d.get('immutability',{}).get('parameters_modes_gradients_normalizers_rng_unchanged'):
            raise RuntimeError('Measurement immutability receipt missing')
        provenance.append(dict(base,path=str(path),sha256=sha(path)))
        costs.extend(dict(base,**r) for r in d['costs'])
        for method,p in d['parameters'].items():
            common=dict(base,method=method)
            # The collector field is retained but explicitly not used to infer
            # discarded fits: its candidate dict may have been cleared.
            parameters.append(dict(common,**{k:v for k,v in p.items() if k!='per_target_task_active'},
                                   discarded_candidate_counter_authoritative=False))
            active.extend(dict(common,**r) for r in p.get('per_target_task_active',[]))
            selection=read(Path(item['root'])/method/'selection.json')['selection']
            history=selection.get('candidate_history',[])
            if method in widths:
                nonoff=[r for r in history if not r.get('off',False)]
                nvars=12*(2+widths[method])
                candidates.append(dict(common,actual_candidate_history=history,
                    fitted_nonoff_six_head_policies=len(nonoff),off_policy_candidates=sum(bool(r.get('off')) for r in history),
                    variables_per_fitted_policy=nvars,
                    fitted_variable_instances=len(nonoff)*nvars,
                    selected_policy=selection.get('selected'),
                    definition='2 groups x 6 cells x (alpha,beta,content coefficients); OFF evaluated without optimization; instances are not deployed capacity'))
        for method,timing in d['inference'].items():
            common=dict(base,method=method)
            timings.append(dict(common,**{k:v for k,v in timing.items()
                                          if k not in ('row_ids','repetitions','parameters')}))
            repetitions.extend(dict(common,repetition=i,seconds=x)
                               for i,x in enumerate(timing.get('repetitions',[])))
    # A condition requires all four unique seeds even for technical timings.
    keyed=defaultdict(dict)
    for row in timings:
        key=(row['dataset'],row['method'])
        if row['seed'] in keyed[key]:raise RuntimeError('Duplicate cost seed')
        keyed[key][row['seed']]=row
    summaries=[]
    for (dataset,method),entries in keyed.items():
        if set(entries)!={5201,5202,5203,5204}:raise RuntimeError('Incomplete four-seed cost condition')
        measured=[r for r in entries.values() if r['status']=='MEASURED']
        common=dict(dataset=dataset,method=method,n_training_seeds=4,n_measured_seeds=len(measured))
        if len(measured)==4:
            for metric in ('median_seconds','mean_seconds','rows_per_second','compile_seconds',
                           'peak_cuda_increment_over_resident'):
                values=np.asarray([r[metric] for r in measured],float)
                summaries.append(dict(common,metric=metric,mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),
                                      min=float(values.min()),max=float(values.max()),
                                      scope='seed-level variation of shared-device measurements; repetitions not independent seeds'))
        else:
            summaries.append(dict(common,metric='latency',status='INCOMPLETE_MEASUREMENT',
                                  reason='Do not average only successful seed measurements'))
    component_entries=defaultdict(dict)
    for row in costs:
        component_entries[row['dataset'],row['component']][int(row['seed'])]=row
    component_summary=[]
    for (dataset,component),entries in component_entries.items():
        if set(entries)!={5201,5202,5203,5204}:raise RuntimeError('Incomplete cost component seed condition')
        values=[r['seconds'] for r in entries.values()]
        row=dict(dataset=dataset,component=component,n_training_seeds=4,
                 scope=next(iter(entries.values()))['scope'])
        if all(v is not None for v in values):
            a=np.asarray(values,float)
            row.update(status='MEASURED_COMPONENT',mean_seconds=float(a.mean()),sample_sd_seconds=float(a.std(ddof=1)),
                       min_seconds=float(a.min()),max_seconds=float(a.max()))
        else:row.update(status='NOT_MEASURED',reason='Do not infer missing timings from steps')
        component_summary.append(row)
    for name,rows in [('COST_four_seed_training_components.csv',costs),
                      ('COST_four_seed_training_component_summary.csv',component_summary),
                      ('COST_four_seed_parameters.csv',parameters),
                      ('COST_four_seed_active_parameters.csv',active),
                      ('COST_four_seed_reader_candidates.csv',candidates),
                      ('COST_four_seed_inference.csv',timings),
                      ('COST_four_seed_inference_repetitions.csv',repetitions),
                      ('COST_four_seed_inference_summary.csv',summaries)]:write(name,rows)
    receipt=dict(status='PASS',n_prefixes=len(provenance),minimum_training_seeds=4,
                 required_measured_methods=sorted(REQUIRED),delivery_receipts=provenance,
                 inclusive_prefix_double_counted=False,
                 direct_family_split='UNAVAILABLE_DO_NOT_SPLIT',
                 aux_diagnostic_compute='not separately measured; do not claim zero cost',
                 model_files_written=0)
    (REPORT/'COST_FOUR_SEED_ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',prefixes=len(provenance),timing_rows=len(timings))))


if __name__=='__main__':main()
