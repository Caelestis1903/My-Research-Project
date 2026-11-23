"""Verify G_FROZEN control identity on original training draws and frozen Validation endpoints."""
import csv
import hashlib
import io
import json
import tarfile

import numpy as np
import pandas as pd
import torch

from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.rng import normal
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.protocol import view_key, derive_seed
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import row_ids_hash
from .final_evaluation import GAUSS, PUB, PUBBASE, require
from .history_stage import stopping
from .protocol import ART, DATASETS, M, P, PACKAGE, PLAN, REPORT, ROOT, SEEDS, digest, git, read, sha, v3, write

AIG = '6f5742c2085c1e49d1767b0a7079bb9f28160eda'
EMA02 = '38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27'
HISTORY = 'f9c2a8ef7b29b899e63b1beba823df1b6ea22c72'
AIG_REPORT = 'research_reports/2026-11-01_aig_m/'
EMA_REPORT = 'research_reports/2026-11-02_ema_view_design_v2/'
V3_REPORT = 'research_reports/2026-11-02_mdmt_information_momentum_v3/'
H_REPORT = 'research_reports/2026-11-03_mdmt_history_information/'
V3 = 'experiments/conditional_information_momentum_20261102_v3/'
KERNEL = 'vendor/ema02/vendor/gradient_aug_20261031_v2/'
ID_FIELDS = ('arm','method','seed','dataset','epoch','model_hash','config_hash','source_hash','trajectory_id',
             'selection_id','endpoint_id','independent_run_id','role','source_arm','reused','new_training',
             'source_path','source_sha256','protocol_id','data_protocol','checkpoint','estimator','split')


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def table(data):
    return csv.DictReader(io.StringIO(data.decode()))


def audit():
    require(not stopping(ART,'G_FROZEN'), 'Stop/pause control is active')
    trigger = read(ART/'G_FROZEN_TRIGGER.json')
    require(trigger['status']=='TRIGGERED' and trigger['A_DOSE_gate_sha256']==sha(ART/'tables/A_DOSE/decisions.json')
            and trigger['fixed_gaussian']==dict(domain=0,mix=.5,kappa=dict(movielens=.05,kuairand=.1))
            and trigger['seeds']==list(SEEDS) and trigger['test_used_for_selection'] is False, 'Trigger mismatch')
    sources = {}
    def blob(commit, path):
        data = git('show', f'{commit}:{path}')
        sources[f'{commit}:{path}'] = checksum(data)
        return data

    ml = json.loads(blob(PUBBASE, GAUSS+'MOVIELENS_SOURCE_AUDIT.json'))
    kr = json.loads(blob(PUBBASE, GAUSS+'RUN_MANIFEST.json'))
    protocol = v3.load_protocol()
    keys = ('raw_sha256','array_hashes','split_row_id_hashes','split_row_fingerprints','encoder_hash','scaler_hash',
            'train_domain_counts','split_rows','preprocessing_fit')
    current = {d:read(ART/f'{d}_data_identity.json') for d in DATASETS}
    for d, old in zip(DATASETS, (ml['data_identity'], kr['prepared']['data'])):
        require(all(current[d][k]==old[k] for k in keys), f'{d} historical population')
        require(digest(current[d])==read(ART/'DATA_IDENTITY.json')[d], f'{d} active population')
        cfg = ml['hyperparameters'][d]
        require(cfg['target_domains']==[0] and cfg['epochs']==50 and cfg['batch_size']==40960 and
                all(cfg[k]==v3.config('A1',d)[k] for k in ('rho','lr','weight_decay','lambda_aug','kappa')),
                f'{d} historical Gaussian parameters')

    for p in ('rng.py','objectives.py'):
        ref = blob(M, V3+'vendor/ema02/'+p)
        require(checksum(ref)==sha(v3.HERE/'vendor/ema02'/p) and
                ref==blob(EMA02, 'experiments/ema_view_design_20261102_v2/'+p), 'EMA02 numerical source')
    for p in ('objectives.py','protocol.py','views.py','vendor_ga31/model.py'):
        ref = blob(PUBBASE,'experiments/mdmt_history_information_20261103/vendor/v3/'+KERNEL+p)
        require(ref==blob(AIG,'experiments/gradient_aug_20261031_v2/'+p) and
                checksum(ref)==sha(v3.HERE/KERNEL/p), 'Original Gaussian primitive changed')
    for p in ('gradient_aug_aig_m_20261101/run.py','gradient_aug_aig_m_20261101/objectives.py',
              'gradient_aug_momentum_20261101/objectives.py'):
        blob(AIG,'experiments/'+p)
    for p in ('objectives.py','worker.py','config.py'):
        blob(PUBBASE,'experiments/gaussian_domain_coverage_20261104/'+p)

    # Each physical endpoint is identified by the original source, irrespective of later aliases.
    archived = {}
    for d,name,method in [('movielens',PUB,'GAUSSIAN_A1'),('kuairand',GAUSS+'ENDPOINTS.csv','G-D0')]:
        data = blob(PUBBASE,name)
        require(checksum(data)==(kr['prepared']['endpoint_table']['sha256'] if d=='movielens' else
                                 kr['endpoint_sha256']), 'Official source table checksum')
        rows = [r for r in table(data) if (r['dataset'],r['method'],r['split'],r['estimator'],r['checkpoint'])
                == (d,method,'validation','fixed_ema','selected')]
        require(sorted(int(r['seed']) for r in rows)==list(SEEDS), 'Incomplete or duplicate control')
        for r in rows:
            seed = int(r['seed'])
            require(r['trajectory_id']==r['independent_run_id'] and r['selection_id'] and r['endpoint_id']
                    and r['protocol_id']=='AUC_SELECTED_V1' and r['data_protocol']=='FULLTRAIN_NATURAL80_10_10',
                    'Historical selection identity')
            if d=='movielens':
                rec, = [a for a in ml['selection_audit'] if a['arm']=='G-D0' and a['seed']==seed]
                require(rec['validation_curve_epochs']==50 and int(r['epoch'])==rec['epoch'] and
                        all(rec[k]==r[k] for k in ('source_hash','model_hash','config_hash','selection_id','trajectory_id')),
                        'MovieLens selected control')
            else:
                job, = [j for j in kr['prepared']['jobs'] if j['arm']=='G-D0' and j['seed']==seed]
                require(all(job[k]==r[k] for k in ('config_hash','trajectory_id','source_hash')) and
                        job['config_hash']==digest(job['config']) and all(job['config'][k]==v for k,v in
                        ml['hyperparameters'][d].items()) and job['config']['betas']==[.9,.999], 'KuaiRand selected control')
            archived[d,seed]={k:r[k] for k in ID_FIELDS if k in r}
    curves = list(table(blob(PUBBASE, GAUSS+'VALIDATION_CURVES.csv')))
    for seed in SEEDS:
        curve = sorted((r for r in curves if (r['arm'],r['estimator'],int(r['seed']))==('G-D0','fixed_ema',seed)),
                       key=lambda r:int(r['epoch']))
        best = max(curve,key=lambda r:float(r['macro_auc']))
        require([int(r['epoch']) for r in curve]==list(range(1,51)) and
                all(best[k]==archived['kuairand',seed][k] for k in ('model_hash','epoch')), 'KR earliest Validation maximum')

    # Inventory all three row-keyed A1 result lines. The history table explicitly reuses earlier jobs.
    candidates = []
    available = {d:set() for d in DATASETS}
    selected = {}
    for commit,name in [(EMA02,EMA_REPORT+'tables/validation_metrics.csv'),
                        (P,V3_REPORT+'tables/endpoints.csv'),(HISTORY,H_REPORT+'endpoints.csv')]:
        for r in table(blob(commit,name)):
            if (r['arm'],r.get('split',r.get('role','validation')),r['checkpoint'],r.get('model_kind',r.get('estimator'))) != (
                    'A1','validation','selected','fixed_ema'):
                continue
            d,seed = r['dataset'],int(r['seed'])
            config = v3.config('A1',d)
            if r.get('config_json'):
                old=json.loads(r['config_json'])
                require(all(old[k]==config[k] for k in ('rho','kappa','lambda_aug','target_domain','batch_size')), 'A1 config')
            epoch,h=int(r['epoch']),r.get('model_hash',r.get('state_hash'))
            require(selected.setdefault((d,seed),(epoch,h))==(epoch,h),'Inconsistent A1 identities')
            matches = config['target_domain']==0
            if matches:
                available[d].add(seed)
            candidates.append(dict(commit=commit,path=name,dataset=d,seed=seed,source_key=r['key'],
                                   selected_epoch=epoch,model_hash=h,target_domain=config['target_domain'],
                                   historical_reference=r.get('historical_reference'),matches_d0=matches))
    require(sorted(available['movielens'])==[2022,2024] and not available['kuairand'],'Reassess available controls')
    five_path='research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv'
    aliases = [dict(dataset=r['dataset'],seed=int(r['seed']),model_hash=r['model_hash'],source_ref=r['source_result'])
               for r in table(blob(M,five_path)) if
               (r['arm'],r['split'],r['checkpoint'],r['model_kind'])==('Gaussian','validation','selected','fixed_ema')]
    require(len(aliases)==10 and all('gradient_aug_aig_m_20261101/attempt_20261101_1720/validation_metrics.csv' in r['source_ref'] for r in aliases),
            'Completion baseline source changed')

    # Compare the new prescribed row-keyed Gaussian directly to the original GPU draw receipts.
    archives = [(AIG,AIG_REPORT+'raw_attempts_2026-11-01_aig_m.tar.gz'),
                (PUBBASE,GAUSS+'raw_training_scalars.tar.gz')]
    traces = {}
    for d,(commit,name) in zip(DATASETS, archives):
        data=blob(commit,name)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for seed in SEEDS:
                if d=='movielens':
                    trial='S2/movielens/G1' if seed in (2022,2024) else 'S5/movielens/A1'
                    root=f'attempt_20261101_1720/runs/{trial}/seed{seed}/attempt1/'
                    paths=[root+'batch_trace.csv',root+'view_stats.csv']
                    b,v=[next(csv.DictReader(io.TextIOWrapper(archive.extractfile(p)))) for p in paths]
                    require(b['epoch']==v['epoch']=='1' and b['global_step']==v['global_step']=='1','ML trace position')
                    traces[d,seed]=dict(paths=paths,original=b['original_row_hash'],rows=int(v['target_rows']),
                                        key=v['view_key'],seed=int(v['view_seed']),noise=v['view_hash'])
                else:
                    path=f'execution/artifacts/gaussian_domain_coverage_20261104/attempt1/runs/kuairand.G-D0.s{seed}/launch_1/batch_trace.jsonl'
                    b=json.loads(archive.extractfile(path).readline())
                    v,=b['views']
                    require((b['epoch'],b['batch'],b['global_step'],v['domain'])==(1,1,1,0),'KR trace position')
                    traces[d,seed]=dict(paths=[path],original=b['original_row_hash'],rows=v['rows'],key=v['key'],
                                        seed=v['seed'],noise=v['noise_hash'],target_row_ids_hash=v['row_ids_hash'])

    samples = []
    for d,col in [('movielens','age'),('kuairand','tab')]:
        path = protocol['datasets'][d]['csv']
        require(sha(path)==current[d]['raw_sha256'], 'Raw source differs')
        groups = pd.read_csv(path,usecols=[col],dtype='int16')[col].to_numpy()
        train=groups[:int(.8*len(groups))]
        ids=np.arange(len(train),dtype=np.int64)
        if d=='kuairand':
            ids=ids[np.isin(train,(0,1,4))]
        require(row_ids_hash(ids)==current[d]['split_row_id_hashes']['train'], 'Training row population')
        fields=protocol['datasets'][d]['source']['perturbable_sparse_fields']
        require('domain_indicator' not in fields, 'Domain embedding in view')
        for seed in SEEDS:
            loader=torch.Generator().manual_seed(seed+11000)
            batch=ids[torch.randperm(len(ids),generator=loader)[:40960].numpy()]
            rows=batch[np.isin(train[batch],(1,18)) if d=='movielens' else train[batch]==0]
            ctx=dict(study='EMA02V2',dataset=d,split_id='original',seed=seed,epoch=1,global_step=1)
            noise=normal(ctx,rows,(len(fields),16),'A1|gaussian',torch.empty(0,dtype=torch.float32))
            observed=checksum(noise.numpy().tobytes())
            old=traces[d,seed]
            require(old['rows']==len(rows) and old['original']==(row_ids_hash(batch) if d=='movielens' else checksum(batch.tobytes()))
                    and old['key']==view_key(d,seed,'train',1,'gaussian',1) and old['seed']==derive_seed(old['key']),
                    'Different original first batch or frozen GA31 key')
            if d=='kuairand':
                require(old['target_row_ids_hash']==checksum(rows.tobytes()),'Different targeted original rows')
            require(old['noise']!=observed, 'Unexpected identical noise; reassess control')
            samples.append(dict(dataset=d,seed=seed,epoch=1,batch=1,rows=len(batch),target_rows=len(rows),
                                row_ids_sha256=checksum(batch.tobytes()),d0_row_ids_sha256=checksum(rows.tobytes()),
                                fields=fields,registered_noise_shape=list(noise.shape),registered_context=ctx,
                                registered_role='A1|gaussian',registered_noise_sha256=observed,
                                archived_first_draw=old,noise_equal=False))

    exact={d:sorted(ss) for d,ss in available.items()}
    return dict(stage='G_FROZEN',status='TECHNICAL_BLOCKED',author='coder_gaussian',trigger_sha256=sha(ART/'G_FROZEN_TRIGGER.json'),
                reason='The registered EMA02V2 row-keyed Gaussian differs from the GA31-R3 archived draw for all ten D0 controls. Only two MovieLens seeds and zero KuaiRand seeds have eligible row-keyed D0 selected controls.',
                exact_control_seeds=exact,missing_control_seeds={d:sorted(set(SEEDS)-available[d]) for d in DATASETS},
                design_sha256=sha(REPORT/'DESIGN_SPECIFICATION.md'),audit_code_sha256=sha(PACKAGE/'gaussian_control_audit.py'),
                pinned_sources=sources,verified_population_fields=keys,frozen_gaussian=ml['hyperparameters'],
                raw_population_sha256={d:current[d]['raw_sha256'] for d in DATASETS},
                registered_rng='EMA02V2 SplitMix64 / Box-Muller by original row and field coordinate',
                archived_rng='GA31-R3 device-local torch.Generator; per-field torch.randn in target batch order',
                matching_objective='One D0 view; original BCE + 0.5 * (view D0 BCE sum - original D0 BCE sum) / (2B). Detached per-field embedding RMS; domain excluded.',
                required_matrix='both datasets by all five seeds',registered_archival_controls=list(archived.values()),
                alternative_row_keyed_records=candidates,completion_aliases=aliases,
                first_batch_fixtures=samples,gaussian_scientific_jobs_dispatched=0,new_test_evaluations=0,
                original_runbook=sha(PLAN/'00_EXECUTION_RUNBOOK.md'),resource_rules=sha(PLAN/'08_RESOURCES_AND_SCHEDULER.md'))


if __name__=='__main__':
    receipt=audit()
    write(ART/'SOURCE_G_CONTROL_AUDIT.json',receipt)
    print(receipt['status'],receipt['exact_control_seeds'])
