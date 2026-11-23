"""S5 of the 2026-11-22 plan. Use freeze, evaluate and report in that order."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import resource
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

from . import results
from .protocol import (ART, DATASETS, M, MISSING, P, PACKAGE, REPORT, ROOT, SEEDS,
                       V3_MANIFEST, digest, git, jobs, read, sha, v3)
from .reuse import FIVE, SUP
from .state import generation, request

PUBBASE = 'c8675ae2923c15b6965884b140b5b646690cd95e'
PUB = 'research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv'
GAUSS = 'research_reports/2026-11-04_gaussian_domain_coverage/'
PUBLISHED = ('M3OE_FIXED1_ADAPTED', 'HYPERGATE', 'AUTOIFS')
STRONG = dict(zip(DATASETS, PUBLISHED[:2]))
BRANCHES = ('A_HISTORY', 'A_DOSE', 'A_SCOPE', 'B_SEARCH', 'AB_FROZEN', 'G_FROZEN')
B_METHODS = ('B_ADAPT', 'B_STATIC', 'B_UNIFORM')
A_METHODS = ('A20', 'A21', 'A22', results.DOSE[0], *results.SCOPE)
FINAL_FILES = ('final_evaluation.py', 'tests/test_final_evaluation.py', 'results.py',
               'state.py', 'protocol.py', 'reuse.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_blob(commit, name):
    data = git('show', f'{commit}:{name}')
    return data, hashlib.sha256(data).hexdigest()


def immutable(path, value):
    """Publish complete scalar records with link-if-absent. Never overwrite a winner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload=value, sha256=digest(value))
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(record, f, allow_nan=False, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                require(read(path) == record, f'Immutable record differs: {path}')
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temp.unlink(missing_ok=True)


def payload(path):
    record = read(path)
    require(digest(record['payload']) == record['sha256'], f'Damaged record {path}')
    return record['payload']


def approval(path, stage):
    receipt = read(path)
    require(receipt['stage'] == stage and receipt['verdict'] in ('PASS', 'PASS WITH FIXES')
            and receipt['blocking_issues'] == [], f'{stage} needs an independent PASS')
    return receipt


def checked_file(evidence, path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    require(expected is None or actual == expected, f'Source changed: {path}')
    evidence[str(path)] = actual
    return read(path) if path.suffix == '.json' else None


def blocked_gaussian(art, selection, decision, evidence):
    audit_path = art/'SOURCE_G_CONTROL_AUDIT.json'
    review_path = REPORT/'G_CONTROL_REVIEW_RECEIPT.json'
    bound = decision['evidence']
    require(bound.get(str(audit_path)) and bound.get(str(review_path)), 'Missing Gaussian control evidence')
    audit = checked_file(evidence, audit_path, bound[str(audit_path)])
    checked_file(evidence, review_path, bound[str(review_path)])
    review = approval(review_path, 'G_CONTROL')
    require(audit['stage']=='G_FROZEN' and audit['status']=='TECHNICAL_BLOCKED' and audit['reason'] and audit['trigger_sha256']
            and audit['author'] and review['reviewer'] and review['reviewer']!=audit['author']
            and review['audit_sha256']==sha(audit_path), 'Gaussian control audit/review mismatch')
    trigger = checked_file(evidence, art/'G_FROZEN_TRIGGER.json', audit['trigger_sha256'])
    coverage = audit['exact_control_seeds']
    require(set(coverage)==set(DATASETS) and all(set(seeds)<=set(SEEDS) for seeds in coverage.values())
            and any(set(seeds)!=set(SEEDS) for seeds in coverage.values()), 'Missing Gaussian control gap')
    require(trigger['status']=='TRIGGERED' and trigger['split']=='validation'
            and trigger['test_used_for_selection'] is False and trigger['unaugmented'] in A_METHODS
            and not decision.get('results') and all(cfg['candidate']==trigger['unaugmented']
            for cfg in selection.get('datasets',{}).values()), 'Blocked Gaussian selection must stay unaugmented')


def terminal_gates(art, selection, evidence):
    """Verify the completed core runs, raw B evidence and the registered conditional DAG."""
    conditional = selection['branches']
    core_gate_hashes = selection['core_gate_sha256']
    require(set(core_gate_hashes)=={'A_CORE',*DATASETS}, 'Selection must bind the A and both B gate revisions')
    accepted_a = {}
    require(set(conditional) == set(BRANCHES), 'All conditional branches must be resolved')
    results_by_job = {}
    anchors = defaultdict(set)
    for stage in ('A_CORE', 'B_CORE'):
        approved = approval(REPORT/f'{stage}_REVIEW_RECEIPT.json', stage)
        audit = checked_file(evidence, art/f'{stage}_AUDIT.json')
        checked_file(evidence, REPORT/f'{stage}_REVIEW_RECEIPT.json')
        check = checked_file(evidence, art/f'{stage}_CHECK.json', approved['check_sha256'])
        require(audit['status'] == 'SOURCE_READY' and check['status'] == 'PASS' and
                audit['source_hash'] == check['source_hash'] == approved['source_hash'] and
                approved['design_sha256'] == sha(REPORT/'DESIGN_SPECIFICATION.md'), 'Core review mismatch')
        checked_file(evidence, art/'REUSE_MANIFEST.json', audit['reuse_sha256'])
        run = checked_file(evidence, art/f'{stage}_RUN.json')
        require(run['status'] == 'COMPLETE' and not run['failed'] and not run['pending'], f'Incomplete {stage}')
        for job in jobs(stage, audit['source_hash']):
            directory = art/'attempt1/runs'/job['key']
            result = checked_file(evidence, directory/'result.json')
            ack = checked_file(evidence, directory/'COMPLETE_VERIFIED.json')
            require(result['job'] == job and result['epochs'] == 50 and
                    ack == dict(owner_generation=result['owner_generation'], result_hash=digest(result)),
                    f'Unverified {directory}')
            if stage == 'A_CORE':
                results_by_job[job['dataset'], job['arm'], job['seed']] = directory/'result.json'
            else:
                for tag, epochs in [('ANCHOR', (5,20,35,50)), ('WINDOW', (20,35))]:
                    for epoch in epochs:
                        raw = result['b_outputs'][f'B_{tag}_E{epoch:02}.json']
                        item = checked_file(evidence, raw['path'], raw['sha256'])
                        d = job['dataset']
                        if tag == 'ANCHOR':
                            ident = item['identity']
                            require((ident['dataset'],ident['seed'],ident['epoch'])==(d,job['seed'],epoch)
                                    and ident['actual_joint']==ident['virtual_joint'], 'Wrong B replay anchor')
                            anchors[d].add((str(Path(raw['path']).resolve()),raw['sha256']))
                        else:
                            steps = 20 if d=='movielens' else 49
                            require(item['K']==steps and item['start_step']==epoch*steps and
                                    len(item['row_hashes'])==steps, 'Incomplete parent window')
    previous = None
    for stage in results.STAGES:
        active = stage == 'A_CORE' or previous is not None and stage in previous['next_stages']
        if stage != 'A_CORE':
            require((conditional[stage]['status'] == 'COMPLETE') == active, f'{stage} trigger disagrees')
        if active:
            folder = art/'tables'/stage
            expected_gate = core_gate_hashes['A_CORE'] if stage=='A_CORE' else conditional[stage]['decision_sha256']
            gate = checked_file(evidence, folder/'decisions.json',expected_gate)
            require(gate['stage'] == stage and gate['status'] == 'COMPLETE' and
                    gate['seeds'] == list(SEEDS) and gate['split'] == 'validation' and
                    gate['readout'] == 'fixed_ema' and gate['checkpoint_rule'] == 'selected' and
                    not gate['missing_primary'] and set(gate['datasets']) == set(DATASETS) and
                    gate['statistics_script_sha256'] == sha(results.__file__), f'Incomplete {stage} decision')
            require(set(gate['evidence']) >= {'endpoints.csv','paired_seed.csv','paired_summary.csv'},
                    f'Missing {stage} tables')
            if previous:
                require(gate['previous_decision_sha256'] == digest(previous), 'Broken A decision chain')
            for name, expected in gate['evidence'].items():
                checked_file(evidence, folder/name, expected)
            with (folder/'endpoints.csv').open(newline='') as stream:
                for row in csv.DictReader(stream):
                    if (row['split'],row['readout'],row['checkpoint_rule'],row['cell'])==(
                            'validation','fixed_ema','selected','macro'):
                        k = (row['dataset'],row['arm'],int(row['seed']))
                        ident = (row['trajectory_id'],row['model_hash'],int(row['epoch']))
                        require(accepted_a.setdefault(k,ident)==ident, 'Conflicting selected A gate identities')
            previous = gate
    search = False
    for d in DATASETS:
        folder = art/'tables/B_CORE'
        gate = checked_file(evidence, folder/f'B_GATE_{d}.json',core_gate_hashes[d])
        require(gate['dataset'] == d and len(gate['anchors']) == 20 and
                {(r['seed'],r['epoch']) for r in gate['all_choices']} ==
                {(s,e) for s in SEEDS for e in (5,20,35,50)}, 'Incomplete B gate')
        require({(str(Path(x['path']).resolve()),x['sha256']) for x in gate['anchors']}==anchors[d],
                'B gate uses a different parent replay')
        all_inputs = []
        require(gate['B_PULSE'] in ('TRIGGERED', 'NOT_TRIGGERED'), 'Unresolved pulse')
        if gate['B_PULSE'] == 'TRIGGERED':
            require(len(gate['pulse_inputs']) == 20 and gate['B_SEARCH'] in
                    ('TRIGGERED','MECHANISM_WITHOUT_TRAIN_SIGNAL','FIXED_EMA_ALREADY_SUFFICIENT_FOR_THIS_ACTION',
                     'MACRO_GAIN_WITH_CELL_TRADEOFF','LOW_ACTION_VALUE'), 'Unresolved pulse evidence')
            all_inputs.extend(gate['pulse_inputs'])
        else:
            require(gate['B_SEARCH'] == 'NOT_TRIGGERED', 'Unexpected search')
        search |= gate['B_SEARCH'] == 'TRIGGERED'
        static = checked_file(evidence, folder/f'B_STATIC_LOSO_{d}.json')
        require(digest(static) == gate['static_hash'], 'LOSO freeze mismatch')
        slots=set()
        for item in all_inputs:
            pulse=checked_file(evidence,item['path'],item['sha256'])
            require(pulse['dataset']==d,'Wrong pulse dataset')
            slots.add((pulse['seed'],pulse['epoch'],pulse['family']))
        if all_inputs:
            require(slots=={(s,e,f) for s in SEEDS for e in (20,35) for f in ('true','pseudo0')},
                    'Missing pulse slots')
    require((conditional['B_SEARCH']['status'] == 'COMPLETE') == search, 'B search trigger disagrees')
    for stage, decision in conditional.items():
        require(decision['reason'] and decision['evidence'], f'Unresolved conditional stage {stage}')
        if stage=='G_FROZEN' and decision['status']=='NOT_TRIGGERED' and (art/'G_FROZEN_TRIGGER.json').exists():
            trigger=checked_file(evidence, art/'G_FROZEN_TRIGGER.json')
            require(trigger['status']!='TRIGGERED', 'Gaussian branch was triggered')
        if stage=='G_FROZEN' and decision['status']=='TECHNICAL_BLOCKED':
            blocked_gaussian(art, selection, decision, evidence)
        else:
            require(decision['status'] in ('COMPLETE','NOT_TRIGGERED'), f'Unresolved conditional stage {stage}')
        for name, expected in decision['evidence'].items():
            checked_file(evidence, name, expected)
        if decision['status'] == 'COMPLETE' and stage in ('B_SEARCH','AB_FROZEN','G_FROZEN'):
            # S4 hands over all accepted trajectories, including unselected eta values.
            expected_arms = B_METHODS if stage == 'B_SEARCH' else (stage,)
            required = {(d,a,s,eta) for d in DATASETS for a in expected_arms for s in SEEDS
                        for eta in ((.25,.5) if stage == 'B_SEARCH' else (None,))}
            seen = set()
            for name in decision['results']:
                result = checked_file(evidence, name)
                ack = checked_file(evidence, Path(name).with_name('COMPLETE_VERIFIED.json'))
                job = result['job']
                ident = (job['dataset'],job['arm'],job['seed'],job['config']['eta'] if stage=='B_SEARCH' else None)
                require(ident not in seen and result['epochs']==50 and result['cost_class']=='NEW_SCIENTIFIC_TRAIN'
                        and ack['result_hash']==digest(result) and ack['owner_generation']==result['owner_generation'],
                        'Unverified or duplicate conditional trajectory')
                seen.add(ident)
            require(seen == required, f'Incomplete conditional matrix {stage}')
    if conditional['AB_FROZEN']['status']=='COMPLETE':
        require(search and conditional['A_HISTORY']['status']=='COMPLETE','Missing joint parent stage')
    return results_by_job,accepted_a


def historical(art, split, gaussian=False):
    """Read immutable Git blobs. Join archived Test to frozen Validation by selected-state identity."""
    pop_fields = ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes',
                  'encoder_hash','scaler_hash','split_rows','preprocessing_fit')
    populations = read(art/'DATA_IDENTITY.json')
    kr = json.loads(pinned_blob(PUBBASE,GAUSS+'RUN_MANIFEST.json')[0])
    ml = json.loads(pinned_blob(PUBBASE,GAUSS+'MOVIELENS_SOURCE_AUDIT.json')[0])
    for d, old in zip(DATASETS,(ml['data_identity'],kr['prepared']['data'])):
        now = read(art/f'{d}_data_identity.json')
        require(digest(now)==populations[d] and all(now[k]==old[k] for k in pop_fields),
                f'Historical population identity mismatch for {d}')
    files = [(M,str(FIVE.relative_to(ROOT))), (P,str((SUP/'endpoints.csv').relative_to(ROOT))), (PUBBASE,PUB)]
    if gaussian:
        files.append((PUBBASE,GAUSS+'ENDPOINTS.csv'))
    rows = []
    for commit,name in files:
        blob, file_hash = pinned_blob(commit,name)
        if name == PUB:
            require(file_hash==kr['prepared']['endpoint_table']['sha256'], 'Published table audit mismatch')
        for r in csv.DictReader(io.StringIO(blob.decode())):
            if r['checkpoint'] != 'selected' or r.get('split', r.get('role')) != split:
                continue
            d, seed = r['dataset'], int(r['seed'])
            if name == PUB or name.startswith(GAUSS):
                arm, kind = r['method'], r['estimator']
                if name == PUB and arm in PUBLISHED and kind == 'online':
                    require(r['role']=='PUBLISHED_BASELINE', 'Published role mismatch')
                elif gaussian and kind == 'fixed_ema' and (
                        (d=='movielens' and name==PUB and arm=='GAUSSIAN_A1') or
                        (d=='kuairand' and name.startswith(GAUSS) and arm=='G-D0')):
                    arm='G-D0'
                else:
                    continue
                require((r['protocol_id'],r['data_protocol'])==('AUC_SELECTED_V1','FULLTRAIN_NATURAL80_10_10')
                        and r['selection_id'] and r['config_hash'] and r['source_hash'], 'Historical protocol mismatch')
                trajectory, h, config = r['trajectory_id'],r['model_hash'],r['config_hash']
            else:
                allowed = {'A0':'A00','P01':'A20'} if commit==M else {'P00':'A10','P10':'A30'}
                if r['arm'] not in allowed or (commit==P and seed in MISSING):
                    continue
                arm, kind = allowed[r['arm']], r.get('model_kind',r.get('estimator'))
                if kind != 'fixed_ema':
                    continue
                trajectory, h = r['key'],r.get('model_hash',r.get('state_hash'))
                config = r.get('configuration') or digest(v3.config(r['arm'],d,.05))
            scores = {k:v for k,v in r.items() if v and k.endswith(
                ('_auc','_logloss','_count','_positive','_negative'))}
            rows.append(dict(dataset=d,arm=arm,seed=seed,readout=kind,split=split,checkpoint_rule='selected',
                trajectory_id=trajectory,model_hash=h,epoch=int(r['epoch']),config_hash=config,
                source_hash=r.get('source_hash') or file_hash,selection_id=r.get('selection_id') or trajectory,
                source_ref=commit+':'+name,source_file_sha256=file_hash,data_hash=populations[d],
                reuse_status='REUSE_NUMERIC',test_exposure='HISTORICAL_EXPOSED',**scores))
    return rows


def state_id(row):
    return {k:row[k] for k in ('dataset','arm','seed','readout','epoch','model_hash','trajectory_id',
                                'config_hash','data_hash','source_hash')}


def accepted_result(path, d, arm, seed):
    r = read(path)
    job = r['job']
    require((job['dataset'],job['arm'],job['seed']) == (d,arm,seed) and job['key'] == path.parent.name and
            path.parent.parent.name=='runs', 'Result job mismatch')
    row, = [x for x in results.completed_rows(path,r) if x['checkpoint_rule']=='selected' and x['readout']=='fixed_ema']
    require((row['dataset'],row['arm'],row['seed'],row['trajectory_id'],row['source_hash'],row['config_hash']) ==
            (d,arm,seed,job['key'],job['source_hash'],digest(job['config'])), 'Selected provenance mismatch')
    # Read the accepted owner generation's per-epoch Validation scalars, including resumed attempts.
    curve = {}
    for f in path.parent.glob(r['owner_generation'][:10]+'-attempt*/epoch_*.json'):
        rec = read(f)
        epoch = rec['epoch']
        require(epoch not in curve and rec['ram_state_hash'] ==
                read(f.with_name(f'RAM_{epoch:02}.json'))['state_hash'], 'Duplicate/unverified Validation epoch')
        score = rec['validation']['fixed_ema']
        curve[epoch] = score
    require(set(curve) == set(range(1,51)), 'Incomplete Validation curve')
    winner = max(range(1,51), key=lambda e: results.finite(curve[e]['macro_auc']))
    require(winner==row['epoch'] and results.metrics(row)==results.metrics(curve[winner]),
            'Selected state differs from earliest Validation maximum')
    return dict(row,config=job['config'],source_ref=str(path.resolve()),result_hash=digest(r))


def selected_ram(row):
    path = Path(row['source_ref'])
    attempt = path.parents[2]
    owner = generation(attempt)
    require(owner == row['owner_generation'], 'Selected RAM owner generation changed')
    reply = request(attempt,'get',key=row['trajectory_id'])
    state, r = reply['state'], reply['result']
    require(state is not None and r is not None and digest(r)==row['result_hash'] and r==read(path),
            'Selected RAM transaction differs from accepted result')
    kind=row['readout']
    # The November 16 guardian deliberately compacts completed packets to three selected fields.
    # Its current job registry and immutable accepted result retain the transaction identities.
    registry=request(attempt,'status')['jobs'][row['trajectory_id']]
    require(registry['status']=='COMPLETE' and registry['epoch']==50 and registry['job']==r['job']
            and registry['result']==r and r['owner_generation']==owner and r['epochs']==50 and
            r['data_hash']==row['data_hash'] and r['group_hash']==row['group_hash'] and
            r['attempt_id']==row['physical_attempt'] and state['selected_epochs'][kind]==row['epoch'] and
            results.metrics(state['selected_validation'][kind])==results.metrics(row),
            'Selected RAM identity changed')
    weights=state['selected_models'][kind]
    require(v3.state_digest(weights)==row['model_hash']==r['selected_hashes'][kind], 'Selected RAM hash changed')
    return weights


def matrix(selection, d):
    cfg = selection['datasets'][d]
    candidate = cfg['candidate']
    arms = list(dict.fromkeys((*results.CORE,*PUBLISHED,candidate)))
    pairs = list(results.comparisons('A_CORE',None,d))
    def compare(a,b):
        require(a and b and (a in (*A_METHODS,*B_METHODS,*results.CORE,'AB_FROZEN','G_FROZEN','G-D0') or a in PUBLISHED)
                and b in (*A_METHODS,*B_METHODS,*results.CORE,*results.DOSE,*PUBLISHED,'G-D0','AB_FROZEN'),
                'Unknown comparison arm')
        for arm in (a,b):
            if arm not in arms:
                arms.append(arm)
        if a!=b and (a,b) not in pairs:
            pairs.append((a,b))
    require(candidate in (*A_METHODS,*B_METHODS,'AB_FROZEN','G_FROZEN') and cfg['parameters'], 'Missing selected method')
    require(cfg['strong_internal']==('G-D0' if candidate=='G_FROZEN' else 'A00'), 'Wrong internal guard')
    require(cfg['strongest_published']==STRONG[d], 'Wrong strongest published model')
    compare(candidate, STRONG[d])
    compare(candidate, cfg['strong_internal'])
    base = cfg.get('unaugmented') if candidate=='G_FROZEN' else candidate
    if candidate=='G_FROZEN':
        require(selection['branches']['G_FROZEN']['status']=='COMPLETE', 'Gaussian branch closed')
        compare(candidate,base)
    components = [cfg['a_single'],cfg['b_single']] if base=='AB_FROZEN' else [base]
    if base=='AB_FROZEN':
        require(selection['branches']['AB_FROZEN']['status']=='COMPLETE' and
                components[0] in A_METHODS and components[1] in B_METHODS, 'Joint branch/constituents mismatch')
        for a in components:
            compare(base,a)
        require(cfg['stronger_single'] in components,'Freeze the stronger Validation single')
    for a in components:
        if a in A_METHODS:
            require(selection['branches']['A_HISTORY']['status']=='COMPLETE','History controls missing')
            lam=cfg['parameters']['lambda_pair']
            require(lam in (.025,.05) and cfg['matched_adam']==('A41_ADAM_L025' if lam==.025 else 'A10'),
                    'Mismatched supervision dose')
            compare(a,cfg['matched_adam'])
            for control in ('A21','A22'):
                compare('A20',control)
                if lam==.05:
                    compare(a,control)
            require((a not in results.HISTORIES or lam==.05) and
                    (a!=results.DOSE[0] or lam==.025),'Wrong history arm for selected dose')
            if lam==.025:
                require(selection['branches']['A_DOSE']['status']=='COMPLETE','Dose stage closed')
                compare(*results.DOSE)
            if a in results.SCOPE:
                require(selection['branches']['A_SCOPE']['status']=='COMPLETE' and
                        cfg['scope']==('STAR_SKIP' if a==results.SCOPE[0] else 'EXPERT'), 'Scope stage closed or wrong scope')
                compare(a,cfg['full_history'])
        else:
            require(a in B_METHODS and selection['branches']['B_SEARCH']['status']=='COMPLETE'
                    and cfg['parameters']['eta'] in (.25,.5), 'B search/eta mismatch')
            for method in B_METHODS:
                compare(method,'A00')
            compare('B_ADAPT','B_STATIC')
            compare('B_ADAPT','B_UNIFORM')
            compare('B_STATIC','B_UNIFORM')
    return arms,pairs


def authority(art):
    evidence = {}
    selection = checked_file(evidence, art/'S5_SELECTION.json')
    path = REPORT/'S5_REVIEW_RECEIPT.json'
    code = approval(path,'S5')
    checked_file(evidence,path)
    for name in FINAL_FILES:
        checked_file(evidence,PACKAGE/name,code['files'][name])
    checked_file(evidence,REPORT/'DESIGN_SPECIFICATION.md',code['design_sha256'])
    checked_file(evidence,REPORT/'CODE_IMPLEMENTATION_REPORT_FINAL.md',code['report_sha256'])
    path = REPORT/'S5_SELECTION_REVIEW.json'
    receipt = approval(path,'S5_SELECTION')
    checked_file(evidence,path)
    require(receipt['selection_sha256']==sha(art/'S5_SELECTION.json') and
            receipt['code_review_sha256']==sha(REPORT/'S5_REVIEW_RECEIPT.json') and
            receipt['reviewer'] and receipt['reviewer']!=selection['author'] and
            Path(selection['artifacts_root']).resolve()==art and selection['split']=='validation' and selection['test_used_for_selection'] is False and
            selection['status'] in ('FINAL_CANDIDATE','NO_CANDIDATE'), 'Selection approval mismatch')
    require(sha(V3_MANIFEST)=='7034a3bb59e6d08e3296183258e0e7b16ab74d0248ee0a3b75c526a680462cc0',
            'Unexpected v3 source manifest')
    for row in read(V3_MANIFEST)['files']:
        require(sha(v3.HERE/row['path'])==row['sha256'], 'Frozen v3 source differs')
    checked_file(evidence,V3_MANIFEST)
    return selection,evidence


def freeze(art):
    selection,evidence = authority(art)
    core_results,accepted_a = terminal_gates(art,selection,evidence)
    checked_file(evidence,art/'DATA_IDENTITY.json')
    for d in DATASETS:
        checked_file(evidence,art/f'{d}_data_identity.json')
    packet = dict(status=selection['status'],selection=selection,selected=[],comparisons={},evidence=evidence,
                  checkpoint_rule='earliest strict full Validation Macro AUC maximum E1-E50')
    if selection['status']=='FINAL_CANDIDATE':
        require(set(selection['datasets'])==set(DATASETS), 'Both datasets are required')
        gauss=any(cfg['candidate']=='G_FROZEN' for cfg in selection['datasets'].values())
        archive=results.index_rows(historical(art,'validation',gauss))
        audited=results.index_rows(results.audited_reuse(read(art/'REUSE_MANIFEST.json')))
        forms=[]
        for d in DATASETS:
            cfg=selection['datasets'][d]
            arms,pairs=matrix(selection,d)
            packet['comparisons'][d]=[list(p) for p in pairs]
            def form(arm):
                return cfg['history_form'] if arm in (results.DOSE[0],*results.SCOPE) else arm
            forms.append((form(cfg['candidate']),form(cfg.get('unaugmented')),form(cfg.get('a_single')),
                          cfg.get('b_single'),cfg.get('scope','S_DENSE')))
            for arm in arms:
                kind='online' if arm in PUBLISHED else 'fixed_ema'
                for seed in SEEDS:
                    key=(d,arm,seed,kind,'selected')
                    if key in archive:
                        row=archive[key][0]
                        if arm in results.CORE:
                            ref,values=audited[key]
                            row=dict(row,config=ref['config'])
                            if arm==cfg['candidate']:
                                require(all(row['config'].get(k)==v for k,v in cfg['parameters'].items()),
                                        'Historical method parameters differ from S0')
                            require((ref['epoch'],ref['model_hash'],ref['trajectory_id'])==
                                    (row['epoch'],row['model_hash'],row['trajectory_id']) and
                                    values==results.metrics(row), 'Archive selection differs from S0')
                    else:
                        path=core_results.get((d,arm,seed))
                        if path is None:
                            path=Path(cfg['result_paths'][arm][str(seed)]).resolve()
                            require(path.is_relative_to(art) and path.name=='result.json',
                                    'Selected result must be a registered local trajectory')
                        if arm in (*B_METHODS,'AB_FROZEN','G_FROZEN'):
                            stage='B_SEARCH' if arm in B_METHODS else arm
                            require(str(path.resolve()) in {str(Path(x).resolve()) for x in
                                    selection['branches'][stage]['results']}, 'Selected job absent from accepted S4 matrix')
                        row=accepted_result(path,d,arm,seed)
                        require(row['data_hash']==read(art/'DATA_IDENTITY.json')[d], 'Training data identity changed')
                        if arm in B_METHODS:
                            require(row['config']['eta']==cfg['parameters']['eta'], 'B control eta mismatch')
                        params=cfg['parameters'] if arm==cfg['candidate'] else {}
                        require(all(k in row['config'] and row['config'][k]==v for k,v in params.items()),
                                'Declared selected parameters differ from the training job')
                        if arm in (*A_METHODS,*results.DOSE,'A10','A30'):
                            expected_lambda=(.025 if arm in results.DOSE else cfg['parameters']['lambda_pair']
                                             if arm in results.SCOPE else .05)
                            require(row['config']['lambda_pair']==expected_lambda, 'Incorrect PAIR dose')
                        selected_ram(row)
                        checked_file(evidence,path)
                        checked_file(evidence,path.with_name('COMPLETE_VERIFIED.json'))
                    if arm in (*results.CORE,*A_METHODS,*results.DOSE):
                        require(accepted_a[d,arm,seed]==(row['trajectory_id'],row['model_hash'],row['epoch']),
                                'Frozen A state differs from the accepted Validation decision')
                    packet['selected'].append(row)
        require(forms[0]==forms[1], 'A common final method and scope are required across datasets')
        # Validate all mixed-source Validation supports before any new Test is read.
        validated=results.index_rows(packet['selected'])
        eligible=[]
        for d in DATASETS:
            cfg=selection['datasets'][d]
            cand=cfg['candidate']
            def gain(left,right):
                return statistics.fmean(validated[d,left,s,*results.PRIMARY][1]['macro','auc']-
                                        validated[d,right,s,*results.PRIMARY][1]['macro','auc'] for s in SEEDS)
            base=cfg.get('unaugmented',cand)
            if base=='AB_FROZEN':
                single=cfg['stronger_single']
                other=cfg['a_single'] if single==cfg['b_single'] else cfg['b_single']
                require(gain(single,other)>=0,'Wrong stronger single-method comparison')
            control=(cfg['matched_adam'] if base in A_METHODS else cfg['stronger_single']
                     if base=='AB_FROZEN' else 'A00')
            eligible.append(gain(cand,cfg['strong_internal'])>0 and gain(base,'A00')>0 and gain(base,control)>0)
        require(any(eligible),'No target dataset has an independent Validation gain')
        a_gate = read(art/'tables/A_HISTORY/decisions.json') if selection['branches']['A_HISTORY']['status']=='COMPLETE' else {}
        for d,cfg in selection['datasets'].items():
            base=cfg.get('unaugmented',cfg['candidate'])
            arm=cfg['a_single'] if base=='AB_FROZEN' else base
            if arm in A_METHODS:
                form=cfg.get('history_form',arm)
                require(form==a_gate.get('history_form') and (arm not in results.HISTORIES or arm==form),
                        'Final history form differs from the A target freeze')
                if selection['branches']['A_DOSE']['status']=='COMPLETE':
                    dose=read(art/'tables/A_DOSE/decisions.json')['datasets'][d]
                    require(cfg['parameters']['lambda_pair']==dose['lambda_pair'] and
                            cfg['matched_adam']==dose['matched_adam'], 'Final dose differs from the A decision')
                if selection['branches']['A_SCOPE']['status']=='COMPLETE':
                    scope=read(art/'tables/A_SCOPE/decisions.json')['scope']
                    require(scope is not None and scope==cfg.get('scope','S_DENSE'), 'Final scope differs from A gate')
    else:
        require(selection.get('reason') and not selection.get('datasets'), 'Explain no-candidate closure')
    immutable(art/'FINAL_FREEZE.json',packet)
    return packet


def frozen(art):
    selection,current=authority(art)
    packet=payload(art/'FINAL_FREEZE.json')
    require(selection==packet['selection'] and all(packet['evidence'].get(k)==v for k,v in current.items()),
            'Review/selection differs from FINAL_FREEZE')
    for name,h in packet['evidence'].items():
        require(sha(name)==h, f'Frozen evidence changed: {name}')
    return packet


def once(folder, freeze_hash, row, compute):
    """A crash after intent and before a durable scalar outcome leaves a closed, indeterminate slot."""
    identity=state_id(row)
    key=digest(identity)
    done=folder/(key+'.json')
    started=folder/(key+'.started.json')
    expected=dict(freeze_sha256=freeze_hash,identity=identity)
    if done.exists():
        rec=payload(done)
        require(all(rec[k]==v for k,v in expected.items()), 'Scalar receipt identity mismatch')
        return rec
    require(not started.exists(), f'Indeterminate prior Test access; restore the scalar receipt for {key}')
    def mark_scoring():
        immutable(started,expected)
    scores,cost=compute(mark_scoring)
    require(started.exists() or row['reuse_status']=='REUSE_NUMERIC', 'Missing Test access intent')
    results.metrics(scores)
    rec=dict(**expected,scores=scores,cost=cost)
    immutable(done,rec)
    return rec


def evaluate(art,device):
    packet=frozen(art)
    require(packet['status']=='FINAL_CANDIDATE', 'No candidate eligible for new Test')
    folder=art/'S5'
    folder.mkdir(exist_ok=True)
    with (folder/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h=digest(packet)
        if (folder/'EVALUATED.json').exists():
            return receipt_index(folder,packet)
        selected=packet['selected']
        archive=historical(art,'test',any(r['arm']=='G-D0' for r in selected))
        lookup={(r['dataset'],r['arm'],r['seed'],r['readout']):r for r in archive}
        require(len(lookup)==len(archive),'Duplicate historical Test endpoint')
        receipts={}
        def source_row(row):
            r=lookup[row['dataset'],row['arm'],row['seed'],row['readout']]
            require(state_id(r)==state_id(row) and r['selection_id']==row['selection_id']
                    and r['source_file_sha256']==row['source_file_sha256'],'Archive Validation/Test identity mismatch')
            results.metrics(r)
            return r
        archived_support={}
        # Check every unconsumed RAM/archive identity before opening the first new population.
        for row in selected:
            if row['reuse_status']=='REUSE_NUMERIC':
                src=source_row(row)
                support=tuple((float(src[c+'_count']),float(src[c+'_positive'])) for c in results.CELLS)
                require(archived_support.setdefault(row['dataset'],support)==support, 'Archived Test populations differ')
            elif not (folder/(digest(state_id(row))+'.json')).exists():
                selected_ram(row)
        for d in DATASETS:
            bundle=model=None
            for row in (r for r in selected if r['dataset']==d):
                require(not (art/'USER_STOPPED').exists() and not (art/'S5_PAUSE').exists(),'S5 interrupted')
                def score(mark_scoring):
                    nonlocal bundle,model
                    start=time.monotonic()
                    if row['reuse_status']=='REUSE_NUMERIC':
                        r=source_row(row)
                        scores={k:v for k,v in r.items() if k.endswith(('_auc','_logloss','_count','_positive','_negative'))}
                        return scores,dict(cost_role='REUSE_NUMERIC',test_evaluations=0,forward_rows=0,
                                           forwards=0,worker_seconds=0,wall_seconds=time.monotonic()-start)
                    weights=selected_ram(row)
                    protocol=v3.load_protocol()
                    if bundle is None:
                        bundle,identity=v3.load_dataset(d,p=protocol)
                        require(digest(identity)==row['data_hash'],'Evaluation data identity mismatch')
                    v3.setup(row['seed'],device)
                    if model is None:
                        model=v3.make_model(bundle,protocol['datasets'][d]['source']).to(device)
                    model.load_state_dict(weights,strict=True)
                    require(v3.state_digest(model.state_dict())==row['model_hash'], 'Loaded weights differ')
                    if device.type=='cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    mark_scoring()
                    scores,c=v3.evaluate(model,bundle.test_x,bundle.test_y,device)
                    return scores,dict(cost_role='FINAL_TEST_EVALUATION',test_evaluations=1,
                        forwards=c['forwards'],forward_rows=c['rows'],worker_seconds=time.monotonic()-start,
                        evaluator_seconds=c['seconds'],selected_ram_bytes=sum(v.numel()*v.element_size()
                        for v in weights.values()),
                        wall_seconds=time.monotonic()-start,peak_allocated=(torch.cuda.max_memory_allocated(device)
                        if device.type=='cuda' else None),peak_reserved=(torch.cuda.max_memory_reserved(device)
                        if device.type=='cuda' else None),host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                rec=once(folder,h,row,score)
                key=digest(state_id(row))
                receipts[key]=digest(rec)
            del bundle,model
        index=dict(freeze_sha256=h,receipts=receipts)
        immutable(folder/'EVALUATED.json',index)
        return index


def receipt_index(folder,packet):
    h=digest(packet)
    idx=payload(folder/'EVALUATED.json')
    require(idx['freeze_sha256']==h and set(idx['receipts'])=={digest(state_id(r)) for r in packet['selected']},
            'Incomplete Test matrix')
    for row in packet['selected']:
        identity=state_id(row)
        k=digest(identity)
        rec=payload(folder/(k+'.json'))
        require(rec['freeze_sha256']==h and rec['identity']==identity and digest(rec)==idx['receipts'][k],
                'Final scalar receipt changed')
    return idx


def summarize(packet, folder):
    endpoints,absolute,pairs,summaries,costs,interactions,main=[],[],[],[],[],[],[]
    data=defaultdict(dict)
    freeze_hash=digest(packet)
    populations={}
    source_rows={}
    for row in packet['selected']:
        source_rows[row['dataset'],row['arm'],row['seed']]=row
        identity=state_id(row)
        key=digest(identity)
        rec=payload(folder/(key+'.json'))
        require(rec['identity']==identity and rec['freeze_sha256']==freeze_hash, 'Unbound endpoint')
        exposure='HISTORICAL_EXPOSED' if row['reuse_status']=='REUSE_NUMERIC' else 'FROZEN_HISTORICAL_POPULATION'
        costs.append(dict(**identity,receipt=key,**rec['cost'],main_steps=0,virtual_steps=0,pulse_steps=0))
        for split,wide in [('validation',row),('test',rec['scores'])]:
            pop=tuple((int(float(wide[c+'_count'])), int(float(wide[c+'_positive']))) for c in results.CELLS)
            require(populations.setdefault((split,row['dataset']),pop)==pop, 'Endpoint population mismatch')
            values=results.metrics(wide)
            data[split,row['dataset'],row['arm']][row['seed']]=values
            for cell in dict.fromkeys(c for c,_ in values):
                cfg=packet['selection']['datasets'][row['dataset']]
                roles=[k for k in ('candidate','strong_internal','strongest_published','matched_adam','a_single','b_single')
                       if cfg.get(k)==row['arm']]
                endpoints.append(dict(**identity,split=split,cell=cell,auc=values.get((cell,'auc')),
                    logloss=values.get((cell,'logloss')),support=wide.get(cell+'_count'),positive=wide.get(cell+'_positive'),
                    source_ref=row['source_ref'],source_file_sha256=row.get('source_file_sha256'),receipt=key,
                    role='|'.join(roles),family='published' if row['arm'] in PUBLISHED else
                    ('candidate' if 'candidate' in roles else 'internal'),checkpoint_rule='selected',
                    reuse_status=row['reuse_status'],test_exposure=exposure if split=='test' else None))
    for (split,d,arm),by_seed in data.items():
        require(set(by_seed)==set(SEEDS),'Five seeds required for absolute table')
        cfg=packet['selection']['datasets'][d]
        roles=[k for k in ('candidate','strong_internal','strongest_published','matched_adam','a_single','b_single')
               if cfg.get(k)==arm]
        main_row=dict(dataset=d,arm=arm,split=split,readout='online' if arm in PUBLISHED else 'fixed_ema',
                      role='|'.join(roles),family='published' if arm in PUBLISHED else
                      'candidate' if 'candidate' in roles else 'internal',n=5,checkpoint_rule='selected',
                      sources='|'.join(dict.fromkeys(source_rows[d,arm,s]['source_ref'] for s in SEEDS)))
        main.append(main_row)
        for cell,metric in by_seed[SEEDS[0]]:
            values=[by_seed[s][cell,metric] for s in SEEDS]
            valid=all(x is not None for x in values)
            main_row[f'{cell}_{metric}_mean']=statistics.fmean(values) if valid else None
            main_row[f'{cell}_{metric}_sd']=statistics.stdev(values) if valid else None
            absolute.append(dict(dataset=d,split=split,arm=arm,cell=cell,metric=metric,n=5 if valid else 0,
                mean=statistics.fmean(values) if valid else None,sd=statistics.stdev(values) if valid else None,
                readout='online' if arm in PUBLISHED else 'fixed_ema',checkpoint_rule='selected'))
    for split in ('validation','test'):
        for d,comparisons in packet['comparisons'].items():
            cfg=packet['selection']['datasets'][d]
            for left,right in comparisons:
                l,r=data[split,d,left],data[split,d,right]
                base=dict(comparison_id=left+'-'+right,dataset=d,split=split,left=left,right=right,
                          left_readout='online' if left in PUBLISHED else 'fixed_ema',
                          right_readout='online' if right in PUBLISHED else 'fixed_ema',checkpoint_rule='selected')
                for cell,metric in l[SEEDS[0]]:
                    a={s:l[s][cell,metric] for s in SEEDS}
                    b={s:r[s][cell,metric] for s in SEEDS}
                    if any(x is None for x in (*a.values(),*b.values())):
                        continue
                    st=results.paired_stats(a,b)
                    summaries.append(dict(**base,cell=cell,metric=metric,**st,
                        guard=st['mean_delta']>=results.GUARD if cell in results.CELLS and metric=='auc' else None))
                    for s in SEEDS:
                        lr,rr=source_rows[d,left,s],source_rows[d,right,s]
                        pairs.append(dict(**base,cell=cell,metric=metric,seed=s,left_value=a[s],right_value=b[s],
                                          delta=a[s]-b[s],left_hash=lr['model_hash'],right_hash=rr['model_hash'],
                                          left_epoch=lr['epoch'],right_epoch=rr['epoch']))
            if cfg.get('unaugmented',cfg['candidate'])=='AB_FROZEN':
                for cell,metric in data[split,d,'AB_FROZEN'][SEEDS[0]]:
                    terms=[data[split,d,a] for a in ('AB_FROZEN',cfg['a_single'],cfg['b_single'],'A00')]
                    if any(t[s][cell,metric] is None for t in terms for s in SEEDS):
                        continue
                    diffs={s:terms[0][s][cell,metric]-terms[1][s][cell,metric]-terms[2][s][cell,metric]+
                           terms[3][s][cell,metric] for s in SEEDS}
                    interactions.append(dict(dataset=d,split=split,cell=cell,metric=metric,a=cfg['a_single'],
                        b=cfg['b_single'],**results.paired_stats(diffs,dict.fromkeys(SEEDS,0.))))
    return dict(main_table=main,endpoints=endpoints,absolute_summary=absolute,paired_seed=pairs,paired_summary=summaries,
                factorial_interactions=interactions,costs=costs)


def plot(summaries,path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    selected=[r for r in summaries if (r['split'],r['cell'],r['metric'])==('test','macro','auc')]
    fig=Figure(figsize=(9,max(3,.36*len(selected)+1.2)),layout='constrained')
    FigureCanvasAgg(fig)
    ax=fig.subplots()
    ax.axvline(0,color='0.5',lw=.8)
    for i,r in enumerate(selected):
        ax.plot([r['ci95_low'],r['ci95_high']],[i,i],color='#236485')
        ax.scatter([r['mean_delta']],[i],marker='D',color='#236485',s=22)
        ax.scatter([r['delta_'+str(s)] for s in SEEDS],[i-.13]*5,color='#a14d30',s=12)
    ax.set_yticks(range(len(selected)),[r['dataset']+'  '+r['comparison_id'] for r in selected])
    ax.invert_yaxis()
    ax.set_xlabel('Frozen Test Macro AUC difference • individual seeds, mean and 95% paired t interval')
    fig.savefig(path,dpi=150)


def guard_plot(packet,summaries,path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig=Figure(figsize=(9,4),layout='constrained')
    FigureCanvasAgg(fig)
    for ax,d in zip(fig.subplots(1,2),DATASETS):
        cfg=packet['selection']['datasets'][d]
        for role,offset in [('strong_internal',-.1),('strongest_published',.1)]:
            ref=cfg[role]
            cells={r['cell']:r for r in summaries if r['dataset']==d and r['split']=='test' and
                   r['metric']=='auc' and (r['left'],r['right'])==(cfg['candidate'],ref)}
            xs=[i+offset for i in range(6)]
            ax.scatter(xs,[cells[c]['mean_delta'] for c in results.CELLS],label=ref,s=22)
            ax.vlines(xs,[cells[c]['ci95_low'] for c in results.CELLS],[cells[c]['ci95_high'] for c in results.CELLS],
                      alpha=.45,lw=1)
        ax.axhline(results.GUARD,color='#a14d30',ls='--',lw=.8,label='−.001 guard')
        ax.axhline(0,color='0.5',lw=.6)
        ax.set_xticks(range(6),results.CELLS)
        ax.set_title(d+' • '+cfg['candidate'])
        ax.set_ylabel('Frozen Test cell AUC difference')
        ax.legend(fontsize=8)
    fig.savefig(path,dpi=150)


def report(art):
    packet=frozen(art)
    folder=art/'S5'
    if packet['status']=='NO_CANDIDATE':
        # Only already-exposed archived core rows. Partial pairs remain historical observations.
        folder.mkdir(exist_ok=True)
        clues=[r for r in historical(art,'test') if r['arm'] in results.CORE]
        results.write_csv(folder/'historical_core_context.csv',clues,['dataset','arm','seed','test_exposure'])
        closure=dict(status='NO_CANDIDATE',reason=packet['selection']['reason'],new_test_evaluations=0,
                     historical_core_context_sha256=sha(folder/'historical_core_context.csv'),
                     freeze_sha256=digest(packet),stage_evidence=packet['evidence'])
        immutable(folder/'S5_REPORT.json',closure)
        return closure
    receipt_index(folder,packet)
    tables=summarize(packet,folder)
    for name,rows in tables.items():
        results.write_csv(folder/(name+'.csv'),rows,['dataset','split','cell','metric'])
    plot(tables['paired_summary'],folder/'paired_test_macro_auc.png')
    guard_plot(packet,tables['paired_summary'],folder/'paired_test_cell_auc.png')
    decisions={}
    for d,cfg in packet['selection']['datasets'].items():
        cand=cfg['candidate']
        pub,internal=STRONG[d],cfg['strong_internal']
        idx={(r['left'],r['right'],r['cell'],r['metric']):r['mean_delta'] for r in tables['paired_summary']
             if r['split']=='test' and r['dataset']==d}
        def effect(a,b):
            def change(c,m):
                return idx[a,b,c,m] if (a,b,c,m) in idx else -idx[b,a,c,m]
            cells={c:change(c,'auc') for c in results.CELLS}
            worst=min(cells,key=cells.get)
            return dict(left=a,right=b,delta_auc=change('macro','auc'),delta_ll=change('macro','logloss'),
                        cells=cells,worst_cell=worst,guard=cells[worst]>=results.GUARD)
        total,strong=effect(cand,pub),effect(cand,internal)
        comp_control=(cfg['matched_adam'] if cand in A_METHODS else cfg['stronger_single']
                      if cand=='AB_FROZEN' else internal)
        component=effect(cand,comp_control)
        delta=total['delta_auc']
        status='BELOW_TOTAL_THRESHOLD'
        if delta>=.002:
            status=('MACRO_GAIN_WITH_CELL_TRADEOFF' if not(total['guard'] and strong['guard']) else
                    'TOTAL_GAIN_WITHOUT_COMPONENT_GAIN' if min(strong['delta_auc'],component['delta_auc'])<=0 else
                    'FINAL_STRONG' if delta>=.003 else 'FINAL_CANDIDATE')
        decisions[d]=dict(status=status,total=total,strong_internal=strong,component=component)
    out=dict(status='COMPLETE',freeze_sha256=digest(packet),datasets=decisions,
             new_test_evaluations=sum(c['test_evaluations'] for c in tables['costs']),
             files={n:sha(folder/n) for n in [*(k+'.csv' for k in tables),
                                                'paired_test_macro_auc.png','paired_test_cell_auc.png']})
    immutable(folder/'S5_REPORT.json',out)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','evaluate','report'))
    parser.add_argument('--artifacts',type=Path,default=ART)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    art=args.artifacts.resolve()
    if args.action=='evaluate':
        out=evaluate(art,torch.device(args.device))
    else:
        out=(freeze if args.action=='freeze' else report)(art)
    print(json.dumps({'action':args.action,'status':out.get('status','EVALUATED')}))


if __name__=='__main__':
    main()
