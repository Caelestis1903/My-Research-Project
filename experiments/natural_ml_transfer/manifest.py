"""One-to-one original commands plus all prespecified support/budget controls."""
import csv
import json
from pathlib import Path
from .common import HERE,ART,ARCHIVE,REPORT,PYTHON,source_hashes,cache_population,dump

def configurations():
    old={}
    with (REPORT/'HISTORICAL_ML_RUN_CONFIG_INDEX.csv').open() as handle:
        for row in csv.DictReader(handle):old[(row['stage'],row['historical_run_id'])]=json.loads(row['historical_config_json'])
    result=[]
    def add(study,record,scope,population,budget='original_epochs',flags=None):
        old_id=record['run_id'];run_id=study+'__'+scope+'__'+old_id
        root=ART/study/scope
        cfg=dict(study=study,scope=scope,population=population,budget=budget,
            run_id=run_id,historical_run_id=old_id,historical_config=record,
            flags=flags,seed=record['seed'],epochs=100,lr=.001,batch_size=40960,
            optimizer=record['optimizer'].lower(),
            run_dir=str(root/'runs'/old_id),state_dir=str(root/'states'/('runs' if study=='CM19' else '')/old_id),
            source_hashes=source_hashes(study),config_path=str(ART/'transfer_configs'/(run_id+'.json')),
            status='REVIEW_PASS',exploratory_replication=True)
        cfg['command']=[PYTHON,'-m','experiments.natural_ml_transfer.run','--config',cfg['config_path'],'--device','cuda:0']
        result.append(cfg)
    queue=json.loads((ARCHIVE/'N19/queue_manifest.json').read_text())['tasks']
    n19=[]
    for task in queue:
        command=task['command']
        if 'experiments/n19_train.py' not in command or '--engineering' in command:continue
        flags=command[command.index('experiments/n19_train.py')+1:]
        if '--dataset' not in flags or flags[flags.index('--dataset')+1]!='movielens':continue
        record=old[('N19',task['name'])];n19.append((record,flags))
        add('N19',record,'natural_full','movielens_natural',flags=flags)
    assert len(n19)==236 and len({r['run_id'] for r,_ in n19})==236
    capacity={'fixed','rank_uniform','rank_protected','domain_codebook','cell_codebook'}
    controls=[(r,f) for r,f in n19 if r['suite']=='natural' and r['method'] in capacity]
    assert len(controls)==40
    for record,flags in controls:
        add('N19',record,'natural_equal_N','movielens_natural_matched_n187500',flags=flags)
        add('N19',record,'natural_matched_exposure','movielens_natural','main178125',flags)
    cm19=[r for r in json.loads((ARCHIVE/'CM19/run_manifest.json').read_text())['runs'] if r['dataset']=='movielens']
    assert len(cm19)==436
    for record in cm19:add('CM19',record,'natural_full','movielens_natural')
    controls=[r for r in cm19 if r['stage']=='G0' and r['model'] in ('fixed','star0','star1')]
    assert len(controls)==24
    for record in controls:
        add('CM19',record,'artificial_fixed2022','movielens_artificial_d2_fixed2022')
        add('CM19',record,'natural_equal_N','movielens_natural_matched_n187500')
        add('CM19',record,'natural_matched_exposure','movielens_natural','main178125')
    assert len(result)==824 and len({r['run_id'] for r in result})==824
    return result

def prepare():
    for pop in ('movielens_natural','movielens_artificial_d2_fixed2022','movielens_natural_matched_n187500'):cache_population(pop)
    runs=configurations()
    for cfg in runs:dump(cfg['config_path'],cfg)
    dump(ART/'transfer_manifest.json',dict(status='REVIEW_PASS',
        counts={'N19':316,'CM19':508},runs=runs,historical_reuse_applied=False,
        note='All 24 artificial CM19 references remain actual runs unless explicitly identity-audited for historical reuse.'))
    diagnostics=[]
    def diagnostic(cfg,kind):
        ident=cfg['study']+'__'+cfg['scope']+'__'+(cfg['historical_run_id'] if cfg['study']=='CM19' else 'scope')+'__'+kind
        diagnostics.append(dict(id=ident,study=cfg['study'],scope=cfg['scope'],kind=kind,
            config_path=cfg['config_path'],requires_complete_scope=True,
            command=[PYTHON,'-m','experiments.natural_ml_transfer.diagnostics','--config',cfg['config_path'],'--kind',kind,'--device','cpu']))
    for scope in ('natural_full','natural_equal_N','natural_matched_exposure'):
        cfg=next(r for r in runs if r['study']=='N19' and r['scope']==scope)
        for kind in ('rank','dynamics')+(('routing','credit') if scope=='natural_full' else ()):diagnostic(cfg,kind)
    for cfg in runs:
        c=cfg['historical_config']
        if cfg['study']=='CM19' and c['stage']=='G0' and c['model'] in ('fixed','star0','star1'):diagnostic(cfg,'local')
    representatives={}
    for cfg in runs:
        c=cfg['historical_config']
        if cfg['study']=='CM19' and cfg['scope']=='natural_full':representatives.setdefault((c['model'],c['variant']),cfg)
    for cfg in representatives.values():diagnostic(cfg,'cost')
    dump(ART/'transfer_diagnostic_manifest.json',dict(status='REVIEW_PASS',tasks=diagnostics,
        costs_scope='One selected-checkpoint representative per model/variant; raw profiler measurements only.',
        test_read=False,training_updates=0))
    print(json.dumps({'N19':316,'CM19':508,'total':len(runs),'raw_diagnostics':len(diagnostics)}))

if __name__=='__main__':prepare()
