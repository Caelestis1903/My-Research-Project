"""Frozen AO04 training and zero-update readout with natural population injection."""
import argparse
import json
import math
from pathlib import Path
import torch
from .bootstrap import HERE,ART,SEEDS,activate,population,sources as base_sources,sha,dump
OUT=ART/'AO04'

def sources():
    return dict(base_sources('ao04'),wrapper_ao04=sha(__file__),wrapper_ao04_diagnostics=sha(HERE/'ao04_diagnostics.py'))

def prepare(device='cuda:0'):
    b=population();activate('ao04')
    import ao04_v12_runner as r
    import ao04_v12_protocol as p
    # Only dataset identity, derived parameter count and execution device change.
    settings={'movielens':('natural',)}
    counts={name:{str(d):int((x[:,-1]==d).sum()) for d in range(3)}
        for name,x in [('train',b.train_x),('validation',b.val_x),('test',b.test_x)]}
    anchors={'encoder_hash':b.features.encoder_hash,'scaler_hash':b.features.scaler_hash,
        **{'base_split_'+key:value for key,value in b.base_split_fingerprints.items()}}
    p.DATASET_SETTINGS=settings;r.DATASET_SETTINGS=settings
    r.EXPECTED_COUNTS={'movielens':{'natural':counts}}
    r.FINGERPRINT_ANCHORS={'movielens':{'natural':anchors}}
    r.DEFAULT_DEVICE=device;r.load_stage_s1_bundle=lambda *a,**kw:b
    # Embedding vocabularies are derived from the common natural training schema.
    def param_total(dataset,arm):
        feats=r.features_from_schema(b.features)
        with r.preserve_rng_state():
            model=r.build_ao04_model(feats,arm,seed=2022)
        return sum(v.numel() for v in model.parameters())
    r.expected_param_total=param_total
    return b,r

def configurations():
    return [dict(id=f'AO04_{arm}_seed{seed}',arm=arm,seed=seed,epochs=50,batch_size=40960,
        population='movielens_natural',population_seed=2022,run_dir=str(OUT/'runs'/f'{arm}_seed{seed}'),
        config_path=str(OUT/'configs'/f'{arm}_seed{seed}.json')) for seed in SEEDS for arm in ('B0','B1','B2')]

def manifest():
    configs=configurations()
    for config in configs:
        config['sources']=sources();dump(config['config_path'],config)
    dump(OUT/'manifest.json',dict(status='REVIEW_PASS',configurations=configs,sources=sources(),test_deferred=True))

def check():
    gate=json.loads((OUT/'review_gate.json').read_text())
    assert gate['verdict']=='PASS' and not gate['blocking_issues'] and gate['sources']==sources()

def execute(config,device):
    check();assert config['sources']==sources() and config['seed'] in SEEDS and config['arm'] in ('B0','B1','B2')
    run=Path(config['run_dir'])
    if run.exists():raise FileExistsError('Preserve attempt, then same-seed full restart: '+str(run))
    torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    b,r=prepare(device)
    # Independent copies prevent races; original deterministic B0 construction pairs all arms.
    b,init=r.paired_init_phase('movielens',config['seed'],init_dir=str(OUT/'paired_init'/config['id']))
    cfg=dict(config,device=device,learning_rate_main=.01,learning_rate_arch=.001,weight_decay=1e-5,
        betas=[.9,.999],eps=1e-8,independent_review=False,review='primary under explicitly invoked start.md',
        exploratory_replication=True,seed_adapter='original _train_val_loop_impl; registered four seeds',
        interruption_policy='preserve attempt and same-seed full-budget restart')
    r._assert_locked_config(cfg,'movielens')
    run.mkdir(parents=True)
    # Original public front door only permits two old seeds. Its fit body is unchanged.
    result=r._train_val_loop_impl(config['arm'],'movielens',config['seed'],str(run),cfg,init.init_state_path,bundle=b)
    r.state_writer(config['arm'],'movielens',config['seed'],str(run),cfg,b,result,init.init_state_path)
    assert result.epochs_completed==50 and result.main_steps==50*math.ceil(len(b.train_x)/40960)
    assert result.examples_processed==50*len(b.train_x) and result.arch_steps==50 and r.TEST_PREDICTION_CALLS==0
    assert config['sources']==sources()
    dump(run/'receipt.json',dict(status='COMPLETE_TRAINING_TEST_DEFERRED',epochs=50,steps=result.main_steps,
        examples=result.examples_processed,arch_steps=result.arch_steps,arch_examples=result.arch_examples_processed,
        selected_epoch_zero_based=result.selected_epoch,selected_epoch=result.selected_epoch+1,
        wall_seconds=result.run_duration_seconds,peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2,
        paired_initial_state_sha256=init.initial_state_hash,sources=sources(),test_evaluated=False))
    dump(run/'checkpoint_index.json',{p.name:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        for p in run.iterdir() if p.suffix in ('.pt','.pth')})

def freeze(closure_path):
    check();closure=json.loads(Path(closure_path).read_text())
    assert closure['training_complete'] and closure['development_disposition']=='CONFIGURATIONS_AND_SELECTION_FROZEN' and not closure['test_read']
    entries={}
    for cfg in configurations():
        run=Path(cfg['run_dir']);assert json.loads((run/'receipt.json').read_text())['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
        entries[cfg['id']]=dict(config_sha256=sha(run/'config.json'),selection_sha256=sha(run/'selection.json'),
            checkpoints={p.name:sha(p) for p in run.glob('*.pth') if p.name.endswith('checkpoint.pth') or p.name.startswith('common_epoch_')})
    dump(OUT/'TEST_FREEZE.json',dict(global_test_unlock=True,runs=entries,sources=sources(),closure_sha256=sha(closure_path)))

def test(device):
    check();f=json.loads((OUT/'TEST_FREEZE.json').read_text());assert f['sources']==sources()
    b,r=prepare(device)
    import ao04_v12_test_evaluator as old
    old.DATASET_SETTINGS={'movielens':('natural',)};old.DEFAULT_DEVICE=device;old.load_stage_s1_bundle=lambda *a,**kw:b
    configs=configurations()
    def verify():
        for cfg in configs:
            run=Path(cfg['run_dir']);e=f['runs'][cfg['id']]
            assert sha(run/'config.json')==e['config_sha256'] and sha(run/'selection.json')==e['selection_sha256']
            assert all(sha(run/name)==digest for name,digest in e['checkpoints'].items())
    verify()
    old.final_test_evaluator({(c['arm'],c['seed'],'m3oe'):c['run_dir'] for c in configs},
        unlock_manifest_path=str(OUT/'TEST_FREEZE.json'),dataset='movielens',setting='natural')
    verify()
    dump(OUT/'test_receipt.json',dict(status='COMPLETE_FROZEN_TEST',optimization_steps=0,sources=sources(),freeze_sha256=sha(OUT/'TEST_FREEZE.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['manifest','train','freeze','test'])
    p.add_argument('--config',type=Path);p.add_argument('--closure',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='manifest':manifest()
    elif a.action=='train':execute(json.loads(a.config.read_text()),a.device)
    elif a.action=='freeze':freeze(a.closure)
    else:test(a.device)

if __name__=='__main__':main()
