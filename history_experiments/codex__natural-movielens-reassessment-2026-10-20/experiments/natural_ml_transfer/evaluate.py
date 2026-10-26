"""Primary-approved scope closure, SHA freeze, then original zero-update tests."""
import argparse
import importlib
import json
from pathlib import Path
from .common import ART,ARCHIVE,source_hashes,configure,dump,sha
from .run import check

def freeze(study,scope,closure):
    decision=json.loads(Path(closure).read_text())
    assert decision['training_complete'] and not decision['test_read']
    assert decision['development_disposition']=='CONFIGURATIONS_AND_SELECTION_FROZEN'
    configs=[r for r in json.loads((ART/'transfer_manifest.json').read_text())['runs'] if r['study']==study and r['scope']==scope]
    assert configs and set(decision['run_ids'])=={r['run_id'] for r in configs}
    rows={}
    for cfg in configs:
        check(cfg);run=Path(cfg['run_dir']);c=json.loads((run/'config.json').read_text())
        receipt=json.loads((run/'receipt.json').read_text())
        assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
        states=Path(c['state_directory'])
        rows[cfg['historical_run_id']]=dict(selection_sha256=sha(run/'selection_frozen.json'),
            checkpoint_sha256={label:sha(states/file) for label,file in [('selected','selected.pt'),('final','last.pt')]},
            config_sha256=sha(run/'config.json'),receipt_sha256=sha(run/'receipt.json'),global_run_id=cfg['run_id'])
    training={p.name:sha(p) for p in (ARCHIVE/study/'experiments').glob('cm19*.py')} if study=='CM19' else {}
    dump(ART/study/scope/'TEST_FREEZE.json',dict(runs=rows,training_source_hashes=training,
        reassessment_source_hashes=source_hashes(study),closure_sha256=sha(closure),exploratory_replication=True))

def evaluate(cfg,device):
    check(cfg);dm=configure(cfg)
    freeze_path=dm.OUT/'TEST_FREEZE.json';f=json.loads(freeze_path.read_text())
    assert f['reassessment_source_hashes']==source_hashes(cfg['study'])
    expected=f['runs'][cfg['historical_run_id']]
    run=Path(cfg['run_dir']);c=json.loads((run/'config.json').read_text());states=Path(c['state_directory'])
    assert sha(run/'config.json')==expected['config_sha256']
    assert sha(run/'selection_frozen.json')==expected['selection_sha256']
    for label,file in [('selected','selected.pt'),('final','last.pt')]:assert sha(states/file)==expected['checkpoint_sha256'][label]
    if cfg['study']=='N19':
        old=importlib.import_module('n19_evaluate');old.OUT=dm.OUT;old.population=dm.population
        import torch
        old.evaluate(run,torch.device(device))
    else:
        train=importlib.import_module('cm19_train');train.OUT=dm.OUT;train.ART=dm.ART;train.population=dm.population
        old=importlib.import_module('cm19_evaluate');old.OUT=dm.OUT;old.ART=dm.ART;old.population=dm.population
        old.evaluate(cfg['historical_run_id'],device)
    for label,file in [('selected','selected.pt'),('final','last.pt')]:assert sha(states/file)==expected['checkpoint_sha256'][label]
    dump(run/'reassessment_test_receipt.json',dict(status='COMPLETE_FROZEN_TEST',optimization_steps=0,
        freeze_sha256=sha(freeze_path),original_test_receipt_sha256=sha(run/'test_receipt.json'),
        checkpoint_sha256_before_and_after=expected['checkpoint_sha256'],exploratory_replication=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','test']);p.add_argument('--study');p.add_argument('--scope')
    p.add_argument('--closure',type=Path);p.add_argument('--config',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='freeze':freeze(a.study,a.scope,a.closure)
    else:evaluate(json.loads(a.config.read_text()),a.device)
if __name__=='__main__':main()
