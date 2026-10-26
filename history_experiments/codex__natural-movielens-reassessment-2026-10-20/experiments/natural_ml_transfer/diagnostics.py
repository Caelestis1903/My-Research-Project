"""Scope the original raw mechanism measurements to the reassessment population."""
import argparse
import importlib
import json
from pathlib import Path
from .common import ART,configure,ARCHIVE,dump,sha
from .run import check

def execute(cfg,kind,device='cpu'):
    check(cfg)
    scope=[r for r in json.loads((ART/'transfer_manifest.json').read_text())['runs'] if r['study']==cfg['study'] and r['scope']==cfg['scope']]
    for item in scope:
        receipt=json.loads((Path(item['run_dir'])/'reassessment_receipt.json').read_text())
        assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
    dm=configure(cfg)
    if cfg['study']=='N19':
        name={'rank':'n19_rank_diagnostics','routing':'n19_routing_diagnostics',
              'credit':'n19_credit_diagnostics','dynamics':'n19_dynamics'}[kind]
        module=importlib.import_module(name);module.OUT=dm.OUT;module.population=dm.population
        if kind in ('rank','credit'):
            # Exact original calculations, restricted to the declared ML scope.
            # No fictitious KuaiRand population and no unrelated KR remeasurement.
            file=ARCHIVE/'N19/experiments'/(name+'.py')
            source=file.read_text();target="for dataset in ['movielens', 'kuairand']:"
            assert source.count(target)==1
            exec(compile(source.replace(target,"for dataset in ['movielens']:"),str(file),'exec'),module.__dict__)
            module.OUT=dm.OUT;module.population=dm.population
        module.main()
    else:
        train=importlib.import_module('cm19_train');train.OUT=dm.OUT;train.ART=dm.ART;train.population=dm.population
        module=importlib.import_module('cm19_local_mechanism' if kind=='local' else 'cm19_costs')
        module.OUT=dm.OUT;module.population=dm.population
        (module.local if kind=='local' else module.measure)(cfg['historical_run_id'],device)
    dump(dm.OUT/'diagnostic_receipts'/(cfg['historical_run_id']+'__'+kind+'.json'),dict(
        status='COMPLETE_RAW_DIAGNOSTIC',kind=kind,global_run_id=cfg['run_id'],training_updates=0,
        test_read=False,scope='MovieLens only; original measurement formulas',source_hashes=cfg['source_hashes']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--kind',choices=['rank','routing','credit','dynamics','local','cost'],required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();execute(json.loads(a.config.read_text()),a.kind,a.device)
if __name__=='__main__':main()
