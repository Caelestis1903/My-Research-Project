"""D-V2: unchanged historical runner; inject core natural population and outputs."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
from .bootstrap import (ART,HERE,SEEDS,activate,population,sources,dump,sha,
                        dependency_inventory,actual_module_paths)

def optimizer_digest(optimizer):
    """Hash values without mutating state or RNG during read-only test evaluation."""
    import torch
    h=hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):
            value=value.detach().cpu().contiguous();h.update(str((value.dtype,tuple(value.shape))).encode());h.update(value.numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):h.update(str(key).encode());add(value[key])
        elif isinstance(value,(list,tuple)):
            for item in value:add(item)
        else:h.update(repr(value).encode())
    add(optimizer.state_dict());return h.hexdigest()

def execute(config,device='cuda:0',review_file=None):
    bundle=population()
    root=activate('v2')
    import v2_20261016 as old
    old.load_stage_s1_bundle=lambda dataset,*args,**kwargs:bundle
    old.sources=lambda:sources('v2')
    old.OUT=ART/'V2'
    review=Path(review_file) if review_file else HERE.parents[1]/'research_plans/2026-10-20_natural_movielens_reassessment/V2_REVIEW.md'
    review_record=dict(path=str(review.resolve()),sha256=sha(review))
    runner=old.Runner(config['arm'],config['seed'],config['run_dir'],
        optimizer=config['optimizer'],lr=config['lr'],device=device,dataset='movielens',epochs=50,
        code_commit='source_archive/inventory.json')
    runner.config.update(study_id='NATURAL_ML_D_V2',population='movielens_natural',
        historical_producer=str(root/'experiments/v2_20261016.py'),
        independent_review=True,review=review_record,exploratory_replication=True,
        test_history='historical test repeatedly used; exploratory replication',
        dependency_inventory=dependency_inventory('v2'),actual_module_paths=actual_module_paths('v2'),
        validation_scope='historical V2 full validation; unchanged selection rule')
    dump(runner.dir/'config.json',runner.config)
    evaluate=runner.evaluate;test_calls=[]
    def observed_evaluate(split,epoch,key):
        if split!='test':return evaluate(split,epoch,key)
        freeze=runner.dir/'selection_frozen.json'
        if not freeze.is_file():raise RuntimeError('test before selection freeze')
        before=(old.initial_state_hash(runner.model),optimizer_digest(runner.opt),runner.step_count)
        start=datetime.now(timezone.utc).isoformat()
        result=evaluate(split,epoch,key)
        after=(old.initial_state_hash(runner.model),optimizer_digest(runner.opt),runner.step_count)
        if before!=after:raise RuntimeError('test evaluation mutated training state')
        test_calls.append(dict(checkpoint=key,epoch=epoch,start_utc=start,
            end_utc=datetime.now(timezone.utc).isoformat(),selection_freeze_sha256=sha(freeze),
            zero_updates=True,model_sha256=before[0],optimizer_sha256=before[1],steps=before[2]))
        dump(runner.dir/'test_access_progress.json',dict(status='TEST_EVALUATION_IN_PROGRESS',
            test_access_started=True,calls=test_calls,exploratory_replication=True))
        return result
    runner.evaluate=observed_evaluate
    runner.fit()
    dump(runner.dir/'test_access_receipt.json',dict(status='COMPLETE',test_access_started=bool(test_calls),
        calls=test_calls,zero_updates_all_calls=all(x['zero_updates'] for x in test_calls),
        selection_frozen_before_all_test_calls=True,exploratory_replication=True,
        historical_freeze_flag_note='original selection_frozen.json flag is a pre-test snapshot',
        source_hashes=sources('v2'),review=review_record,
        original_run_manifest_sha256=sha(runner.dir/'run_manifest.json')))
    dump(runner.dir/'execution_manifest.json',dict(status='COMPLETE',
        original_run_manifest_sha256=sha(runner.dir/'run_manifest.json'),
        final_test_access_receipt_sha256=sha(runner.dir/'test_access_receipt.json'),
        source_hashes=sources('v2'),review=review_record,exploratory_replication=True))

def configurations():
    return [dict(id=f'V2_{opt}_{arm}_seed{seed}',group='V2',seed=seed,arm=arm,
        optimizer=opt,lr=lr,epochs=50,run_dir=str(ART/'V2'/opt/arm/f'seed{seed}'))
        for opt,lr in [('adam',.01),('sgd',.1)] for arm in ('SN','SB') for seed in SEEDS]

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--review-file',type=Path);a=p.parse_args()
    execute(configurations()[a.index],a.device,a.review_file)
if __name__=='__main__':main()
