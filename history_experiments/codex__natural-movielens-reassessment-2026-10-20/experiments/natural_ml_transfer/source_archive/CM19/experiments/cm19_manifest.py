"""Expand the preregistered Cartesian matrix; no results-driven filtering."""
import json
from collections import Counter
from cm19_train import OUT,dump
MODELS=['fixed','original','original_sample','star0','star1','pepnet','hamur0','hamur1','mmoe','ple']

def make():
    runs=[];aliases=[]
    def add(stage,models,variants=['native'],opts=['adam','adamw'],decays=['standard'],datasets=['kuairand','movielens'],**extra):
        for model in models:
            for variant in variants:
                for dataset in datasets:
                    for opt in opts:
                        for decay in decays:
                            for seed in [2022,2023,2024,2025]:
                                c=dict(stage=stage,model=model,variant=variant,dataset=dataset,optimizer=opt,decay_mode=decay,seed=seed,batch_size=40960,epochs=50 if dataset=='kuairand' else 100,lr=.003 if dataset=='kuairand' else .001,**extra)
                                run_id=f'{stage}__{dataset}__{model}__{variant}__{opt}__{decay}__s{seed}'
                                for k,v in extra.items():run_id+=f'__{k}_{v}'
                                c['run_id']=run_id;runs.append(c)
    add('G0',MODELS)
    add('G0M',['star_joint','hamur_joint'])
    add('G0MC',['star_private_joint','hamur_private_joint'])
    add('G1',['fixed','star0','star1'],opts=['adam'],decays=['wd0','shared_wd0','slot_wd0','both_wd0','local_adamw'])
    for r in runs:
        if r['stage']=='G1' and r['decay_mode']=='wd0':aliases.append(dict(stage='G1',model=r['model'],run_id=r['run_id'].replace('__adam__','__adamw__'),source=r['run_id'],reason='Adam/AdamW exactly equal at WD0, foreach=False; basic_checks zero error ten steps'))
    add('G2',['fixed','star0','star1'],variants=['no_bypass','additive','bypass_scale','no_modulation'])
    for r in runs:
        if r['stage']=='G2' and r['variant']=='additive':aliases.append(dict(stage='G2',run_id=r['run_id'].replace('__additive__','__capacity_replacement__'),source=r['run_id'],reason='G2-04 capacity replacement is G2-02 additive factors; identical graph/parameters'))
    add('G3',['pepnet'],variants=['ep_neutral','pp_neutral','both_neutral'])
    add('G3',['hamur0','hamur1'],variants=['no_adapter','static','private_hyper'])
    add('G3',['mmoe','ple'],variants=['private','private_matched'])
    for fraction in [.25,.5]:
        for schedule in ['normal_epochs','matched_updates']:
            add('G4',['fixed','star0','star1'],opts=['adam'],fraction=fraction,schedule=schedule)
    add('G4LONG',['fixed','star0'],datasets=['movielens'],max_steps=2350)
    assert len({r['run_id'] for r in runs})==len(runs)
    for r in runs:dump(OUT/'configs'/(r['run_id']+'.json'),r)
    manifest=dict(status='DESIGN_FROZEN_PENDING_REVIEW',independent_training_runs=len(runs),nominal_configurations=len(runs)+len(aliases),counts=dict(Counter(r['stage'] for r in runs)),runs=runs,aliases=aliases)
    dump(OUT/'run_manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ['runs','aliases']},indent=2))
if __name__=='__main__':make()
