"""Frozen execution configuration and explicitly registered experiment arms."""
from pathlib import Path
import copy
import json
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'research_plans/2026-10-28'
DATASETS = ('movielens', 'kuairand')
DEV_SEEDS = (2022, 2023)
LOCKED_SEEDS = (2024, 2025, 2026)
P1 = ('TARGET_MATCH', 'STATIC_LOCAL', 'CLUSTER_ONLY', 'FIXED_READ',
      'RELATION_ONLY', 'CLUSTER_RELATION')
SWITCHES = ('GAIN_CLUSTER', 'GAIN_RELATION', 'DYNAMIC_SPLIT', 'CONTRASTIVE',
            'SOURCE_WRITE', 'LOW_RANK')
UNIMPLEMENTED_ARMS = {'BASP_B','BASP_C','CONTINUOUS_GAIN','GAIN_SHUFFLE',
                      'TARGET_SUPERVISED_LIBRARY','UNIFORM_OPPORTUNITY'}


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key, value in node.value:
        name = loader.construct_object(key, deep=deep)
        if name in result:
            raise ValueError('Duplicate configuration key: '+str(name))
        result[name] = loader.construct_object(value, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def validate_schema(value, reference, path='config'):
    if isinstance(reference, dict):
        if not isinstance(value, dict) or set(value) != set(reference):
            raise ValueError('Unknown or missing key at '+path)
        for key in reference:
            validate_schema(value[key], reference[key], path+'.'+key)
    elif type(value) is not type(reference) or value != reference:
        raise ValueError('Frozen value differs at '+path)


def load_config(path=None):
    cfg = yaml.load((PLAN/'08_config.yaml').read_text(), Loader=UniqueLoader)
    if path is not None:
        candidate = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
        validate_schema(candidate, cfg)
    cfg = copy.deepcopy(cfg)
    cfg['status'] = 'EXECUTION'
    cfg['run_gpu_experiments_now'] = True
    cfg['seeds'] = dict(development=list(DEV_SEEDS), locked_reevaluation=list(LOCKED_SEEDS))
    cfg['budgets'].update(p5_max_adaptation_trajectories=24, locked_total_max=24,
                          prefix_dataset_seed_max=10)
    cfg['execution'] = dict(protocol='BASP-A-20261028-v1', model_storage='RAM_ONLY',
                            development_seeds=list(DEV_SEEDS), locked_seeds=list(LOCKED_SEEDS),
                            reference_commit='03c338092008a71b2d360a61b06d985eac551ec7')
    return cfg


def arms():
    return {a['id']: a for a in json.loads((PLAN/'09_experiments.json').read_text())['arms']}


def validate_arm(arm):
    if set(arm['switches']) != set(SWITCHES):
        raise ValueError('Six explicit switches required')
    if any(type(x) is not bool for x in arm['switches'].values()):
        raise ValueError('Switches must be booleans')
    if arm.get('relation_fallback') not in ('NONE', 'FIXED'):
        raise ValueError('Explicit relation_fallback required')
    if arm.get('train_scope') not in ('ADAPTER_JOINT', 'FULL_MODEL_JOINT'):
        raise ValueError('Unknown train_scope')
    if arm['switches']['CONTRASTIVE'] and not arm['switches']['GAIN_CLUSTER'] and not arm.get('aux_matched', False):
        raise ValueError('CL without gain clustering requires aux_matched=true')
    return arm


def require_implemented(arm):
    if arm['id'] in UNIMPLEMENTED_ARMS or arm['kind'] not in ('BASP_A','TARGET_MATCH'):
        raise NotImplementedError('Conditional method requires its own reviewed implementation: '+arm['id'])
    return arm


def rng(seed, dataset, namespace, epoch=0, source=0, task=0, component=0):
    coordinates=[int(seed),DATASETS.index(dataset),int(namespace),int(epoch),int(source),int(task)]
    if component:
        coordinates.append(int(component))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(coordinates)))


def torch_seed(seed, dataset, namespace, source=0, task=0, component=0):
    # Preserve the canonical historical native/pilot initialization coordinates.
    seq = np.random.SeedSequence([int(seed), DATASETS.index(dataset), int(namespace),
                                 int(source), int(task), int(component), 0])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def plan(cfg):
    return dict(protocol=cfg['execution']['protocol'], development_seeds=list(DEV_SEEDS),
                locked_seeds=list(LOCKED_SEEDS), P1=list(P1), budgets=cfg['budgets'],
                model_storage='RAM_ONLY', historical_E='LOCKED_UNTIL_GLOBAL_FREEZE',
                arms=list(arms().values()))
