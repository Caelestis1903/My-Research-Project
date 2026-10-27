"""The reviewed YAML plus explicit, pre-training user/design resolutions."""
from __future__ import annotations
from pathlib import Path
import copy
import yaml
from .state import digest, sha_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'research_plans/26-10-27_v2/07_EXECUTION_CONFIG.yaml'
METHODS = ('B_REFERENCE', 'DIRECT', 'READ', 'TARGET_ONLY',
           'LOWRANK_RELATION', 'FULL_RELATION')
SEEDS = (2022, 2023, 2024, 2025)
DATASETS = ('movielens', 'kuairand')


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError('Duplicate YAML key: ' + str(key))
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _same_schema(value, reference, path='config'):
    if isinstance(reference, dict):
        if not isinstance(value, dict) or set(value) != set(reference):
            raise ValueError('Unknown or missing configuration key at ' + path)
        for key in reference:
            _same_schema(value[key], reference[key], path + '.' + key)
    elif value != reference or type(value) is not type(reference):
        raise ValueError('Frozen configuration value differs at ' + path)


def load_config(path=DEFAULT):
    reference = yaml.load(DEFAULT.read_text(), Loader=UniqueLoader)
    data = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    _same_schema(data, reference)
    if tuple(data['methods']) != METHODS:
        raise ValueError('Unknown method ID or changed method list')
    expected = dict(h_dim=256, z_dim=2, phi_dim=64, request_dim=32,
                    basis_count=2, function_residual_rank=4, relation_rank=2,
                    reader_dim=64, residual_head_dims=[258, 32, 1])
    if any(data['model'].get(k) != v for k, v in expected.items()):
        raise ValueError('Invalid model dimensions')
    if (data['training']['max_epochs'], data['training']['batch_size'],
        data['training']['candidate_epochs']) != (20, 2048, list(range(5, 21))):
        raise ValueError('Invalid frozen training protocol')
    cfg = copy.deepcopy(data)
    cfg['seeds'] = dict(development=list(SEEDS), uncertainty_only=[], formal=list(SEEDS))
    cfg['comparison']['dev_seeds'] = 4
    cfg['formal']['seeds'] = list(SEEDS)
    cfg['resolution'] = dict(
        design='research_reports/26-10-27_v2/DESIGN_SPECIFICATION.md',
        user_seed_override=list(SEEDS), model_storage='RAM_ONLY',
        native=dict(epochs=24, batch_size=40960, lr=.001, weight_decay=1e-5,
                    eligible_epochs=list(range(12, 25))),
        pilot=dict(passes=12, batch_size=40960, lr=.001, weight_decay=1e-5),
        roles=dict(T0=['fit'], F=['P_form', 'P_read'], S=['P_cal', 'P_update'],
                   Q=['Q_explore'], V=['validation'], E=['test_exposed'],
                   legacy=['Q_legacy_confirm_exposed']),
        target_only_width=40, request_activation='GELU', null_logit=0.,
        prefix_offset='native_logits_only', reset_adam_after_warmup=False,
        local_evidence_required_seeds=4)
    cfg['planning_file_sha256'] = sha_file(path)
    cfg['design_sha256'] = sha_file(ROOT / cfg['resolution']['design'])
    cfg['config_hash'] = digest(cfg)
    return cfg


def rng(seed, dataset, stage, epoch=0, source=0, task=0):
    import numpy as np
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [int(seed), DATASETS.index(dataset), int(stage), int(epoch), int(source), int(task)])))


def torch_seed(seed, dataset, stage, source=0, task=0):
    import numpy as np
    return int(rng(seed, dataset, stage, source=source, task=task).integers(0, 2**32, dtype=np.uint32))


def plan(cfg):
    return dict(protocol=cfg['protocol'], config_hash=cfg['config_hash'],
                seeds=list(SEEDS), prefix_count=8, adaptation_count=48,
                prefix=cfg['resolution']['native'], pilot=cfg['resolution']['pilot'],
                methods=list(METHODS), model_storage='RAM_ONLY',
                stage2='CONDITIONAL_NOT_IMPLEMENTED',
                runs=[dict(dataset=d, seed=s, methods=list(METHODS))
                      for d in DATASETS for s in SEEDS])

