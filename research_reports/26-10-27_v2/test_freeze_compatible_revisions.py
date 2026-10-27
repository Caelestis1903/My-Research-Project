import importlib.util
import importlib
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('compatible_freeze_helper', HERE/'freeze_compatible_revisions.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def load_snapshot_data(label, source_hash):
    package = helper.SNAPSHOTS/source_hash[:8]/'package'
    spec = importlib.util.spec_from_file_location(label, package/'__init__.py', submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    spec.loader.exec_module(module)
    return importlib.import_module(label+'.data').Data


def test_two_views_authorize_real_old_and_new_validators_and_require_quorum(tmp_path):
    cfg = helper.load_config()
    equivalence, versions = helper.source_equivalence()
    for dataset in helper.DATASETS:
        full_hash = helper.OLD if dataset == 'movielens' else helper.NEW
        for seed in helper.SEEDS:
            base = dict(protocol=cfg['protocol'], dataset=dataset, seed=seed,
                        config_hash=cfg['config_hash'], source_hash=full_hash,
                        data_role_hash=dataset+'_data', preprocessor_hash=dataset+'_preprocessor')
            selected = {m:f'{dataset}_{seed}_{m}_selected' for m in helper.METHODS}
            final = {m:f'{dataset}_{seed}_{m}_final' for m in helper.METHODS}
            identity = dict(base, selected=selected, final=final)
            worker = tmp_path/'workers'/dataset/f'seed{seed}'
            helper.write_json(worker/'READY.json', dict(status='READY_FOR_GLOBAL_FREEZE', identity=identity))
            helper.write_json(worker/'source_identity.json', versions[full_hash])
            helper.write_json(worker/'effective_config.json', dict(config=cfg))
            helper.write_json(worker/'data_manifest.json', dict(data_role_hash=base['data_role_hash'], preprocessor_hash=base['preprocessor_hash']))
            for method in helper.METHODS:
                helper.write_json(tmp_path/'runs'/dataset/method/f'seed{seed}'/'run_receipt.json',
                                  dict(identity=dict(base, method=method), selected_model_hash=selected[method], final_model_hash=final[method]))
    decision = dict(status='FINAL_METHODS_FROZEN', stage2={d:'NOT_TRIGGERED' for d in helper.DATASETS})
    common, canonical, views, assignment = helper.build_views(tmp_path, cfg, decision, equivalence, versions, dict(unit_fixture=True))
    assert len(common['workers']) == 8
    assert views['GLOBAL_FREEZE.json']['workers'] == views['GLOBAL_FREEZE_FP32_STANDARD.json']['workers']
    assert views['GLOBAL_FREEZE.json']['canonical_identity'] == views['GLOBAL_FREEZE_FP32_STANDARD.json']['canonical_identity'] == canonical
    old_data_class = load_snapshot_data('relation_fp32_r1_fixture', helper.OLD)
    old_validator = old_data_class.__new__(old_data_class)
    new_data_class = load_snapshot_data('relation_fp32_r2_fixture', helper.NEW)
    new_validator = new_data_class.__new__(new_data_class)
    for validator, dataset, view in [(old_validator, 'movielens', 'GLOBAL_FREEZE.json'),
                                     (new_validator, 'kuairand', 'GLOBAL_FREEZE_FP32_STANDARD.json')]:
        validator.dataset = dataset
        identity = common['workers'][f'{dataset}/seed2022']
        validator.authorize_final(views[view], identity)
        assert validator.outer_frozen
        wrong = 'GLOBAL_FREEZE_FP32_STANDARD.json' if view == 'GLOBAL_FREEZE.json' else 'GLOBAL_FREEZE.json'
        with pytest.raises(PermissionError):
            validator.authorize_final(views[wrong], identity)
    missing = tmp_path/'workers'/'kuairand'/'seed2025'/'READY.json'
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        helper.build_views(tmp_path, cfg, decision, equivalence, versions, dict(unit_fixture=True))

