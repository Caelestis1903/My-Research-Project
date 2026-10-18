"""Primary-only freeze for the actual completed N1/quota/fixed-ML branch."""
import datetime
import json
from pathlib import Path
import pandas as pd
import n17_common as c


def read(path):
    return json.loads(Path(path).read_text())


def main():
    out = c.OUT
    destinations = [out / name for name in [
        'FINAL_COMPARISONS.json', 'GLOBAL_MODEL_AND_PROTOCOL_FREEZE.json', 'TEST_manifest.json']]
    assert not any(path.exists() for path in destinations), 'preserve the original freeze'
    gates = read(out / 'N1_diagnostic_gates.json')
    assert not any(row['extend_H200'] for row in gates['optimizers'].values())
    local = read(out / 'N3_confirmation_summary/confirmation_gate.json')
    assert local['status'] == 'FROZEN' and not local['need64'] and not local['N4_trigger']
    closures = read(out / 'CONDITIONAL_BRANCH_DECISIONS_PRETEST.json')
    assert closures['status'] == 'FROZEN_PRETEST_CLAIM_SCOPE'
    assert closures['N4']['status'] == 'NOT_TRIGGERED'
    assert closures['N5_C']['status'] == 'TRIGGERED_REQUIRED_FIXED_POPULATION'
    stages = {row['stage']: row['status'] for row in read(out / 'execution_status.json')['stages']}
    assert stages['N0'] == stages['N6_pretraining'] == 'COMPLETE'
    mechanism_validations = ['N2_fixedpoint_validation.json', 'N3_development_validation.json',
                             'N3_SGD_control_validation.json', 'N3_confirmation_validation.json']
    for filename, count in zip(mechanism_validations, [16, 16, 16, 8]):
        audit = read(out / filename)
        assert audit['status'] == 'COMPLETE_VALIDATION' and len(audit['anchors']) == count
    validations = ['N1_pilot_validation.json', 'N1_confirmation_validation.json',
                   'N2_quota_validation.json', 'N5_pilot_validation.json', 'N5_confirmation_validation.json']
    expected = [32, 32, 16, 16, None]
    for filename, count in zip(validations, expected):
        audit = read(out / filename)
        assert audit['status'] == 'COMPLETE_VALIDATION' and not audit['failed']
        assert all(row['status'] == 'VALIDATED_COMPLETE' for row in audit['runs'])
        assert count is None or len(audit['runs']) == count
    ml_manifest = read(out / 'N5_confirmation_manifest.json')
    assert len(read(out / 'N5_confirmation_validation.json')['runs']) == len(ml_manifest['tasks'])
    for filename in ['N5_pilot_manifest', 'N5_confirmation_manifest']:
        manifest = read(out / (filename + '.json'))
        status = read(out / (filename + '.status.json'))
        assert set(status['tasks']) == {task['name'] for task in manifest['tasks']}
        assert all(task['status'] == 'COMPLETE' for task in status['tasks'].values())

    comparisons = []
    for prefix, filename in [('N1', 'N1_winners_frozen.json'), ('N5', 'N5_winners_frozen.json')]:
        winners = read(out / filename)
        assert winners['status'] == 'FROZEN'
        for pair in winners['pairs']:
            opt, seed = pair['optimizer'], pair['seed']
            for kind in ['common', 'own']:
                row = dict(group=f'{prefix}_{kind}_{opt}', seed=seed, budget=100,
                           estimand='common eta q intervention' if kind == 'common' else 'equal-budget per-q LR winners')
                for arm in ['SN', 'SB']:
                    uses = [use for use in winners['uses'] if use['optimizer'] == opt and
                            use['seed'] == seed and use['arm'] == arm and use[kind]]
                    assert len(uses) == 1
                    row[arm] = str(out / 'runs' / uses[0]['name'])
                comparisons.append(row)
    quota = {}
    for task in read(out / 'N2_quota_manifest.json')['tasks']:
        cmd = task['command']; directory = Path(cmd[cmd.index('--run-dir') + 1])
        conf = read(directory / 'config.json')
        key = (conf['seed'], conf['arm'])
        assert key not in quota and conf['optimizer'] == gates['N2_optimizer']
        quota[key] = str(directory)
    for seed in range(2032, 2040):
        comparisons.append(dict(group='N2_quota_' + gates['N2_optimizer'], seed=seed, budget=100,
                                estimand='common eta q intervention under fixed domain quotas',
                                SN=quota[(seed, 'SN')], SB=quota[(seed, 'SB')]))
    for group in {row['group'] for row in comparisons}:
        assert sorted(row['seed'] for row in comparisons if row['group'] == group) == list(range(2032, 2040))

    artifact_path = out / 'completed_artifact_index.json'
    artifacts = {row['run']: row for row in read(artifact_path)['runs']}
    unique = {str(Path(row[arm]).resolve()): Path(row[arm]) for row in comparisons for arm in ['SN', 'SB']}
    runs = {}; tasks = []
    for resolved, directory in sorted(unique.items()):
        conf = read(directory / 'config.json'); receipt = read(directory / 'receipt.json')
        assert not conf['phase'].endswith('_pilot') and conf['all_epoch_models_retained']
        assert not conf['engineering'] and receipt['epochs'] == 100 and not conf['test_access']
        assert not (directory / 'test_receipt.json').exists(), 'test already read; use original freeze'
        for filename, signature in conf['source_hashes'].items():
            assert c.sha(c.ROOT / filename) == signature
        for field, filename in [('model_capture_source_sha256', 'n17_capture.py'),
                                ('ml_source_sha256', 'n17_ml.py'), ('quota_source_sha256', 'n17_quota.py')]:
            if field in conf:
                assert conf[field] == c.sha(Path(__file__).with_name(filename))
        trajectory = pd.read_csv(directory / 'trajectory.csv')
        assert trajectory.epoch.tolist() == list(range(1, 101))
        selected = int(trajectory.loc[trajectory.selection_macro_auc.idxmax(), 'epoch'])
        assert selected == receipt['selected_epoch']
        archived = artifacts[directory.name]['all_epoch_models']; models = {}
        for epoch in range(1, 101):
            info = archived[f'epoch{epoch:03d}_model.pt']; path = Path(info['path'])
            assert path.is_file() and path.stat().st_size == info['bytes']
            assert path.resolve() == (directory / 'all_models' / f'epoch{epoch:03d}_model.pt').resolve()
            models[str(epoch)] = dict(path=str(path), sha256=info['sha256'])
        runs[resolved] = dict(config_sha256=c.sha(directory / 'config.json'),
                              trajectory_sha256=c.sha(directory / 'trajectory.csv'),
                              receipt_sha256=c.sha(directory / 'receipt.json'),
                              selection_file_sha256=c.sha(directory / 'selection_frozen.json'),
                              selection_epoch=selected, models=models)
        name = 'N17test_' + directory.name
        tasks.append(dict(name=name, command=[c.PYTHON, '-u', 'experiments/n17_test_readout.py',
            '--freeze', str(destinations[1]), '--run-dir', str(directory), '--device', '{device}'],
            log=str(out / 'logs' / (name + '.log'))))
    assert len(tasks) == len(runs) and len({row['name'] for row in tasks}) == len(tasks)
    c.dump(destinations[0], dict(status='FROZEN', comparisons=comparisons,
        primary_family=['N1_common_adam:selected:macro_auc', 'N1_common_sgd:selected:macro_auc'],
        orientation='SB minus SN; aliases reuse the same trajectory and never increase n', test_read=False))
    provenance = validations + mechanism_validations + ['N0/receipt.json', 'LITERATURE_COVERAGE.md',
        'N1_winners_frozen.json', 'N5_winners_frozen.json',
        'N1_diagnostic_gates.json', 'CONDITIONAL_BRANCH_DECISIONS_PRETEST.json',
        'N3_confirmation_summary/confirmation_gate.json', 'completed_artifact_index.json']
    c.dump(destinations[1], dict(status='FROZEN', signed_by='primary orchestrator', independent_review=False,
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        all_training_and_selection_definitions_frozen=True, all_required_or_triggered_stages_closed=True,
        test_readout_source_sha256=c.sha(Path(__file__).with_name('n17_test_readout.py')),
        builder_source_sha256=c.sha(__file__), comparison_sha256=c.sha(destinations[0]),
        evidence_sha256={name: c.sha(out / name) for name in provenance}, runs=runs,
        scope='Only current N1/quota/fixed-ML H100 branch; no pilot test; historical test reused; '
              'conditional geometry is not a new formal training comparison. Archived model hashes are '
              'verified against each actual binary by the test reader before prediction.'))
    c.dump(destinations[2], dict(cwd=str(c.ROOT), reservation_mib=1900, tasks=tasks,
        freeze_sha256=c.sha(destinations[1]), comparison_sha256=c.sha(destinations[0]),
        source_sha256=c.sha(__file__), status='READY_WAIT_PRIMARY_REVIEW_AND_LAUNCH'))
    print('frozen comparisons', len(comparisons), 'unique formal runs', len(runs))


if __name__ == '__main__':
    main()
