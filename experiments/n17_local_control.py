"""Primary-only mechanical preparation and completeness audit for frozen N3."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import n17_common as c


def prepare(a):
    frozen = json.loads(a.candidate.read_text())
    assert frozen['status'] == 'FROZEN'
    choice = frozen['candidate']
    assert choice is not None, 'NO_DEFINED_CANDIDATE: no confirmation matrix'
    assert choice['block'] in ['C_embedding', 'C_shared_STAR', 'C_skip', 'C_star_mlp', 'C_experts']
    assert choice['epoch'] in [10, 40]
    supplement = a.kind.endswith('_supplement')
    development = a.kind == 'development_supplement'
    if development:
        assert choice['need64'], 'development precision supplement not triggered'
    elif supplement:
        assert a.gate is not None
        gate = json.loads(a.gate.read_text())
        assert gate['need64'] and gate['candidate_sha'] == c.sha(a.candidate)
    winners = json.loads((c.OUT / 'N1_winners_frozen.json').read_text())
    resource = json.loads((c.OUT / 'N1_diagnostic_gates.json').read_text())
    opt = resource['N2_optimizer']
    seeds = range(2032, 2036) if development else range(2036, 2040)
    tasks = []
    anchors = []
    for seed in seeds:
        pair = next(x for x in winners['pairs'] if x['optimizer'] == opt and x['seed'] == seed)
        for arm in ['SN', 'SB']:
            checkpoint = Path(pair[arm]) / f"epoch{choice['epoch']:03d}.pt"
            assert checkpoint.is_file()
            parent = json.loads((Path(pair[arm]) / 'receipt.json').read_text())
            assert parent['epochs'] >= 100
            name = f"N3{a.kind}_{opt}_{arm}_seed{seed}_E{choice['epoch']}_{choice['block']}"
            directory = c.OUT / ('N3_' + a.kind) / name
            assert not directory.exists(), 'preserve existing results; use existing manifest to resume'
            command = [c.PYTHON, '-u', 'experiments/n17_local.py', '--checkpoint', str(checkpoint), '--seed', str(seed),
                       '--optimizer', opt, '--lr', str(pair['common_lr']), '--anchor-arm', arm, '--out', str(directory),
                       '--draws', '32', '--start-draw', '32' if supplement else '0', '--only-block', choice['block'], '--device', '{device}']
            tasks.append(dict(name=name, command=command, log=str(c.OUT / 'logs' / (name + '.log'))))
            anchors.append(dict(name=name, seed=seed, anchor_arm=arm, epoch=choice['epoch'], checkpoint=str(checkpoint), checkpoint_sha=c.sha(checkpoint)))
    assert len(tasks) == 8 and not a.out.exists()
    c.dump(a.out, dict(cwd=str(c.ROOT), reservation_mib=1900, tasks=tasks, anchors=anchors, kind=a.kind,
                       candidate_sha=c.sha(a.candidate), gate_sha=c.sha(a.gate) if a.gate else None,
                       source_sha=c.sha(__file__), local_source_sha=c.sha(Path(__file__).with_name('n17_local.py')), test_read=False))
    print(a.kind, len(tasks), choice['block'], choice['epoch'])


def audit(a):
    manifest = json.loads(a.manifest.read_text())
    records = []
    for task in manifest['tasks']:
        command = task['command']
        directory = Path(command[command.index('--out') + 1])
        conf = json.loads((directory / 'config.json').read_text())
        receipt = json.loads((directory / 'receipt.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['anchor_unchanged'] and not conf['engineering']
        assert receipt['config_sha'] == c.sha(directory / 'config.json')
        assert receipt['source_sha'] == conf['source_sha'] == c.sha(Path(__file__).with_name('n17_local.py'))
        assert conf['checkpoint_sha'] == c.sha(conf['checkpoint'])
        for flag, key, cast in [('--seed', 'seed', int), ('--optimizer', 'optimizer', str),
                                ('--lr', 'lr', float), ('--anchor-arm', 'anchor_arm', str)]:
            assert conf[key] == cast(command[command.index(flag) + 1])
        assert Path(conf['checkpoint']).resolve() == Path(command[command.index('--checkpoint') + 1]).resolve()
        assert receipt['epoch'] == conf['epoch'] and receipt['draws'] == conf['draws'] and receipt['start_draw'] == conf['start_draw']
        start = int(conf['start_draw']); count = int(conf['draws'])
        assert count == 32 and start in (0, 32)
        expected = set(range(start, start + count))
        draws = pd.read_csv(directory / 'draws.csv')
        assert set(draws.draw) == expected and not draws.duplicated(['draw', 'arm']).any()
        assert set(draws.arm) == {'SN', 'SB'} and len(draws) == 2 * count
        assert draws.row_ids_sha.map(lambda x: len(x) == 64).all()
        macro = pd.read_csv(directory / 'macro_readouts.csv')
        cell = pd.read_csv(directory / 'cell_readouts.csv')
        assert set(macro[macro.draw >= 0].draw) == set(cell[cell.draw >= 0].draw) == expected
        assert not macro.duplicated(['draw', 'case', 'split']).any()
        assert not cell.duplicated(['draw', 'case', 'split', 'domain', 'task']).any()
        assert set(macro.split) == set(cell.split) == {'selection', 'diagnostic'}
        for frame in [macro, cell]:
            assert np.isfinite(frame[['auc', 'logloss', 'train_J', 'applied_block_update_norm']].to_numpy()).all()
        blocks = [conf['only_block']] if conf['only_block'] else ['C', 'C_embedding', 'C_shared_STAR', 'C_skip', 'C_star_mlp', 'C_experts', 'P']
        for draw in [-1, *sorted(expected)]:
            part = macro[macro.draw == draw]
            cases = set(part.case)
            required = {'unupdated_anchor'} if draw == -1 else {'native_SN', 'native_SB', *[block + '_full' for block in blocks]}
            assert required.issubset(cases)
            assert (part.groupby('case').size() == 2).all()
            assert (cell[cell.draw == draw].groupby('case').size() == 12).all()
            assert cases == set(cell[cell.draw == draw].case)
            if draw >= 0:
                for block in blocks:
                    if block == 'P':
                        continue
                    full = part[part.case == block + '_full']
                    undefined = bool(full.direction_undefined.any())
                    for suffix in ['_norm', '_direction']:
                        assert ((block + suffix) in cases) == (not undefined)
        records.append(dict(name=task['name'], directory=str(directory), draws=count, start_draw=start,
                            seed=conf['seed'], anchor_arm=conf['anchor_arm'], epoch=conf['epoch'],
                            files={name: c.sha(directory / name) for name in ['config.json', 'receipt.json', 'draws.csv', 'macro_readouts.csv', 'cell_readouts.csv']}))
    assert not a.out.exists()
    c.dump(a.out, dict(status='COMPLETE_VALIDATION', manifest_sha=c.sha(a.manifest), source_sha=c.sha(__file__), anchors=records, test_read=False))
    print('validated N3 anchors', len(records))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--candidate', type=Path, required=True)
    q.add_argument('--kind', choices=['confirmation', 'development_supplement', 'confirmation_supplement'], required=True)
    q.add_argument('--gate', type=Path)
    q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('audit')
    q.add_argument('--manifest', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a) if a.action == 'prepare' else audit(a)
