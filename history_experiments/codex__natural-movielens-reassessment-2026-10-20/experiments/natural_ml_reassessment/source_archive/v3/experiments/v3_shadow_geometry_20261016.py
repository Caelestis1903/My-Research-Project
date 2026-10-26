"""Read-only missing direction telemetry on the frozen V11 shadow replay."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import v3_state_shadow_20261016 as shadow

REPLAY_SHA = '9127a433f34e1143d5280112c7f54f08f592fb82f1ac429d5bcbdb22922aa9d5'
FLOOR = 1e-12


def direction_metrics(delta, native, raw):
    """Full-coordinate double-precision scalar telemetry, no graph mutations."""
    delta, native, raw = [x.detach().double() for x in (delta, native, raw)]
    dn, nn, rn = [float(x.norm()) for x in (delta, native, raw)]
    return dict(delta_norm=dn, native_norm=nn,
                delta_minus_native_norm=float((delta-native).norm()),
                cosine_to_native=(float(delta.dot(native)/(dn*nn))
                                  if min(dn, nn) >= FLOOR else None),
                cosine_to_same_condition_raw=(float(delta.dot(raw)/(dn*rn))
                                              if min(dn, rn) >= FLOOR else None),
                comparison_informative=bool(min(dn, nn) >= FLOOR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    base = shadow.base
    parent_receipt = args.parent / 'receipt.json'
    parent = json.loads(parent_receipt.read_text())
    assert parent['status'] == 'COMPLETE' and parent['source_sha'] == REPLAY_SHA
    assert parent['stages'] == [1, 10, 40] and parent['test_access'] is False
    assert base.sha(shadow.__file__) == REPLAY_SHA
    parent_hashes = {name: base.sha(args.parent / name)
                     for name in ('receipt.json', 'summary.csv', 'cells.csv')}
    for directory, sha in zip(parent['source_runs'], parent['source_receipt_sha']):
        assert base.sha(Path(directory) / 'receipt.json') == sha
    rows, native, raw = [], {}, {}

    def geometry_only(model, bundle, device, split, theta, delta, metadata, groups):
        # Replay will restore its anchor after all probes. This hook only reads
        # independent delta vectors; it never loads a model, backprops or draws RNG.
        if metadata['condition'] == 'BASE':
            for block, indices in dict(groups, ALL=np.arange(delta.numel())).items():
                rows.append(dict(metadata, block=block,
                                 **direction_metrics(delta[indices], delta[indices],
                                                     delta[indices])))
            return dict(metadata), []
        key = (metadata['seed'], metadata['epoch'], metadata['current_arm'])
        condition_key = key + (metadata['condition'],)
        if metadata['condition'] == 'native' and metadata['norm_control'] == 'raw':
            assert key not in native
            native[key] = delta.detach().clone()
        assert key in native
        if metadata['norm_control'] == 'raw':
            assert condition_key not in raw
            raw[condition_key] = delta.detach().clone()
        assert condition_key in raw
        blocks = dict(groups, ALL=np.arange(delta.numel()))
        for block, indices in blocks.items():
            values = direction_metrics(delta[indices], native[key][indices],
                                       raw[condition_key][indices])
            rows.append(dict(metadata, block=block, **values))
        return dict(metadata), []

    argv, evaluate = sys.argv, shadow.evaluate
    try:
        shadow.evaluate = geometry_only
        sys.argv = ['frozen-shadow-replay', '--natural', parent['source_runs'][0],
                    '--balanced', parent['source_runs'][1], '--out', str(args.out),
                    '--device', args.device]
        shadow.main()
    finally:
        sys.argv, shadow.evaluate = argv, evaluate
    replay = json.loads((args.out / 'receipt.json').read_text())
    assert replay['checkpoint_checks'] == parent['checkpoint_checks']
    assert replay['draws'] == parent['draws']
    assert all(base.sha(args.parent / name) == sha for name, sha in parent_hashes.items())
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(['seed', 'epoch', 'current_arm', 'condition',
                                'norm_control', 'block']).any()
    assert set(frame.epoch) == {1, 10, 40}
    assert set(frame.current_arm) == {'BASE', 'SN', 'SB'}
    assert set(frame.block) == {'C', 'P', 'R', 'E', 'ALL'}
    assert len(native) == 6 and len(raw) == 42
    keys = ['seed', 'epoch', 'current_arm', 'condition', 'norm_control',
            'norm_informative']
    parent_frame = pd.read_csv(args.parent / 'summary.csv')
    actual_keys = set(frame[keys].itertuples(index=False, name=None))
    expected_keys = set(parent_frame[keys].itertuples(index=False, name=None))
    assert actual_keys == expected_keys and len(frame) == 5 * len(parent_frame)
    matched = frame[(frame.block == 'ALL') &
                    (frame.norm_control == 'match_native_global_norm')]
    assert np.all(matched.comparison_informative)
    relative = abs(matched.delta_norm - matched.native_norm) / matched.native_norm
    assert float(relative.max()) <= 2e-6
    # The frozen replay temporarily emitted metadata-only placeholders. Remove
    # them explicitly; this supplement does not claim performance computation.
    (args.out / 'summary.csv').unlink()
    (args.out / 'cells.csv').unlink()
    path = args.out / 'update_geometry.csv'
    frame.to_csv(path, index=False)
    base.dump(args.out / 'receipt.json', dict(
        status='COMPLETE', mode='SHADOW_DIRECTION_GEOMETRY_SUPPLEMENT',
        source_sha=base.sha(__file__), replay_source_sha=REPLAY_SHA,
        design_sha=base.sha(shadow.audit.OUT / 'V11_shadow_geometry_supplement_design.json'),
        seed=parent['seed'], stages=parent['stages'], parent=str(args.parent),
        parent_hashes=parent_hashes, source_runs=parent['source_runs'],
        source_receipt_sha=parent['source_receipt_sha'],
        checkpoint_checks=replay['checkpoint_checks'], draws=replay['draws'],
        rows=len(frame), norm_match_max_relative_error=float(relative.max()),
        hashes={'update_geometry.csv': base.sha(path)},
        performance_recomputed=False, test_access=False, independent_review=False,
        scope='Read-only full-coordinate geometry; identical frozen native path, '
              'shadow histories and heldout batches checked against completed '
              'parent. Performance exists only in the unchanged parent.'))


if __name__ == '__main__':
    main()
