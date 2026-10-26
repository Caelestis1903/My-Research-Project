"""Primary-only summaries of completed, frozen N2 fixed-point artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import n17_common as c


def ratio(a, b):
    return float(a / b) if b != 0 else None


def coordinates(order, block):
    selected = []
    for row in order:
        name = row['name']
        sub = ('C_embedding' if name.startswith('embedding.') else
               'C_shared_STAR' if name in ('shared_weight', 'shared_bias') else
               'C_skip' if name.startswith('skip_conn.') else
               'C_star_mlp' if name.startswith('star_mlp.') else
               'C_experts' if name.startswith('expert.') else None)
        if block == 'ALL' or row['block'] == block or (row['block'] == 'C' and sub == block):
            selected.extend(range(row['start'], row['end']))
    return np.asarray(selected, dtype=np.int64)


def summary(frame, groups):
    excluded = set(groups) | {'seed', 'epoch', 'k', 'batch_size', 'coordinates'}
    columns = [key for key in frame.select_dtypes(include='number') if key not in excluded]
    rows = []
    for keys, part in frame.groupby(groups, dropna=False):
        assert len(part) == part.seed.nunique() == 4
        base = dict(zip(groups, keys))
        for key in columns:
            values = part[key].dropna().to_numpy(dtype=float)
            rows.append({**base, 'metric': key, 'n_training_seeds': 4,
                         'n_defined': len(values), 'mean': float(values.mean()) if len(values) else None,
                         'std': float(values.std(ddof=1)) if len(values) > 1 else None,
                         'minimum': float(values.min()) if len(values) else None,
                         'maximum': float(values.max()) if len(values) else None})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validation-equivalence', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    equivalence = None
    allowed_sources = None
    if args.validation_equivalence:
        equivalence = json.loads(args.validation_equivalence.read_text())
        assert equivalence['status'] == 'PASS_SELF_REVIEW_VALIDATION_ONLY'
        assert equivalence['estimator_unchanged_after_validation_reversal']
        assert equivalence['original_source_sha256'] == equivalence['original_after_reversal_sha256']
        assert c.sha(equivalence['original_source']) == equivalence['original_source_sha256']
        assert c.sha(equivalence['corrected_source']) == equivalence['corrected_source_sha256']
        allowed_sources = {equivalence['original_source_sha256'], equivalence['corrected_source_sha256']}
    anchor_file = c.OUT / 'N2_N3_anchor_freeze.json'
    anchors = json.loads(anchor_file.read_text())
    assert anchors['status'] == 'FROZEN'
    frozen = {row['checkpoint']: row['sha256'] for row in anchors['anchors']}
    raw, contrasts, mappings, sources, identities = [], [], [], [], set()
    source_signatures = set()
    for task in manifest['tasks']:
        command = task['command']; directory = Path(command[command.index('--out') + 1])
        config = json.loads((directory / 'config.json').read_text())
        receipt = json.loads((directory / 'receipt.json').read_text())
        assert receipt['status'] == 'COMPLETE' and not config['engineering']
        assert config['k'] == receipt['k'] == 512 and config['batch_size'] == 40960
        assert receipt['config_sha'] == c.sha(directory / 'config.json')
        assert receipt['source_sha'] == config['source_sha']
        if allowed_sources:
            assert config['source_sha'] in allowed_sources
            if config['source_sha'] == equivalence['corrected_source_sha256']:
                assert config['identity_validation_version'] == 'same_graph_v1'
                assert config['original_geometry_source_sha'] == equivalence['original_source_sha256']
        source_keys = ('quota_source_sha', 'base_sources') if allowed_sources else ('source_sha', 'quota_source_sha', 'base_sources')
        source_signatures.add(json.dumps({key: config[key] for key in source_keys}, sort_keys=True))
        assert config['checkpoint_sha'] == frozen[config['checkpoint']] == c.sha(Path(config['checkpoint']))
        base = {key: config[key] for key in ('optimizer', 'seed', 'anchor_arm', 'epoch')}
        identity = (base['seed'], base['anchor_arm'], base['epoch'])
        assert identity not in identities; identities.add(identity)
        rows = json.loads((directory / 'geometry.json').read_text())
        lookup = {(row['arm'], row['variant'], row['block']): row for row in rows}
        blocks = {row['block'] for row in rows}
        assert len(rows) == len(lookup) == 4 * len(blocks)
        assert blocks == {'ALL', 'C', 'P', 'R', 'E', 'C_embedding', 'C_shared_STAR', 'C_skip', 'C_star_mlp', 'C_experts'}
        vector_files = {(arm, variant): directory / f'{arm}_{variant}_moments.npz'
                        for arm in ('SN', 'SB') for variant in ('IID', 'quota')}
        vectors = {key: dict(np.load(path)) for key, path in vector_files.items()}
        assert all(np.isfinite(array).all() for bank in vectors.values() for array in bank.values())
        adam = json.loads((directory / 'adam_mapping.json').read_text())
        amap = {(row['arm'], row['variant'], row['block']): row for row in adam}
        assert len(amap) == len(rows) if config['adam_map'] else not amap
        for row in rows:
            assert all(row[key] == value for key, value in base.items())
            assert row['k'] == 512
            assert all(np.isfinite(value).all() for value in row.values() if isinstance(value, (int, float, list)))
            total = row['sum_trace_prediction']; empirical = row['empirical_covariance_trace']
            identity = receipt['identity']
            by_block = identity.get('cross_forward_block_residual_norm')
            cross = by_block[row['block']] if by_block else identity['residual']
            raw.append({**row, 'predicted_between_fraction': ratio(row['between_trace_prediction'], total),
                        'empirical_over_predicted_trace': ratio(empirical, total),
                        'trace_prediction_residual': empirical - total,
                        'bias_over_mc_rms': ratio(row['mean_bias_norm'], row['mc_rms_se']),
                        'validation_cross_forward_residual_or_upper_bound': cross,
                        'validation_cross_forward_scope': 'first SN-IID draw block norm' if by_block else 'first SN-IID draw whole-vector bound for this block',
                        'validation_cross_over_single_gradient_mc_rms': ratio(cross, np.sqrt(max(0., empirical))),
                        'validation_cross_over_mean_gradient_mc_rms': ratio(cross, row['mc_rms_se'])})
        for block in sorted(blocks):
            ix = coordinates(config['parameter_order'], block)
            for arm in ('SN', 'SB'):
                iid = lookup[arm, 'IID', block]; quota = lookup[arm, 'quota', block]
                assert len(ix) == iid['coordinates'] == quota['coordinates']
                contrasts.append({**base, 'block': block, 'contrast': arm + '_quota_minus_IID',
                                  'empirical_trace_difference': quota['empirical_covariance_trace'] - iid['empirical_covariance_trace'],
                                  'predicted_trace_difference': quota['sum_trace_prediction'] - iid['sum_trace_prediction'],
                                  'empirical_trace_ratio': ratio(quota['empirical_covariance_trace'], iid['empirical_covariance_trace'])})
            gap = {}
            for variant in ('IID', 'quota'):
                sn = lookup['SN', variant, block]; sb = lookup['SB', variant, block]
                gap[variant] = (sb['empirical_covariance_trace'] - sn['empirical_covariance_trace'],
                                sb['sum_trace_prediction'] - sn['sum_trace_prediction'])
                contrasts.append({**base, 'block': block, 'contrast': variant + '_SB_minus_SN',
                                  'empirical_trace_difference': gap[variant][0],
                                  'predicted_trace_difference': gap[variant][1],
                                  'empirical_trace_ratio': ratio(sb['empirical_covariance_trace'], sn['empirical_covariance_trace'])})
                sv = vectors['SN', variant]; bv = vectors['SB', variant]
                difference = sv['mean'][ix] - bv['mean'][ix]
                mapping = {**base, 'block': block, 'variant': variant,
                           'gradient_mean_difference_norm': float(np.linalg.norm(difference)),
                           'gradient_difference_mc_rms_upper_bound': sn['mc_rms_se'] + sb['mc_rms_se']}
                if config['adam_map']:
                    sa = amap['SN', variant, block]; ba = amap['SB', variant, block]
                    mapping.update(mean_update_difference_norm=float(np.linalg.norm(sv['mean_update'][ix] - bv['mean_update'][ix])),
                                   fixed_denominator_mean_update_difference_norm=float(np.linalg.norm(sv['mean_linear_update'][ix] - bv['mean_linear_update'][ix])),
                                   mean_update_difference_mc_rms_upper_bound=sa['update_mc_rms_se'] + ba['update_mc_rms_se'],
                                   SN_jensen_residual_norm=sa['jensen_residual_norm'], SB_jensen_residual_norm=ba['jensen_residual_norm'])
                elif config['optimizer'] == 'sgd':
                    mapping.update(derived_SGD_mean_update_difference_norm=config['lr'] * float(np.linalg.norm(difference)),
                                   derived_SGD_difference_mc_rms_upper_bound=config['lr'] * (sn['mc_rms_se'] + sb['mc_rms_se']))
                mappings.append(mapping)
            contrasts.append({**base, 'block': block, 'contrast': 'quota_interaction',
                              'empirical_trace_difference': gap['quota'][0] - gap['IID'][0],
                              'predicted_trace_difference': gap['quota'][1] - gap['IID'][1], 'empirical_trace_ratio': None})
        files = ['config.json', 'receipt.json', 'geometry.json', 'adam_mapping.json', 'exact_domain_means.npz', 'draw_counts.csv']
        files.extend(path.name for path in vector_files.values())
        sources.append(dict(directory=str(directory.resolve()), files={name: c.sha(directory / name) for name in files}))
    assert identities == {(s, arm, e) for s in range(2032, 2036) for arm in ('SN', 'SB') for e in (10, 40)}
    assert len(source_signatures) == 1
    args.out.mkdir(parents=True, exist_ok=False)
    tables = {'raw_moments': pd.DataFrame(raw), 'trace_contrasts': pd.DataFrame(contrasts), 'mean_mapping_contrasts': pd.DataFrame(mappings)}
    for name, frame in tables.items():
        frame.to_csv(args.out / (name + '.csv'), index=False)
        extras = ['arm', 'variant'] if name == 'raw_moments' else ['contrast'] if name == 'trace_contrasts' else ['variant']
        summary(frame, ['optimizer', 'anchor_arm', 'epoch', 'block', *extras]).to_csv(args.out / (name + '_seed_summary.csv'), index=False)
    c.dump(args.out / 'receipt.json', dict(status='COMPLETE', manifest_sha=c.sha(args.manifest), anchor_freeze_sha=c.sha(anchor_file),
           source_sha=c.sha(__file__), anchors=len(identities), inputs=sources, test_read=False,
           validation_equivalence_sha=c.sha(args.validation_equivalence) if args.validation_equivalence else None,
           scope='four training seeds separately per path and phase; marginal MC RMS sum is not a confidence bound; nonzero noisy vector norm does not establish population mean drift; no unique trace mediation or historical necessity claim'))
    print('summarized', len(identities), 'formal N2 anchors')


if __name__ == '__main__':
    main()
