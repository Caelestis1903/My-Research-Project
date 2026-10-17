"""Descriptive context from completed N2 outputs; no new draws or model access."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import n17_common as c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.summary / 'receipt.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['anchors'] == 16
    raw = pd.read_csv(a.summary / 'raw_moments.csv')
    contrasts = pd.read_csv(a.summary / 'trace_contrasts.csv')
    maps = pd.read_csv(a.summary / 'mean_mapping_contrasts.csv')
    mapping_rows = []
    inputs = {}
    for entry in receipt['inputs']:
        path = Path(entry['directory']) / 'adam_mapping.json'
        assert c.sha(path) == entry['files']['adam_mapping.json']
        mapping_rows.extend(json.loads(path.read_text()))
        inputs[str(path)] = c.sha(path)
    mapping = pd.DataFrame(mapping_rows)
    mapping['zero_sampled_data_gradient'] = mapping.mean_gradient_norm == 0
    mapping.to_csv(a.summary / 'mapping_context.csv', index=False)
    blocks = []
    for block, frame in raw.groupby('block'):
        t = contrasts[contrasts.block == block]
        m = maps[maps.block == block]
        row = {'block': block}
        for variant in ('IID', 'quota'):
            x = t[t.contrast == variant + '_SB_minus_SN'].empirical_trace_ratio
            assert len(x) == 16
            row.update({variant + '_q_ratio_min': float(x.min()), variant + '_q_ratio_max': float(x.max()),
                        variant + '_q_ratio_defined': int(x.notna().sum())})
        x = frame[(frame.variant == 'IID')].predicted_between_fraction
        row['IID_between_percent_max'] = float(100 * x.max())
        x = m.mean_update_difference_norm / m.mean_update_difference_mc_rms_upper_bound
        row['q_mean_update_over_MC_RMS_sum_min'] = float(x.min())
        row['q_mean_update_over_MC_RMS_sum_max'] = float(x.max())
        blocks.append(row)
    pd.DataFrame(blocks).to_csv(a.summary / 'block_context.csv', index=False)
    groups = []
    for (anchor, epoch), part in contrasts[contrasts.block == 'ALL'].groupby(['anchor_arm', 'epoch']):
        row = dict(anchor_arm=anchor, epoch=int(epoch), n_training_seeds=4)
        for contrast, frame in part.groupby('contrast'):
            row[contrast + '_ratio_mean'] = float(frame.empirical_trace_ratio.mean())
        groups.append(row)
    pd.DataFrame(groups).to_csv(a.summary / 'ALL_path_stage_context.csv', index=False)
    allraw = raw[raw.block == 'ALL']
    iid = allraw[allraw.variant == 'IID']
    allmap = maps[maps.block == 'ALL']
    ratio = allmap.fixed_denominator_mean_update_difference_norm / allmap.mean_update_difference_norm
    p_train = json.loads((Path(receipt['inputs'][0]['directory']) / 'config.json').read_text())['p']
    zero = mapping[(mapping.block.isin(['C_shared_STAR', 'C_star_mlp'])) & mapping.zero_sampled_data_gradient]
    zero.to_csv(a.summary / 'zero_sampled_gradient_blocks.csv', index=False)
    c.dump(a.summary / 'context_receipt.json', dict(
        status='COMPLETE_DESCRIPTIVE', source_sha=c.sha(__file__), summary_receipt_sha=c.sha(a.summary / 'receipt.json'),
        inputs=inputs, test_read=False, additional_draws=0,
        between_percent_ALL_IID_range=[float(100 * iid.predicted_between_fraction.min()), float(100 * iid.predicted_between_fraction.max())],
        empirical_over_predicted_ALL_range=[float(allraw.empirical_over_predicted_trace.min()), float(allraw.empirical_over_predicted_trace.max())],
        fixedD_over_native_q_mean_contrast_ALL_range=[float(ratio.min()), float(ratio.max())],
        SB_weight_second_moment=float(3 * np.square(p_train).sum()),
        zero_sampled_gradient_unique_anchors=int(len(zero[['seed', 'anchor_arm', 'epoch']].drop_duplicates())),
        scope='Four training seeds reused across path/stage. MC RMS sum is not a CI. Zero sampled gradient is not by itself a claim about every population row or about module irrelevance. Same-bank trace decomposition is not an independent oracle.'))


if __name__ == '__main__':
    main()
