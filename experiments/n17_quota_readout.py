"""Primary-only paired N2 four-arm interactions after global test readout."""
import argparse
import json
from pathlib import Path
import pandas as pd
import n17_common as c
from n17_n1_control import paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--performance', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    gate_path = c.OUT / 'N1_diagnostic_gates.json'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'FROZEN' and gate['N2_quota_training']
    receipt = json.loads((args.performance / 'receipt.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['comparison_sha'] == c.sha(args.comparisons)
    optimizer = gate['N2_optimizer']
    frame = pd.read_csv(args.performance / 'paired_seed_endpoints.csv')
    iid = frame[frame.group == 'N1_common_' + optimizer].copy()
    quota = frame[frame.group == 'N2_quota_' + optimizer].copy()
    keys = ['seed', 'readout', 'metric', 'budget']
    assert not iid.duplicated(keys).any() and not quota.duplicated(keys).any()
    assert set(iid.seed) == set(quota.seed) == set(range(2032, 2040))
    assert set(iid.budget) == set(quota.budget) == {100}
    assert set(map(tuple, iid[keys].to_numpy())) == set(map(tuple, quota[keys].to_numpy()))
    table = iid.merge(quota, on=keys, suffixes=('_IID', '_quota'), validate='one_to_one')
    raw = []; lifts = []
    for row in table.to_dict('records'):
        base = {key: row[key] for key in keys}
        sn0, sb0, sn1, sb1 = [float(row[key]) for key in ('SN_IID', 'SB_IID', 'SN_quota', 'SB_quota')]
        gap0 = sb0 - sn0; gap1 = sb1 - sn1
        effects = {'IID_SB_minus_SN': gap0, 'quota_SB_minus_SN': gap1,
                   'quota_interaction': gap1 - gap0, 'SN_quota_minus_IID': sn1 - sn0,
                   'SB_quota_minus_IID': sb1 - sb0}
        for effect, value in effects.items():
            raw.append({**base, 'optimizer': optimizer, 'effect': effect, 'difference': value,
                        'SN_IID': sn0, 'SB_IID': sb0, 'SN_quota': sn1, 'SB_quota': sb1,
                        **{arm + '_epoch_' + variant: row[arm + '_epoch_' + variant]
                           for arm in ('SN', 'SB') for variant in ('IID', 'quota')}})
        for arm, before, after in (('SN', sn0, sn1), ('SB', sb0, sb1)):
            lifts.append({**base, 'group': 'N2_absolute_quota_lift_' + arm,
                          'estimand': 'fixed real q=' + arm + ': quota minus IID; generic SN=baseline IID, SB=candidate quota',
                          'real_q_arm': arm, 'SN': before, 'SB': after, 'difference': after - before,
                          'SN_epoch': row[arm + '_epoch_IID'], 'SB_epoch': row[arm + '_epoch_quota']})
    raw = pd.DataFrame(raw); summaries = []
    for keys, part in raw.groupby(['optimizer', 'effect', 'readout', 'metric', 'budget']):
        assert len(part) == part.seed.nunique() == 8
        values = part.sort_values('seed').difference
        record = {**dict(zip(['optimizer', 'effect', 'readout', 'metric', 'budget'], keys)), **paired(values)}
        record['budget'] = int(record['budget'])
        summaries.append(record)
    args.out.mkdir(parents=True, exist_ok=False)
    raw.to_csv(args.out / 'four_arm_paired_seed_effects.csv', index=False)
    pd.DataFrame(lifts).to_csv(args.out / 'absolute_quota_lifts.csv', index=False)
    c.dump(args.out / 'seed_summary.json', summaries)
    c.dump(args.out / 'receipt.json', dict(status='COMPLETE', source_sha=c.sha(__file__),
        gate_sha=c.sha(gate_path), comparison_sha=c.sha(args.comparisons),
        performance_receipt_sha=c.sha(args.performance / 'receipt.json'),
        endpoints_sha=c.sha(args.performance / 'paired_seed_endpoints.csv'),
        scope='eight paired training seeds; fixed secondary endpoints; pointwise t intervals and unadjusted secondary sign p; not unique trace mediation; absolute lift field labels denote IID/quota, real q retained'))
    print('N2 paired four-arm readout complete', optimizer)


if __name__ == '__main__':
    main()
