from experiments.mdmt_five_seed_completion_20261103 import run


def test_frozen_matrix_native_jobs_and_historical_endpoints(tmp_path):
    prepared = run.prepare(tmp_path / 'prepared')
    assert (len(prepared['matrix']), len(prepared['jobs'])) == (40, 30)
    assert len({item['job']['key'] for item in prepared['jobs']}) == 30
    assert {item['job']['seed'] for item in prepared['jobs']} == {2023, 2024, 2025, 2026}
    reused = [r for r in prepared['matrix'] if r['action'] == 'REUSE']
    fixture = tmp_path / 'historical'
    run.write_json(fixture / 'PREPARED.json', dict(prepared, matrix=reused))
    records = run.candidate_records(fixture)
    assert len(records) == 10 and sum(len(r['endpoints']) for r in records) == 120
    entries = run.freeze_entries([r['result'] for r in records])
    assert len(entries) == 60
    for entry in entries:
        assert entry['epoch'] in range(1, 51) and len(entry['model_hash']) == 64
    controls = run.controls(prepared)
    assert len(controls) == 180
    identities = {(r['dataset'], r['arm'], r['seed'], r['model_kind'], r['checkpoint'], r['split'])
                  for r in controls}
    assert len(identities) == len(controls)
    assert run.prepare(tmp_path / 'prepared') == prepared


def verdict_rows(auc=.003, ll=.001, cell=-.001):
    return [dict(seed=seed, delta_macro_auc=auc, delta_macro_logloss=ll,
                 **{f'delta_d{d}t{t}_auc': cell for d in range(3) for t in range(2)})
            for seed in sorted(run.SEEDS)]


def test_five_seed_mean_thresholds_and_independent_loss():
    rows = verdict_rows()
    decision = run.verdict(rows)
    assert decision['macro_auc_pass'] and decision['cell_constraint_pass']
    assert not decision['logloss_decreased']
    assert run.verdict(rows[:4])['status'] == 'INCOMPLETE'
    rows = verdict_rows(auc=.001, ll=-.001)
    decision = run.verdict(rows)
    assert not decision['macro_auc_pass'] and decision['logloss_decreased']
    rows = verdict_rows(auc=.004, cell=-.0011)
    decision = run.verdict(rows)
    assert decision['macro_auc_pass'] and not decision['cell_constraint_pass']
    assert len(decision['violated_cells']) == 6


def test_negative_seeds_and_cross_zero_interval_do_not_veto():
    rows = verdict_rows(cell=0)
    for row, value in zip(rows, [-.004, -.003, -.002, .009, .020]):
        row['delta_macro_auc'] = value
    assert run.stats([r['delta_macro_auc'] for r in rows])['ci95_low'] < 0
    assert run.verdict(rows)['macro_auc_pass']
