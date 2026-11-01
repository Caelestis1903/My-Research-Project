from experiments.gradient_aug_time_select_20261102.checks import formula_checks


def test_formula_mask_shadow_restore_and_gates():
    assert formula_checks()['status']=='PASS'


def test_confirmation_cap_counts_unmatched_history_baselines():
    from experiments.gradient_aug_time_select_20261102.run import make_jobs,confirmation_budget_reason
    planned=[];stopped=[]
    for dataset in ('movielens','kuairand'):
        group=[job for arm in ('A0','A1','B2') for job in
            make_jobs('confirmation',arm,[dataset],[2023,2025,2026])]
        reason=confirmation_budget_reason(20,len(planned),len(group))
        if reason:stopped.append((dataset,reason))
        else:planned.extend(group)
    assert len(planned)==9 and 20+len(planned)<=40
    assert stopped==[('kuairand','confirmation_stage_cap_12_including_new_baselines')]
    assert confirmation_budget_reason(28,6,6) is None
    assert confirmation_budget_reason(29,6,6)=='global_unique_trajectory_cap_40'
