from experiments.gradient_aug_momentum_20261101.checks import (
    formula_checks, semantic_checks, restoration_checks)
from experiments.gradient_aug_momentum_20261101.analyze import method_verdict


def test_formula_degeneracies_and_none_steps():
    assert formula_checks()['status'] == 'PASS'


def test_target_delta_shadow_and_selection():
    assert semantic_checks()['status'] == 'PASS'


def test_RAM_restore_and_shadow_noninterference():
    assert restoration_checks()['status'] == 'PASS'


def test_method_verdict_requires_development_support_and_preserves_validation_guards():
    def record(means, validation_gate=False, validation_guards=None):
        return dict(validation=dict(gain_gate=validation_gate, guards=validation_guards or [],
            comparisons=[dict(macro_auc=dict(mean=mean)) for mean in means]),
            test=dict(gain_gate=True, guards=[]))
    assert method_verdict('P0',record([-.001]))['status'] == 'NO_PROMOTION'
    assert method_verdict('P1_A3',record([.0001,-.0001]))['status'] == 'NO_PROMOTION'
    assert method_verdict('P0',record([.0001]))['status'] == 'NO_PROMOTION'
    assert method_verdict('P0',record([.0012],validation_gate=True))['status'] == 'GO'
    assert method_verdict('P1_A3',record([.0001,.0002]))['status'] == 'GO'
    verdict = method_verdict('P1_A3',record([.0001,.0002],validation_guards=['worst_cell_auc']))
    assert verdict['status'] == 'TRADEOFF'
    assert verdict['guard_splits'] == dict(validation=['worst_cell_auc'])
