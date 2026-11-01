from experiments.gradient_aug_aig_m_20261101.checks import formula_checks,decision_checks,restoration_checks


def test_formula():
    assert formula_checks()['status']=='PASS'


def test_decision():
    assert decision_checks()['status']=='PASS'


def test_RAM_state_and_shadows():
    assert restoration_checks()['status']=='PASS'
