"""CPU algebra reference for the 2026-11-03 plan, NOT a training implementation."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np


def project_cone(h: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x: normals @ x >= 0}; <=4 unit normals."""
    h = np.asarray(h, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, h.size)
    if len(n) > 4 or not np.isfinite(h).all() or not np.isfinite(n).all():
        raise ValueError('Expected finite h and at most four normals')
    if not len(n):
        return h.copy()
    norms = np.linalg.norm(n, axis=1)
    n = n[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(n):
        return h.copy()
    gram, q = n @ n.T, n @ h
    tol = 1e-9 * max(1.0, np.linalg.norm(h))
    best, best_cost = None, np.inf
    for count in range(len(n) + 1):
        for active in itertools.combinations(range(len(n)), count):
            lam = np.zeros(len(n))
            if active:
                idx = np.asarray(active)
                lam[idx] = np.linalg.pinv(gram[np.ix_(idx, idx)], rcond=1e-12) @ (-q[idx])
                if np.max(np.abs((gram @ lam + q)[idx])) > tol:
                    continue
            if np.min(lam) < -tol or np.min(q + gram @ lam) < -tol:
                continue
            candidate = h + n.T @ lam
            cost = float(np.sum((candidate - h) ** 2))
            if cost < best_cost - 1e-14:
                best, best_cost = candidate, cost
    if best is None:
        raise ArithmeticError('KKT enumeration failed; do not silently reset history')
    return best


def repair_history(m: np.ndarray, preconditioner: np.ndarray,
                   pairs: list[tuple[np.ndarray, np.ndarray]],
                   strength: float = 1.0, agreement: float = 0.2) -> np.ndarray:
    """Only old history is projected; current gradient and v are not changed here."""
    if not 0 <= strength <= 1 or len(pairs) > 2:
        raise ValueError('Invalid strength or number of task pairs')
    root = np.sqrt(np.asarray(preconditioner, dtype=np.float64))
    if not np.isfinite(root).all() or np.any(root <= 0):
        raise ValueError('Preconditioner must be finite and strictly positive')
    h = root * m
    normals = []
    for a, b in pairs:
        u, v = root * a, root * b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-12 and nv > 1e-12 and np.dot(u, v) / (nu * nv) >= agreement:
            normals.extend((u / nu, v / nv))
    projected = project_cone(h, np.asarray(normals).reshape(-1, h.size))
    return ((1 - strength) * h + strength * projected) / root


def transport_moment(m: np.ndarray, g: np.ndarray, g_previous_parameter: np.ndarray,
                     step: int, beta: float = 0.9, kappa: float = 0.25,
                     clip: bool = True) -> np.ndarray:
    """g and g_previous_parameter use identical current-step data and include L2."""
    if step < 1 or not 0 <= beta < 1 or not 0 <= kappa <= 1:
        raise ValueError('Invalid step, beta or kappa')
    delta = g - g_previous_parameter
    if clip:
        delta = delta * min(1.0, float(np.linalg.norm(g)) / (float(np.linalg.norm(delta)) + 1e-12))
    return beta * m + (1 - beta) * g + kappa * beta * (1 - beta ** (step - 1)) * delta


def self_test() -> dict:
    tests = {}
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        tests[name] = 'PASS'
    h = np.array([-2.0, 3.0])
    check('cone_one_constraint', np.allclose(project_cone(h, [[1, 0]]), [0, 3]))
    check('cone_opposite_constraints', np.allclose(project_cone(h, [[1, 0], [-1, 0]]), [0, 3]))
    check('cone_empty_identity', np.array_equal(project_cone(h, np.empty((0, 2))), h))
    check('cone_feasible_identity', np.allclose(project_cone([2, 3], [[1, 0], [0, 1]]), [2, 3]))
    check('duplicate_normals', np.allclose(project_cone(h, [[1, 0], [1, 0]]), [0, 3]))
    p = np.array([4.0, 0.25]); pairs = [(np.array([1., 0]), np.array([1., 1]))]
    repaired = repair_history(h, p, pairs)
    check('repair_metric_nonexpansive', np.dot(p, repaired ** 2) <= np.dot(p, h ** 2) + 1e-10)
    check('repair_strength_zero', np.allclose(repair_history(h, p, pairs, strength=0), h))
    check('disagreement_bypass', np.allclose(repair_history(h, p, [(np.array([1., 0]), np.array([-1., 0]))]), h))
    m = np.array([0.3, -0.1]); g = np.array([0.6, 0.5]); old_g = np.array([0.4, 0.4])
    check('transport_kappa_zero_adam', np.allclose(transport_moment(m, g, old_g, 5, kappa=0), .9*m+.1*g))
    check('transport_first_step_no_history', np.allclose(transport_moment(np.zeros(2), g, old_g, 1), .1*g))
    check('same_parameter_no_transport', np.allclose(transport_moment(m, g, g, 5), .9*m+.1*g))
    step = 5; previous_m = (1-.9**(step-1))*old_g
    corrected = transport_moment(previous_m, g, old_g, step, kappa=1, clip=False)/(1-.9**step)
    check('exact_transport_mass_identity', np.allclose(corrected, g))
    rng = np.random.default_rng(20261103)
    for _ in range(100):
        a = rng.normal(size=(4, 8)); x = rng.normal(size=8); z = project_cone(x, a)
        if np.min(a @ z) < -1e-7 or np.linalg.norm(z) > np.linalg.norm(x)+1e-7:
            raise AssertionError('random_cone_properties')
    tests['random_cone_properties_100_cases'] = 'PASS'
    return {'status': 'PASS', 'scope': 'CPU algebra only; no real-model parity, no GPU training, no AUC evidence',
            'tests': tests, 'test_count': len(tests)}

if __name__ == '__main__':
    result = self_test()
    Path(__file__).with_name('DESIGN_CHECKS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
