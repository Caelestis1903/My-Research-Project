"""CPU algebra checks for a planning document; not a model-performance experiment."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20261102)
results = {}
K, P = 24, 11
w = rng.uniform(.01, 1, K); w = .87*w/w.sum()
demand = rng.uniform(0, 1, K)
q = .25/K + .25*w/w.sum() + .5*w*(.05+demand)/(w*(.05+demand)).sum()
H = rng.normal(size=(K, P)); g = rng.normal(size=(K, P))
A = w @ H
est = np.stack([A+w[s]/q[s]*(g[s]-H[s]) for s in range(K)])
results['cv_conditional_unbiased_max_abs'] = float(np.max(np.abs(q@est-w@g)))
assert np.allclose(q@est, w@g, atol=1e-12)
alpha = .01
expected = np.zeros_like(H)
for s in range(K):
    h = H.copy(); h[s] += alpha/q[s]*(g[s]-h[s]); expected += q[s]*h
results['eccm_exposure_correction_max_abs'] = float(np.max(np.abs(expected-((1-alpha)*H+alpha*g))))
assert np.allclose(expected, (1-alpha)*H+alpha*g, atol=1e-12)
results['maximum_history_gain'] = float(np.max(alpha/q))
assert np.all(alpha/q <= 1)
assert np.all(q >= .25/K-1e-15)
assert np.all(w/q <= 4*w.sum()+1e-12)
greal = rng.normal(size=P); wdtheta = rng.normal(size=P)*1e-4
mr = .9*rng.normal(size=P)+.1*(greal+wdtheta)
aux = w@(H/(1-.99**7))
assert np.array_equal(mr/(1-.9**7)+0*aux, mr/(1-.9**7))
results['lambda_zero_numerator_exact'] = True

def edges(labels):
    labels = set(labels)
    return [(t,p,n) for t in (0,1) for p in sorted(labels) for n in sorted(labels) if p[t]==1 and n[t]==0]

ml = edges([(0,0),(1,0),(1,1)]); kr = edges([(0,0),(0,1),(1,0),(1,1)])
assert len(ml)==4 and len(kr)==8
assert (0,(1,0),(0,0)) in ml and (1,(1,1),(1,0)) in ml
results['per_domain_boundary_counts'] = {'ml':len(ml),'kr_max':len(kr)}
# Rejected reference transport: constant true gradient is spuriously amplified.
beta=.9; h_old=0.; u=0.; bad=[]
for k in range(1,4):
    A=h_old; residual=1.-h_old; u=beta*u+(1-beta)*residual
    bad.append(A+u/(1-beta**k)); h_old=beta*h_old+(1-beta)
assert bad[1]>1.
results['rejected_reference_transport_constant_gradient'] = bad
# A post-update table must not be substituted for the pre-update CV table.
wrong=[]
for s in range(K):
    h=H.copy(); h[s]=.9*h[s]+.1*g[s]
    wrong.append(w@h+w[s]/q[s]*(g[s]-h[s]))
wrong_error=float(np.max(np.abs(q@np.array(wrong)-w@g)))
assert wrong_error>1e-8
results['wrong_update_order_detected_error'] = wrong_error
results['status']='PASS_ALGEBRA_ONLY_NO_TRAINING'
out=Path(__file__).with_name('MATH_CHECK_RESULTS.json')
out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(out.read_text())
