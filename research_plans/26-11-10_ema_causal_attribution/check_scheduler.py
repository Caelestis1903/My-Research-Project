"""Scalar-only admission/state check; never starts a model or process."""
from experiments.ema_causal_scheduler_20261110 import admit, candidate, forecast, bounds, CAP, GUARDIAN_CAP

memory=dict(used=10<<30,total=64<<30)
amount=dict(total=1<<30,guardian_predicate=dict(total=1<<30))
assert admit([0],memory,2<<30,amount)==(0,'admitted')  # GPU1 is ineligible.
assert admit([0],memory,0,dict(amount,total=CAP+1))[0] is None
assert CAP==7<<30 and GUARDIAN_CAP==4<<30
assert admit([0],memory,0,dict(amount,total=CAP))[0]==0
assert admit([0],memory,0,dict(amount,guardian_predicate=dict(total=GUARDIAN_CAP+1)))[0] is None
assert admit([0],memory,0,dict(amount,total=4654122048))[0]==0  # Prior fifth-seed case still fits.
three_kr=dict(total=6965659752,guardian_predicate=dict(total=4238324112))
assert admit([1],dict(used=37700000000,total=67181834240),12360000000,three_kr)==(1,'admitted')
assert admit([0],memory,0,amount,True,True)[0] is None
jobs=[dict(job_id=f'job{i}',dataset='movielens',objective='clean',config={'parameter_count':773470}) for i in range(5)]
status=dict(blocks=[],jobs={},payload_bytes=123)
assert candidate(jobs,status,{'job0','job1'})=='job2'  # No artificial two-worker ceiling.
assert candidate(jobs,status,{'job0','job1','job2'})=='job3'  # Adopted launch is not repeated.
a=forecast(status,jobs,{'job0','job1'},'job2','movielens__clean')
assert a['total']==sum(a['terms'].values())
assert all(a['terms'][k]>0 for k in ('future_retained','worker_state','ipc_and_old_commit','pending_anchor','finalizer_row','finalizer_scratch'))
assert a['terms']['guardian_actual']==123
assert admit([0],memory,64<<30,amount)[0] is None
kr=dict(job_id='kr',dataset='kuairand',objective='clean',config={'parameter_count':1614254,'steps_per_epoch':49})
fifth=forecast(dict(status,payload_bytes=4*bounds(kr)['retained']),[kr],[],'kr')
assert fifth['total']==4654122048 and fifth['guardian_predicate']['total']==4039824232
assert admit([0],memory,2<<30,fifth)[0]==0
print('PASS: independent GPU eligibility, no worker ceiling, payload/RSS denial, adoption, serial attribution, all forecast terms')
