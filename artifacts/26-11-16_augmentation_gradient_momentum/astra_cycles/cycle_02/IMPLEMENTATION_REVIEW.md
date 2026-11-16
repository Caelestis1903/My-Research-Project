REVIEW REPORT

1. **Verdict: FAIL.** One unresolved **MAJOR** blocker (R1). Return to a fresh coder, then a fresh distinct reviewer. No implementation corrections were made.

Reviewer: `01a1e411-7a58-7280-963f-9b8a8afa8fbc` (actual `CODEX_THREAD_ID`); distinct from both permanently closed coders `01a1e3f5-7f1c-7ad2-9512-5c86d72c033e` and `01a1e3f5-7f2c-7262-945c-10b1626cfb40`. Scope: **CODE_REVIEW**, file handoff only. Applied ponytail FULL and stop-that-shit CHANGE.

Reviewed source: `d3f518b8d23cbafa9be0b954de995d0edfeaf12ebab3836cd4a723d3eda9d0a0` (87 files). Accepted design: `9e5265df52809e45ac186d5c5a4a2b80ac96e4c952ded35eb92f76c1efcccfb4` (7 files). Protected C1: `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1` (79 files). All were independently matched to actual bytes. The additional TV `tables.py` import is part of the actual closure, not a C1 source edit. Prior TV/C1 numeric journal/result inventories match the coder's before inventory; ten exact CLEAN result byte hashes match. Both stop controls were clear. No old endpoint or RAM owner was contacted.

2. **Requirement-by-requirement verification**

All seven design files, both coder reports/CPU receipts, acceptance/handoff/candidate receipts, current policy files and the actual seven-file package were inspected. Relevant frozen helper/model/evaluator/storage and historical CLEAN source paths were traced. PASS_STATIC means source inspection; it does not report a GPU or RAM execution.

| Requirement | Status | Verification |
| --- | --- | --- |
| Design and scope | PASS | All seven accepted design files, both implementation reports and both CPU receipts inspected; only the three fixed objectives are implemented. No VAT/SAM, HPO, extra baseline training, or efficacy verdict. Evidence: `DESIGN_SPECIFICATION.md 3-10; README.md:1; protocol.py:60-71,297-320`. |
| Pair completion and controls | PASS | R/B/M implement exact S+2*pi0*pi1*Q, centered differentiable population variance and .75C+.25A; no pair tensor/sampler. Literal fixtures independently define ordered self-inclusive pairs and M variance removal. Evidence: `core.py:67-112; checks.py:42-154`. |
| Mass, rows, classes, tails | PASS | Natural cell weight n/(2B), per-class BCE coefficient (1-lambda+lambda*pi)/(2B), unit total and half task masses; singleton variance zero; absent statistics null; actual B retained. Evidence: `core.py:34-56,82-110; runner.py:435-451`. |
| Reference p=.5 derivatives | PASS | Algebra yields CLEAN score derivatives for all three arms; inspected fixed FP64/FP32 fixtures cover unequal classes and true tail denominators. Existing passing receipts reused. Evidence: `checks.py:157-178; core.py:99-111`. |
| Direct source branches | PASS | Priority lambda_zero then all_single_class is label-derived before candidate forward. Frozen helper owns the one forward/backward/Adam/EMA sequence. C=A=L and unavailable statistics remain null. Evidence: `core.py:59-64,153-178; frozen TV core.py:145-150`. |
| Graph and optimizer paths | PASS | Ordinary model(x), source detached gate inputs and trainable gates, one scalar backward and source Adam/EMA. No hooks/probes/gradient substitution; grad=None and explicit-zero activity preserved. Finite gradients checked before ordinary optimizer mutation. Evidence: `core.py:179-198; checks.py:283-362; frozen reference.py:156; auxiliary protocol.py:59-61`. |
| Core API and diagnostics | PASS | Signatures and required keys agree with runtime callers. Prefix grouping, actual gradient/update norms, optional temporary clones, no returned tensors or persistent candidate state. Existing core checks exercise diagnostic identity using independent model copies. Evidence: `core.py:131-198; runner.py:440-454; checks.py:202-279`. |
| Smoke identity restoration | FAIL | R1: populated optimizer baseline is aliased and mutated between comparisons. Existing core identity checks use separate models and do not exercise this runtime helper. Evidence: `runner.py:256-258,505-526,546-553`. |
| CLEAN reuse and baselines | PASS | Ten explicit CLEAN result byte hashes independently verified. Source traces preserve CLEAN BCE plus zero auxiliary, source Adam/EMA, initialization and private seed+11000 loader. Runtime verifies 52 shared dependency bytes, ten initialization identities and 500 reference epoch orders, with all 1500 new arm-epoch comparisons. Evidence: `protocol.py:235-285; runner.py:395-401,428-430; pilot core.py:170-192; pilot run.py:29-50`. |
| Budget and selection | PASS | Exactly 30 new jobs, seeds 2022-2026, 50 epochs, batch40960 with tails, 1500 epochs and 51750 useful updates. Independent online/fixed_ema strict maxima retain earliest ties and same-state LL/cells; fixed E10/E30/E50 remain descriptive. Evidence: `protocol.py:60-71,221-232,308-318; runner.py:424-481`. |
| Data and evaluation | PASS_STATIC | Immutable existing cache only, relocated raw hashes and train/validation identities checked. Frozen evaluator uses unscaled natural validation metrics, not the algorithm objective. Undefined six-cell endpoints pause. No reviewer data-cache load or Test operation. Evidence: `protocol.py:174-186; runner.py:320-329; frozen ema02/evaluate.py:51-61`. |
| Source closure and preservation | PASS | Independently matched candidate 87-file closure, seven-file bundle and protected79 aggregate. Extra frozen TV tables.py is a real transitive import via TV protocol. Whole-byte previous TV/C1 journal/result inventories match runtime before_inventory; old states/owners not contacted. Evidence: `protocol.py:74-111,189-202; PARENT_SOURCE_REVIEW_CANDIDATE.json; RUNTIME_CPU_CHECKS.json`. |
| Admission | PASS_STATIC | Per-card current memory/power strict85 and host strict90; physical <= CUDA-usable100; pending real allocation ACK followed by resampling. No predicted power/projected85 memory/projected90 host/card coupling/process cap. Existing 13 passing CPU fixtures inspected. Evidence: `runner.py:44-80,675-713`. |
| Profiles and both cards | PASS_STATIC | Maximum C2 full process or torch peak plus measured context, raised by live observations. Compatibility uses name/compute capability/usable capacity, not GPU index: compatible cards can share a measured dataset/arm profile even if only one ran that smoke case. Missing compatible profile pauses rather than silently disabling a card. Actual hardware identity remains the next stage. Evidence: `runner.py:121-182,599-610,671-703`. |
| Storage routing and transactions | PASS_STATIC | Absolute C2 root enforced. Frozen TV serve actual register/allocated/commit/get/recover/complete contracts traced. Numeric epoch precedes commit; subsequent get verifies retained continuation and component hashes; ACK binds file bytes. Same-owner restore and epoch50 finalization retain selected pairs. R1 concerns repeated smoke restore, not server root isolation. Evidence: `protocol.py:123-146,364-404; runner.py:185-258,349-374,644-659; frozen TV runner.py:62-143`. |
| Retention and credentials | PASS_STATIC | Parent alone starts foreground storage after code PASS. No owner launcher or old endpoint calls in the new package. Tensor save is BytesIO only; private guardian mode0600 is checked; numeric owner/export receipts omit authkey. Handoff requests parent acceptance and lists30 pairs/1500 commits; context closure remains parent-owned. Evidence: `runner.py:185-194,221-244,774-797; README.md:25-40`. |
| Exports and costs | PASS_STATIC | Exactly30 manifest jobs plus10 explicit CLEAN references. New scientific epochs resolve through RAM journal/local ACK; orphan files stay physical evidence. Count assertions cover320 endpoints/4000 estimator curves/24000 cell curves/51750 useful batches. Row-weighted losses, null/statistic denominators, diagnostics, all physical attempts and seed-level paired arithmetic are retained without scientific verdict. Evidence: `protocol.py:386-418; runner.py:261-317,449-473,613-641; tables.py:41-65,79-210`. |
| Lifecycle and Test seal | PASS_STATIC | Both workspace stop controls read and clear during review; admission/epoch boundaries use them. Technical failure pauses. No hidden science gate or outer-loop cap. Owner acceptance and future eligible Test gate remain original parent responsibilities. Evidence: `protocol.py:136-142; runner.py:424-426,476-487,662-723; accepted design sections7-11`. |

3. **Bugs/deviations found**

**R1 — MAJOR, blocking: repeated smoke restores mutate their Adam baseline.** Locations: [runner.py:257](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/runner.py:257), [runner.py:508](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/runner.py:508), [runner.py:520](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/runner.py:520), [runner.py:551](/home/xzf/Code/phase0-3_tvg_rm_20261116/experiments/natural_quadratic_ranking_20261116/runner.py:551).

`smoke_identities` captures one `base`, then `reset()` repeatedly calls `restore(..., base, ...)`. `restore` passes `base['optimizer']` directly to `Optimizer.load_state_dict`. The loaded step tensor aliases the saved tensor. Adam increments it in place. After the first useful full-batch update, the tail identity baseline has populated Adam state: the diagnostic-off operation changes the saved counter from 1 to 2; diagnostic-on starts from that changed counter and advances it to 3. The comparison at line 520 fails, preventing the mandatory six-case smoke and formal RUN gate.

A focused CPU execution of the actual helper reproduced `AssertionError`, post-step 3, and confirmed pointer aliasing. This reaches CUDA too: installed PyTorch `adam.py:165-175` deliberately keeps noncapturable/nonfused step tensors on CPU; `optimizer.py:780-784` returns those saved tensors unchanged. CPU moments also alias during CPU restores. The core coder's separate-model diagnostic checks do not exercise this runtime repeated-restore path.

Minimal fix: give each smoke reset fresh optimizer tensors before loading; keep the saved baseline unchanged. No change to Adam, the objective, precision, batches, seeds or research design is needed. An in-memory wrapper copying only the optimizer subdict made the same populated-state probe pass all three identity comparisons and restore step 1. **That wrapper was a disposable check; frozen source remains unfixed.**

Focused retest for the fresh coder: first perform one useful update, then call the real `smoke_identities`; require diagnostic-on/off, lambda-zero and all-single-class equality, unchanged saved baseline, and final restored original counter. Bind the repaired source to a fresh review. The real six-case smoke remains the next stage after code PASS.

4. **Corrections made and checks**

Only these review reports were written. No code, design, frozen dependency, source receipt, old artifact, owner or state was edited. The registered five core groups and 13 admission fixtures have passing preserved coder receipts; their implementations were inspected and the broad suites were not rerun.

Reviewer project-Python commands ran with `PYTHONDONTWRITEBYTECODE=1`. Exact executed heredocs and outcomes are stored as Q1–Q3 in the JSON report:

- Q1: independent source/design/protected-byte and prior-disk-inventory verification, ten CLEAN result byte checks, reference counts, stop controls — PASS.
- Q2: populated-Adam execution of actual `smoke_identities`, plus pointer/source-policy inspection — reproduced R1. The probe catches its expected AssertionError, so the command exits 0.
- Q3: same small fixture with a temporary optimizer-copy wrapper — all three identities PASS, final restored step 1. Afterward, an optional O-workspace counterpart-file comparison exited 1 with FileNotFoundError; that file is absent there. Source closure/preservation evidence comes from Q1, not that optional comparison.

Reproduction (the complete actual Q2 invocation):

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python - <<'PY'
import copy,inspect,json
import numpy as np
import torch
from experiments.natural_quadratic_ranking_20261116 import runner as r, protocol as p
class Scores(torch.nn.Module):
 def __init__(self):
  super().__init__(); self.score=torch.nn.Parameter(torch.tensor([.2,-.3]))
 def forward(self,x): return self.score.sigmoid().expand(x.shape[1],2)
model=Scores(); ema=copy.deepcopy(model).requires_grad_(False)
c=p.config('movielens',p.ARMS[0]); opt=p.optimizer(model,c)
x=torch.tensor([[1.,2.,3.,4.],[0.,0.,0.,0.]])
y=torch.tensor([[0.,1.],[1.,0.],[0.,1.],[1.,0.]])
loader=torch.Generator().manual_seed(13022); ctx=dict(dataset='movielens',seed=2022,epoch=1,batch=2,global_step=2)
r.train_step(model,opt,ema,x,y,c,ctx,np.arange(4),diagnose=False)
try:
 r.smoke_identities(model,opt,ema,loader,torch.device('cpu'),x,y,np.arange(4),c,ctx,r.new_costs())
except AssertionError as error:
 print(json.dumps(dict(probe='actual_smoke_identities_after_one_real_Adam_update',result='FAIL_REPRODUCED',exception=type(error).__name__,post_step=int(opt.state[model.score]['step']),gpu_queries=0,ipc_calls=0)))
else:
 print(json.dumps(dict(probe='actual_smoke_identities_after_one_real_Adam_update',result='PASS')))
base=r.identity_state(model,opt,ema,loader,torch.device('cpu'))
r.restore(model,ema,opt,loader,base,torch.device('cpu'))
state_id=next(iter(base['optimizer']['state']))
print(json.dumps(dict(cpu_step_alias=opt.state[model.score]['step'].data_ptr()==base['optimizer']['state'][state_id]['step'].data_ptr(),capturable=opt.param_groups[0]['capturable'],fused=opt.param_groups[0]['fused'])))
print(inspect.getsource(torch.optim.Optimizer._process_value_according_to_param_policy))
PY
```

5. **Remaining risks**

R1 is unresolved. Actual CUDA full/tail identity, full-process residency, both-card hardware compatibility, validation/RAM transport/recovery and separate smoke/resource signoff remain unperformed next-stage work. These pending measurements are not additional code-review failures. No training experiment, GPU query, IPC, Test operation, git action, publication or efficacy analysis was performed. Historical live owner/state preservation is supported here by unchanged disk receipts and no contact, not a new live-RAM verification.

6. **Whether experiments are safe to run**

**No: repair R1 and obtain fresh code PASS before launching the required engineering smoke.** Formal30 additionally requires that real six-case smoke and separate fresh source-bound smoke/resource signoff. Original-parent foreground storage ownership, parent acceptance before runner closure, all existing C1 states and the Test seal remain in force.
