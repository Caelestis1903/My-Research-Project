# C05-R2 prelaunch implementation correction

Aquinas / `01a1ea0c-d126-75f3-ad8a-78914c15e791`, gpt-6.1-sol/high, `fork_context=false`. Original parent: `01a1dfb8-605b-7430-8462-5fd3da0b048f`. No descendants; permanent return after this handoff.

The corrected **unfrozen candidate requires NEW distinct R2 review**. This is a coder report, with no RUN release. Only runner.py and these two R2 reports were written.

The actual production bug was at old runner.py:1114–1115: `admission()` already supplied `sample_id`, so `dict(..., sample_id=current['sample_id'], **decision)` raised `TypeError: dict() got multiple values for keyword argument 'sample_id'` before `_launch`. Evaluating the exact original production expression extracted by AST from the parent's read-only byte copy reproduced that exception with historical SAMPLE `2930478:0:1:1`.

A small pure `admission_row()` now constructs that same row using the raw decision's sample ID. Production and the single narrow fixture call the helper. Every decision field remains present; measurements, admission predicate, reasons, incoming/remaining bytes, envelope, pending selection, queue semantics, decision hash/write and launch ordering are unchanged. AST comparison proves only the production row assignment changed in `_schedule`; the old `finite_checks` differs only by invoking/returning the added fixture. All other existing definitions and top-level statements are identical.

| Identity | Before | After |
| --- | --- | --- |
| runner.py | `7515c4c20aa26828309163fa825404ef93d02cc7e807a97f7bf74e504827b567` | `0aeb48eb66dda88ef579b3fc9abbaa3b850105d5de44bd867be1aa430f59c649` |
| Candidate source55 | `78604351e97221abd78318aa1f10e31aacfcc6c97a0d5d0e99e965d87a64bd80` | `f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d` |

The baseline `/tmp/phase0-3-cycle05-preR2-runner.py` was read-only and remains unchanged. The exact55 path set is unchanged. All **54 other current source hashes** match the frozen R5 receipt and R1 reviewed source; tables.py remains `79d9673e032a4b0eab4bff575fa9594e11bfcb972975a68da53a3a5c6b804965`. All55 old R5 snapshot files still match their original receipt. The JSON report contains all55 candidate hashes, the54 unchanged paths and29 protected design/report/receipt/telemetry hashes verified unchanged.

Source48 remains `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`, receipt `10f3000ae43e2a394b71cdb2febc291888f06bf216dd99c05ddd7b842186fec3`; all48 current members match. External installed Adam remains `b099132d262d6cb16eefa48ab4bae789a0655cafb8571e421ef4fe3d2ff7c34c`. Specification `a1f61ae81535bb33873da72185d9400187c5b988e81c0d3a5019c5505e821af9`, design7 `9a7e826794f4cdc3fcc7c437e1462149a1dc9372b5d5e3a38981f69fb3af299c` and research6 `bfd6230c9cae8971a3a8d793027fc54ba6209d193429340107bf64bb22dc4caf` remain exact. Original reports/FAIL and R1 PASS remain intact. Unchanged core/runtime/R1 proofs are reused with these byte/AST bindings; the previously uncovered production construction is checked here.

`admission_row_check()` passed once, constructing two rows through actual `admission()` and actual production `admission_row()`: eligible with no pending allocation and denied with exact pending identity. Its required numeric fields equal the failed GPU0 SAMPLE, including actual sample ID; CUDA-usable capacity25307578368 and incoming historical bootstrap envelope2027945984 are unchanged recorded inputs. Only the fixture monotonic timestamp is refreshed. Historical numbers are fixture evidence, not a current launch sample. All raw decision keys/values, sample ID, incoming/remaining bytes, predicates, envelope/pending identities, job/purpose/queue/device metadata and exact row key set passed. GPU1's failed-controller log has one dictionary record and no SAMPLE.

The narrow check took **0.002738898037932813s elapsed / 0.002737618000001163s process CPU**. One RAM syntax compile passed. The final verification script, including imports and identity checks, took 6.110483410069719s elapsed / 7.994229814s CPU; tool process wall was 8.936719816s. These are measured final-process costs, not an estimate of total session CPU. Original-expression reproduction and byte reads/hashes are additional nontraining CPU work. Fail-if-called instrumentation recorded0 NVML constructions, hardware queries, train calls, worker launches, owner IPC, model forwards, backwards and Adam steps. No model or optimizer was constructed; no live/virtual Adam, EMA, engineering or scientific training occurred.

All project Python used `PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B`; compilation stayed in RAM. The standalone narrow fixture can be repeated from W:

```sh
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -c 'from experiments.cell_mass_bce_20261117.runner import admission_row_check; print(admission_row_check())'
```

Both workspace controls were ACTIVE/manual_stop_requested=false/termination_reason=null/stop_after_current_cycle=5/cycle06_authorized=false; all four STOP_AUTONOMOUS sentinels were absent.

The parent's concrete [PRELAUNCH_R2_DECISION.json](PRELAUNCH_R2_DECISION.json) authorizes this source correction under specification§7.2; §7.5 telemetry/admission semantics remain unchanged. The read-only R5 TECHNICAL_PAUSE records controller2930478's duplicate-key error. Parent [empty-owner acceptance](/tmp/phase0-3-c05-01a1e9a9/attempt1/PARENT_PRELAUNCH_EMPTY_ACCEPTANCE.json), SHA `8b793b1ce8e60098b6e7bdff7a0ca3248ece3a67beb37f4b32c495a08a2e5bd9`, records owner2926581/session16855 with0 registered jobs/retained states/retained bytes/ACKs and no physical worker records. Those are parent-verified facts; this coder queried no owner. The parent retains liveness polling.

After NEW distinct R2 review, failed-runner handoff/closure and renewed direct empty proof, only the parent executes its already-recorded unchanged quarantine/re-prepare decision and explicitly binds the SAME empty owner to the new reviewed source/new job keys with before/after acceptance. No coder quarantine, prepare, freeze, rebinding or restart occurred. The current candidate intentionally differs from the old frozen source identity; old snapshot/receipts must never be relabeled.

The scientific matrix remains U only, ML/KR×seeds2022..2026×50,40960FP32,500 epochs/17250 updates;30 numeric controls/1530 files; RAM-only/source Adam/fixed EMA/`gate.detach`. C04 published `cbccf5ae234e46bc3f976467de2b3327139f021b` is preserved. The parent completes C05 report/review/publication/delivery then USER_STOPPED, with no C06.

Exact diff against the unchanged pre-R2 byte copy:

```diff
--- before/experiments/cell_mass_bce_20261117/runner.py
+++ after/experiments/cell_mass_bce_20261117/runner.py
@@ -193,6 +193,11 @@
         sample_id=current.get('sample_id'),current_memory_fraction=memory,current_power_fraction=power,
         incoming_measured_full_process_bytes=incoming_measured_full_process_bytes,already_resident_bytes=already_resident_bytes,
         remaining_measured_bytes=remaining,test_accessed=False)
+
+
+def admission_row(j,queue_index,device,decision,pending_identity,envelope):
+    return dict(time=time.time(),purpose=j['purpose'],queue_index=queue_index,key=j['key'],device=device,
+        pending_identity=pending_identity,envelope=envelope,**decision)
 
 
 def pending_matches(pending,record,*,alive):
@@ -1111,8 +1116,7 @@
                         decision=admission(current,incoming,pending=any(v['device']==dev for v in pending.values()))
                         if not decision['predicates']['fresh_known']:
                             current=sample(dev); decision=admission(current,incoming,pending=any(v['device']==dev for v in pending.values()))
-                        row=dict(time=time.time(),purpose=j['purpose'],queue_index=i,key=j['key'],device=dev,
-                            sample_id=current['sample_id'],pending_identity=[v for v in pending.values() if v['device']==dev],envelope=envelope,**decision)
+                        row=admission_row(j,i,dev,decision,[v for v in pending.values() if v['device']==dev],envelope)
                         row['decision_id']=p.digest(row); p.append_json(a/'admission.jsonl',row)
                         if not decision['eligible']:
                             if any(x!='physical_fit' for x in decision['reasons']): break
@@ -1217,7 +1221,31 @@
 
 
 
+def admission_row_check():
+    # Required fields from controller_2930478_gpu0 SAMPLE; only fixture freshness is renewed.
+    current=dict(sample_id='2930478:0:1:1',known=True,device=0,used_bytes=20985675776,
+        nvml_total_bytes=25769803776,power_watts=192.963,power_limit_watts=350.,host_fraction=.5195776199158447,
+        cuda_usable_total_bytes=25307578368,monotonic_end=time.monotonic())
+    incoming=2027945984
+    j=dict(purpose='engineering_smoke',key='fixture-startup')
+    envelope=dict(path='fixture-envelope.json',sha256='fixture-envelope')
+    held=dict(pid=33,key=j['key'],device=0,expected_attempt=1)
+    for pending in ([],[held]):
+        decision=admission(current,incoming,pending=bool(pending)); raw=copy.deepcopy(decision)
+        row=admission_row(j,0,0,decision,pending,envelope)
+        assert row['sample_id']==current['sample_id'] and all(row[k]==v for k,v in raw.items())
+        assert decision==raw and row['eligible']==(not bool(pending))
+        assert row['incoming_measured_full_process_bytes']==row['remaining_measured_bytes']==incoming
+        assert row['already_resident_bytes']==0 and row['predicates']['no_pending']==(not bool(pending))
+        assert row['envelope'] is envelope and row['pending_identity'] is pending
+        assert row['purpose']==j['purpose'] and row['key']==j['key'] and row['queue_index']==row['device']==0
+        assert set(row)==set(raw)|{'time','purpose','queue_index','key','device','pending_identity','envelope'}
+    return dict(status='PASS',sample_id=current['sample_id'],admission_rows=2,raw_decision_preserved=True,
+        incoming_envelope_pending_preserved=True,no_gpu_queries=True,test_accessed=False)
+
+
 def finite_checks():
+    startup_row=admission_row_check()
     from unittest.mock import patch
     from types import SimpleNamespace
     import sys
@@ -1357,7 +1385,7 @@
                 assert peak==12 and p.read_json(ref['path'])['registry_prefix']==prefix
     return dict(status='PASS',groups=7,named_query_retry_attempts=[2,3],terminal_cleanup_failures=2,
         stale_pending_rejected=True,ram_only_restore=True,storage_transcripts=2,actual_core_integration_calls=1,actual_core_diagnostic_points=1,actual_core_virtual_steps=4,
-        numeric_registry_prefix_fixture=True,no_gpu_queries=True,test_accessed=False)
+        numeric_registry_prefix_fixture=True,startup_row=startup_row,no_gpu_queries=True,test_accessed=False)
 
 
 def main():
```

Skipped broad already-passed suites, cache unpickling, GPU queries/training, owner IPC, attempt filesystem moves, source freeze/preparation, runtime retry, Test and research analysis. Fresh R2 review remains required; Aquinas returns permanently for parent closure.
