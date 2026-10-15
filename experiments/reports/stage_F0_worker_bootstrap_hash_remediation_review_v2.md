# REVIEW REPORT: F0 worker bootstrap amendment remediation v2

## 1. Verdict

**PASS WITH FIXES — narrow remediation only. RELEASE NO-GO.** No blocking implementation defect remains. The fixes are the executable isolated acceptance/replay tests added in v2 and the evidence/wording clarifications recorded below. This verdict does not authorize production release or training.

Independent reviewer: `/root/f0_v14_fix_review_v2`.

Design v1 SHA256: `252046a68049d108a684f20aaf3d71be9dccb48677495ccda19a509b78daff95`.

Coder report v2 SHA256: `491bf69c0bdbc8708e2e9bbd050d3213270e643e460e66369d9a165c464140ea`.

Original FAIL review SHA256: `189d67d4b4a635c93a47ee0633ea384110e1f53ccb23ca3a3a282cd1d649bc86`.

Independent full per-file evidence: `experiments/reports/stage_F0_worker_bootstrap_hash_remediation_review_v2_evidence.json`, SHA256 `a1b3c2ed82db8991accaa85a104651f7c1173657d9d309f546408408292e9104`.

## 2. Requirement-by-requirement verification

| Requirement | Independent finding |
| --- | --- |
| Exact amendment predicate | PASS. Actual `validate_current_gate` requires a Mapping, APPROVED status, PASS/PASS WITH FIXES verdict, the frozen scientific digest, and complete runtime digest. Missing/invalid metadata produces its specific PermissionError. No runner import or scientific dependency was added. |
| Preserve stricter runner and separate hash domains | PASS. Runner production source and worker-control dispatch retain their frozen hashes. Runtime is still `stage_F0_queue.code_hash(require_tests=True)` over 28 files; the 21-file control inventory remains independently compared. Runtime digest `7d6bbd9174d3d26c4dce958173444e9ba9778313166aefbc3a32c30f6e9aceb3`; control digest `340266d56dc55fd44cdf01a3fb83be6f22e202313e3f8b645b6715eb99d4570a`. |
| Observed two-field amendment and invalid metadata/hash cases | PASS. Actual tests exercise both missing fields, absent status/verdict individually, PENDING/REJECTED status, PENDING/FAIL verdict, wrong scientific/runtime digest and control-digest substitution. Both real layers reject them. |
| Valid isolated real v1.4 graph for both accepted verdicts | PASS. The formerly skipped fixture now copies bytes with `shutil.copy2`, restores historical source-pointer/status evidence only inside its temporary root, creates/validates a real source snapshot, performs real administrative migration, and builds the postmigration/prelaunch graph. Actual `validate_current_gate(require_release=False)` and runner helper accept both PASS and PASS WITH FIXES. Neither validator/helper is mocked. Dataset/runner-claim sentinels remain untouched during amendment probes. No required fixture is skipped. |
| Current rehold and historical review remain rejected | PASS. The production read-only test updates only an in-memory gate to get past amendment checks; the complete validator still rejects historical stale review/provenance. Production gate is not rewritten. The isolated valid fixture also tests stale legacy review and held/fresh state drift. |
| Old identity and different-nonce release negatives | PASS. Actual new test copies current runs/gate into a temporary root and invokes real `commit_principal_release` with the exact old immutable release identity and a different nonce. Both reject. Hold, gate, release artifact, release pointer and ledger byte hashes remain unchanged after each call. Ledger stages remain exactly PREPARED, RELEASE_ARTIFACT_DURABLE, HOLD_RELEASED, COMMITTED. The real code rejects old-identity replay at changed active source hold and different nonce at ledger identity consistency. No write-capable negative release API is exercised against production. |
| Scope and complete code identities | PASS within the independently checked inventories. All 53 file identities in coder v2 tables match actual bytes. Against the frozen canonical gate's complete control inventory, exactly the three permitted source/test files differ. No other control source differs. Current code inventories remained identical throughout this review. Git history is unavailable; no unrecorded historical textual diff is claimed. |
| Protected production preservation | PASS for current before/after suite evidence and seven specified historical protected identities. Reviewer retained full exact manifests before/after: 2 queue files, 155 statuses, 1,193 files in 115 canonical run directories, 535 checkpoints. All per-file entries and hashes agree. Migration ledger/source snapshot additionally agree. The seven gate/hold/failure/stderr/search/release-ledger/release hashes match design and coder report. |
| Inventory honesty | PASS with an explicit limitation. Coder v2 honestly states original preimplementation broad manifests were not recorded and are unavailable. Its pre/post final-suite summaries independently match actual current files. This satisfies the original FAIL review's instruction to report absence honestly and obtain new isolated verification evidence; it does not reconstruct missing preimplementation evidence. Reviewer retains full current manifests instead of claiming historical knowledge. |
| Applicable regressions | PASS after isolated capacity retest. Targeted actual result: 48 passed, 1 warning, no skips. Related groups produced 83 passed and one shared-scheduler capacity failure; that exact test passes with an independent scheduler root. See commands/results below. |
| Scientific formulas, tensor/mask/routing/weighting semantics, gradients and baselines | No production algorithm change is introduced by this metadata predicate. Runner/queue/migration source hashes remain frozen. Runtime scientific file hashes agree with coder inventories and remain constant during review. No scientific conclusion is drawn from synthetic regression training. |
| Approval record / diagnostic candidate / release | Correctly deferred. This review approves only the narrow implementation. Any subsequently created amendment/candidate must bind the actual final hashes and this independent verdict and retain nonauthorizing flags. Existing production v1.4 gate still fails current validation, intentionally. A separately reviewed successor protocol and distinct release are required. |

## 3. Bugs/deviations found

No CRITICAL or remaining MAJOR implementation error was observed. All three original blocking findings now have the required executable verification or honest evidence disposition.

**MINOR — coder wording overstates absence of training/claims.** The report's blanket statements that tests made no canonical claim or performed no training must be read with the following exact correction: amendment probes load no dataset and create no claim; the existing broader real graph fixture later creates an isolated temporary attempt-00 canonical claim; existing runner regressions train/resume synthetic CPU models in temporary directories. No production canonical claim, production attempt/receipt, dataset-backed research training, GPU research training, formal experiment or checkpoint recovery authorization was created. The synthetic formal/resume tests are regression fixtures and confer no production authorization. This correction is recorded here without rewriting the reviewed coder report.

**MINOR — reviewer invocation collision, resolved.** I initially ran two independent pytest groups concurrently using the same scheduler root. The existing launcher capacity test intentionally reserves four additional slots besides its pytest slot, so the other pytest process caused an expected five-slot refusal. This is a test invocation issue, not a defect in the remediation. The exact test passed in an isolated scheduler root. No code was changed to conceal this failure.

## 4. Corrections made and independent validation

No production source, test, gate, hold, release or old report was edited by this reviewer. Created only this report and its new per-file evidence JSON. Documentation qualifications above are the review corrections.

Targeted command:

```text
bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v14-remediation-review-v2-targeted /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_postabnormal_v14.py experiments/tests/test_stage_F0_run.py
```

Actual result: **48 passed, 1 warning in 252.37s (0:04:12)**. The warning is the existing PyTorch legacy TF32 deprecation warning.

Related command, initially concurrent with the targeted command:

```text
bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v14-remediation-review-v2-related /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_worker_control.py experiments/tests/test_stage_F0_queue.py experiments/tests/test_stage_F0_launcher.py experiments/tests/test_stage_F0_migration.py
```

Actual result: **1 failed, 83 passed in 124.98s**. Sole failure: `test_python_task_wrapper_reserves_before_exec_and_refuses_sixth_process`, with `all five F0 task-process slots are occupied` during its four fixture reservations.

Exact failed-test retest:

```text
F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-remediation-review-v2-launcher-isolated bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v14-remediation-review-v2-capacity /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_launcher.py::test_python_task_wrapper_reserves_before_exec_and_refuses_sixth_process
```

Actual result: **1 passed in 0.07s**. Thus all 132 applicable unique tests were independently observed passing across these executions, with the initial invocation failure retained above.

The reviewer normalized JSON list/tuple representation before comparing exact pre/post manifest entries. This avoids treating equivalent serialized inventories as different in-memory types; the final evidence retains both full inventories and their equal digests.

## 5. Remaining risks and exact identities

| File | Actual SHA256 |
| --- | --- |
| `experiments/stage_F0_postabnormal_v14.py` | `5380ef1c3698f8b4ddfe34fce1bd8a4de15c04ad62f91a156bfcb273bb53298d` |
| `experiments/tests/test_stage_F0_postabnormal_v14.py` | `1827aab62cfe71632bcef9ad0e299ddb74482970919367d9958d2da7f8e8bfbf` |
| `experiments/tests/test_stage_F0_run.py` | `f05918f5e2ac85eb414ff5920935de5d7a619d2201397df2c79ffe9b32ff4b1d` |

| Current protected inventory | Count | SHA256 using coder's path convention |
| --- | ---: | --- |
| Queue | 2 | `1431f78ad0618f91d88e04f8a1c80b19f795d824132b5371b648131eb0bfcd03` |
| Queue statuses | 155 | `82c06fc57e0eee9740236e9c63a33d6af2ec585ee238761ee92137ba290216d1` |
| Canonical run assets | 1,193 | `a0db7e3fb969fbd517451e29374dfeca0ae99723c865a8828df5dddfe7c2042e` |
| Checkpoints | 535 | `219623122ae9c9525dd77aeac34f454006754cac8d16fd8bd2af2e4e69679d63` |

Full evidence additionally preserves uniform output-root-relative manifests and all protected individual hashes. Original broad preimplementation manifests cannot be recovered from the evidence reviewed; this review establishes current suite preservation, not complete historical broad preservation. The actual old scientific snapshot digest is not replaced by the current runtime/control digests.

Current hold remains active with ID `cd2dbf50-3a50-4b81-b976-38f1b03991a3`, exact SHA `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51`. Incident nonce remains `8fc4a33fc7574a368312b4965e24a283`, canonical claim null, receipt absent. Post-review process inspection found no live F0 supervisor/child/pytest. `fuser` found no holders on execution-state or either worker lock. No external GPU process was stopped.

Historical stale review, original one-time v1.4 release and new incident hold remain incompatible with a safe second v1.4 release. Passing this patch does not resolve that incompatibility and does not change the v1.4 two-GPU predicate.

## 6. Whether experiments are safe to run

**Production experiments: NO. RELEASE NO-GO remains.** Regression/validation work in isolated roots is safe within reviewed scope. Root may proceed with the design's administrative nonauthorizing amendment evidence/candidate procedure and separately specified successor protocol. Fresh-only production execution requires that successor's independent review, current incident/hold binding, eligible-GPU scheduling and separate authorized release transaction.
