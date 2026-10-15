# REVIEW REPORT: F0 worker bootstrap amendment remediation v1

## 1. Verdict

**FAIL. RELEASE NO-GO.** The production amendment predicate is correct, but required integration, release-replay and protected-inventory evidence is incomplete. Return to coder; this review must not be represented as approval for an amendment record, release or launch.

Reviewer: independent reviewer `/root/f0_v14_fix_review`.

Reviewed design: `experiments/reports/stage_F0_worker_bootstrap_hash_remediation_design_v1.md`, SHA256 `252046a68049d108a684f20aaf3d71be9dccb48677495ccda19a509b78daff95`.

Reviewed implementation report: `experiments/reports/stage_F0_worker_bootstrap_hash_remediation_implementation_v1.md`, SHA256 `0dd9e8ac65eb86ab11a4a310a6e82a507e1076f87a414bf4a609a6280317fada`.

## 2. Requirement-by-requirement verification

| Requirement | Finding |
| --- | --- |
| Mapping, status APPROVED, verdict PASS/PASS WITH FIXES, frozen and complete runtime digests | PASS: actual `validate_current_gate` at lines 645–655 enforces all conjuncts and rejects missing/invalid approval metadata with a specific PermissionError. |
| Runtime digest and control inventory remain separate | PASS: actual runtime comparison calls `stage_F0_queue.code_hash(require_tests=True)`; control inventory is checked separately at lines 633–637. Recomputed runtime digest is `7d6bbd9174d3d26c4dce958173444e9ba9778313166aefbc3a32c30f6e9aceb3`; recomputed control digest is `8eb97d242a6686ec084c58fdee32cc4cc56cc06ecfa7dfce3d65d5974f79a511`. |
| Runner strict check and worker-control dispatch preserved | PASS within verified frozen control inventory: their actual file hashes still match the canonical gate's old inventory. No runner import or dependency cycle was introduced in the v1.4 predicate. |
| Only three permitted source/test files differ | PASS within the complete 21-file frozen control inventory: only `stage_F0_postabnormal_v14.py`, `test_stage_F0_postabnormal_v14.py`, `test_stage_F0_run.py` differ. All 52 existing file entries in the coder report's tables matched actual bytes. No Git history exists in this workspace, so the exact old source/test textual diff is unavailable from Git. |
| Metadata and digest rejection cases | PASS: actual tests cover missing status/verdict, PENDING/REJECTED status, PENDING/FAIL verdict, observed two-field amendment, wrong scientific/runtime digests, and control-digest substitution. |
| Both accepted verdicts on a valid isolated real v1.4 graph and real runner helper | FAIL: the required real-graph acceptance path resides in a test skipped under the current hold. The added current-production read-only probe passes metadata checks but always expects full validation failure; it is not a valid isolated acceptance graph. |
| Reject current production graph read-only | PASS: the current-graph regression rejects full validation, preserving historical review/provenance checks. Its generic error assertion could be strengthened to bind the intended stale historical review/rehold rejection. |
| Original release identity replay and different nonce rejected on isolated current rehold, without mutation or added release stages | FAIL: no such regression exists. Existing release/claim scenarios are in the skipped legacy graph test and do not exercise both required release calls on the current reactivated hold. |
| Applicable control regressions | PASS for executed cases: reviewer independently ran all six required groups; 130 passed, 1 skipped, 1 warning. The skipped required integration remains a blocking gap. |
| Protected production evidence unchanged | PASS for the seven specified incident/gate/hold/release identities, independently rehashed; hold remains active and receipt absent. Full exact pre/post queue/status/run/checkpoint inventories required by design §§7.1 and 9 were not retained in the implementation report or a referenced evidence artifact, so broad protected-byte preservation is not independently established. |
| New approval record and nonauthorizing diagnostic candidate | Correctly deferred until a successful independent review. Their absence at this stage is not a coder deviation. This FAIL must not be converted into an APPROVED amendment. |
| Formulas, tensors, routing, weighting, gradients and baselines | No mathematical or tensor change is part of this control-only patch. Verified runner, queue, migration and other frozen control source hashes remain unchanged; full scientific/run asset preservation still requires the inventory evidence described below. |

## 3. Bugs/deviations found

1. **MAJOR — required integration test disabled.** `experiments/tests/test_stage_F0_postabnormal_v14.py:295–301` newly skips the real migration/gate/release graph when the live hold is not the original pre-migration hold, which is the current expected state. The approval agreement assertions at lines 441–510 therefore did not run. The test beginning at line 669 modifies an in-memory production gate and only expects PermissionError from the full v1.4 validator. It demonstrates rejection and metadata enforcement, but cannot replace the specification's acceptance test on an otherwise valid isolated graph. Add an executable isolated fixture using preserved frozen evidence, with actual validator/helper acceptance for both verdicts and the required negative cases. Do not repair production history or mock the validator/helper under test.

2. **MAJOR — current-rehold release replay tests missing.** Design §8 explicitly requires invoking the write-capable v1.4 release API only in isolated clones, with the original committed identity and a different nonce, and proving both reject the new failure hold without changing hold bytes or adding ledger stages. Actual test search and inspection show no implementation of these scenarios. Add these tests against cloned current incident/release evidence and assert protected clone bytes and ledger stages remain unchanged. Never invoke this negative test API against production.

3. **MAJOR — required protected inventory evidence incomplete.** The implementation report contains seven protected identity hashes and complete code inventories, but not the exact pre/post protected queue/status/run-attempt/data/checkpoint inventory requested by design §§7.1 and 9. A statement that no other files changed does not establish that requirement. Supply retained actual before/after evidence if it exists; otherwise report its absence honestly and obtain the required isolated verification evidence without fabricating a preimplementation snapshot. Preserve the historical migration/source inventories and production artifacts.

No CRITICAL defect was observed in the new production predicate. No MINOR implementation defect is recorded separately; PyTorch's existing TF32 deprecation warning is unrelated to this patch.

## 4. Corrections made

None. Per review task instructions, implementation and production artifacts were not edited. Only this independent review report was created. The narrow source correction is not redesigned here.

Independent regression command:

```text
bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh f0-v14-hash-remediation-review /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/tests/test_stage_F0_postabnormal_v14.py experiments/tests/test_stage_F0_run.py experiments/tests/test_stage_F0_worker_control.py experiments/tests/test_stage_F0_queue.py experiments/tests/test_stage_F0_launcher.py experiments/tests/test_stage_F0_migration.py
```

Actual result: `130 passed, 1 skipped, 1 warning in 122.54s (0:02:02)`. The skipped test is the required real graph fixture above. All execution was regression/read-only control validation; no GPU experiment or production release transaction was performed by this reviewer.

## 5. Remaining risks and independently verified identities

| Artifact | Actual SHA256 |
| --- | --- |
| Canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| Active global hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Worker1 failure record | `1ee4f54f83dd4fa6f9b6615e1e7f074be2178ea5e160263aa2aeae14296ed1f4` |
| Worker1 stderr | `2798981cb0b8c4c650c3cb2aa427db201aeb6038ffa2000e8ad75759056dab9a` |
| Frozen search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| Old v1.4 release ledger | `9bdbb2bc801c33a36a6f950e66f6a4a13ba11faa21581a683545e4c9103dc71e` |
| Old immutable v1.4 release | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |

Hold ID remains `cd2dbf50-3a50-4b81-b976-38f1b03991a3`, active true. Failure nonce remains `8fc4a33fc7574a368312b4965e24a283`, canonical claim null; its receipt path does not exist. Process inspection found no matching F0 supervisor/child/runner, only the independent pytest process. Current production canonical gate still carries stale code inventory; preserving its fail-closed rejection is intentional under this scope.

The passing predicate and regressions do not establish safe successor release. Historical v1.4 release evidence must remain immutable; the reactivated hold cannot be bypassed by replay, pointer rewriting or restoring old hold bytes.

## 6. Whether experiments are safe to run

**NO.** Return to coder for the three blocking verification/evidence gaps and then repeat independent review. The active hold remains in force. Even a subsequent PASS for this narrow correction requires the separately designed, implemented, reviewed successor protocol and a distinct principal release transaction before fresh-only execution.
