# REVIEW REPORT — DFD-v4.0 finalization

## 1. Verdict

**PASS WITH FIXES. All blocking corrections are complete.** This admits the finalization sequence below for the actual completed campaign. It does not assert that the pending full-row exports or cleanup have already run.

Reviewed `DESIGN_SPECIFICATION.md`, plans04/05, the user's superseding storage instruction, `CODER_IMPLEMENTATION_REPORT_FINALIZATION.md`, actual `finalize.py`, the previously admitted pipeline inference path, and actual campaign artifacts independently of coder claims.

Locked finalizer SHA256: `3893300091f35fa12930f7b62fe49fff64c242984e8033115b54d7d2318ad50f`.

All-package source hash: `ca1ac6b7646012a342cb407e449aa5e73c355c078575b4d5f07e4476af41e231`.

The previously admitted native/pilot, P1 and P2 training files were compared individually with `REVIEW_CONDITIONAL_APPROVAL.json`; every byte hash remains unchanged. Prefix source hash remains `1273530d12170a63fdda6c68f1b037e5aa67ccf1487a58cf5934518a38b17850`; pipeline source hash remains `fc97ec1d10ba0ceccf354d53569c9d0d8009d764f250c7b0e37373fbdad5c16f`. Finalization is excluded from those fixed training scopes and included in the final evaluation source freeze.

## 2. Requirement-by-requirement verification

| Requirement | Actual implementation and verification | Status |
|---|---|---|
| Freeze the actual completed matrix before exposed scoring | Reads recorded complete decisions; checks exact required/observed completed arms. No new fit, selection or threshold calculation. Current required matrix is12 prefix variants,20 P2 selected predictors and14 genuine pass10 endpoints, totaling46 variants. P3/P4 remain NO_START. | PASS |
| Complete source/data/config/role identity | Manifest includes exact Python source hashes, raw/prepared data identities, role/row hashes, plan/config, prefix config/data manifests, current source/data dependencies, and existing decision files. Evaluation revalidates these identities. | PASS |
| Real selected and fixed-endpoint identity | Native selected and native epoch24 full states, fixed pilot12, selected native cache, P1 OOF states, exact library/readout/policy identities and genuine completed P_read pass10 states are checked. CONST_UNIFORM references the actual UNIFORM formation state. Read-only manifest inspection covered252 temporary complete-state file identities. | PASS |
| Original encoded X through the complete predictor | Prefix predictor and P2 capsule reconstruct native plus applicable downstream components from frozen state. No labels enter forward inference. Epoch24 reference uses the pinned fixed model's own probability forward. Other references use independently frozen selected-native h/z cache. | PASS |
| Numerical and probability semantics | Prefix preserves original FP32 sigmoid followed by FP64 export; P2 preserves its already used FP64 sigmoid/mixture. LL/AUC continue through existing metric implementation. No new clipping, action, routing or policy selection is introduced. | PASS |
| Full-row reconstruction and four-role exports | Evaluation requires max absolute reconstruction error≤1e-6, unchanged model state and agreement with saved development exports where available. Required roles are validation, Q_explore, test_exposed and Q_legacy_confirm_exposed:184 exports. Rows, domain, labels, probabilities, exact predictor/freeze identities and hashed metric/prediction receipts are retained. | PASS; execute all rows next |
| Interrupted export continuation | Only existing completed output receipts with matching freeze/predictor/file hashes are reused. Completion is written after the complete frozen matrix. | PASS |
| Required deployment measurement | `pipeline.benchmark_saved` uses raw encoded X through the complete capsule, batch4096,5 warmups and30 synchronized repetitions. Admit MovieLens10 arms at seed5101 and the equivalent frozen-prefix raw-X loop for KuaiRand NATIVE/LOCAL. No training required. | PASS with reporting conditions below |
| User storage revision | Cleanup restricts the exact campaign root, rejects external traversal, preserves temporary state/config identities before removal, verifies exported information and reconstruction receipts, removes temporary models/cache and checks for remaining model archives. No /home_new access is authorized. | PASS; execute only after all export and timing receipts |

## 3. Bugs/deviations found

1. **MAJOR, corrected:** the first epoch24 comparison reused the adapter path and was insufficiently independent. Its reference now invokes the actual pinned fixed model forward on epoch24 weights and transposed raw X.
2. **MAJOR, corrected:** a uniform FP64 sigmoid finalizer would change original prefix probability precision and potentially AUC ties. Prefix reconstruction/cache reference now preserve the original FP32 sigmoid; P2 keeps its original distinct precision. The manifest records the distinction.
3. **MAJOR, corrected:** final prefix identity needed to lock actual saved configuration and data manifest, beyond model state alone. The implementation now checks and freezes both and rechecks them before evaluation.
4. **MINOR, reporting constraint:** the benchmark's static `candidate_functions_per_task` is not a measured executed-function count. LOCAL computes three source pilots per task and gathers one. OFF soft policies still compute five candidates while skipping the gating forward. The final report must distinguish actual evaluated functions from contributing functions and state the actual selected policy. No training or inference algorithm change is required.

No CRITICAL issue remains. The finalization code introduces no research algorithm, gradient update, baseline change, seed change, new experiment or post-exposure selection.

## 4. Corrections and independent checks

The owner made the corrections above only in the new finalization module. The reviewer did not alter research implementation.

`REVIEWER_FINALIZATION_ACCEPTANCE.json` records the independent read-only actual-manifest and CUDA0 checks. Every one of46 actual variants was reconstructed on the first production-shaped4096-row validation batch; model states remained unchanged. Across those checks the maximum raw-X/reference error was8.359663294932318e-8 and the maximum P2 saved-validation difference was2.220446049250313e-16.

After the last prefix-precision-only revision, both data schemas and all three affected prefix variant types were rechecked at seed5101 on4096 rows. Raw-X/reference error was0; the largest difference from original saved prefix validation exports was1.1920928955078125e-7, below the unchanged1e-6 acceptance tolerance. P2 code and path were unchanged. The initial46-variant check's preview freeze hash is historical, not an actual committed freeze; the final freeze must be generated from the locked current source.

At review completion, no final freeze had been committed, no exposed output had been scored, and no production cleanup had run. These checks did not retrain models or analyze experimental effectiveness.

## 5. Remaining risks and execution conditions

Full-row184-output reconstruction is mandatory during the admitted evaluation; the sampled independent checks do not replace it. Stop on any identity, numerical or completeness failure.

Complete the12 required raw-X inference structures before cleanup. Record GPU0 load, device, FP32/TF32 settings, actual parameter counts and actual evaluated/contributing function counts. Call the existing production configuration first. Timing repetitions are measurement repetitions, not research seeds; concurrent load limits interpretation. No speed-superiority claim is warranted from the admission itself.

Cleanup code enforces export/reconstruction receipts but does not itself enforce completion of the separate benchmark. The orchestrator must schedule cleanup only after the benchmark receipts are complete and retained. Model deletion is the user's explicit superseding instruction; after deletion, predictions cannot be independently regenerated from retained model weights. Keep the source/data/config/state identities and complete prediction/metric receipts.

## 6. Whether execution is safe

**Safe to execute in this order:** freeze the current complete actual manifest; run all184 four-role exports and full-row raw-X reconstruction; finish the12 required raw-X benchmarks; then delete only this campaign's temporary models/cache and internal aliases after receipt verification. Evaluation and benchmarks may be ordered either way after the immutable freeze, but both must finish before cleanup. No new training or formal experiment is admitted by this report.

Machine admission is `REVIEW_FINALIZATION_APPROVAL.json`, copied into the exact artifact root. All blocking findings are corrected; no additional audit or experiment is required before this sequence.

## Post-cleanup specification-completeness correction

The orchestrator requested an additional read-only check of plan05§3.2 after execution and cleanup. That section explicitly requires frozen P_form OOF demand quartile cutpoints to be applied using predicted μ on Q_explore and final evaluation roles. **This mechanism-information requirement is not complete.** The earlier finalization review covered predictor/role exports and reconstruction but omitted this separate required mechanism export; its broad completeness statement must be read with this correction.

Actual `diagnostics.oof_support_tables` produces `P1_DEMAND_GROUPS.json` solely from `p1_oof.npz`, so those are P_form OOF groups. The outer-role implementation exists only within `pipeline.fixed_content_interventions`, which belongs to untriggered P3. A read-only campaign-file inspection found no Q/final demand-group or intervention exports. The final NPZ schema contains row_id/domain/y/p and predictor/freeze identities, without refitη, μ or quartile assignments. The retained `information_data` files contain only MovieLens fit_row_id/w/m/tau; these are not Q/outer η outputs. No `.pt` or `.pth` model remains. The preserved state hashes and normalizer metadata do not reconstruct deleted trained parameters or their predictions.

**Severity: MAJOR** for mechanism-information retention and specification completeness. It is not a CRITICAL numerical error in the registered P1/P2 decisions: those rules consume the separately retained P_form OOF results and complete whole-role/cell prediction metrics, not these outer μ groups. This correction does not establish or reject a research mechanism; it limits which mechanism claims the retained artifacts can substantiate.

Exact outer refitη/μ cannot be recovered from the inspected retained information. In particular, ETA_DIRECT selected candidates5/6 and fixed P_read pass10 outputs have undergone subsequent readout fitting and must not substitute for frozen P_form-refitη. Repeated OOF values, centered formation weights, hashes or aggregate scores likewise cannot supply the missing outer-row predictions. Record the item as MISSING and unavailable after the authorized model/cache cleanup; do not invent assignments, retrain, modify frozen predictors or imply full mechanism-report compliance. No additional experiment is admitted by this correction.
