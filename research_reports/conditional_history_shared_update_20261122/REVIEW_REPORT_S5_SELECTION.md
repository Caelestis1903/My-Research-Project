# REVIEW REPORT — S5 scientific selection

Reviewer: `/root/review_selection`. Date: 2026-11-23. Scope: independent review of the actual Validation-only scalar selection and proposed S5 comparison matrix. This reviewer is distinct from the selection author and the S5 code reviewer.

## 1. Verdict: PASS

The actual `S5_SELECTION.json` legitimately selects unaugmented A21, static PAIR, lambda .05 and S_DENSE, targeting MovieLens and applying the same method parameters to KuaiRand. This accepts the scientific selection handoff for the normal final-freeze process. It does not establish final Test qualification, complete FINAL_FREEZE, verify current RAM availability, or authorize additional training.

Exact selection SHA-256: `97812388eeb4f1bc50c54cb8075433045ae88466b0508e8965a117115c211460`.

Exact corrected code-review receipt SHA-256: `95a60b981fda886e824f3d958bbf55c13d19857196f54d9d1cbb88bcd3985f39`.

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
|---|---|
| Original authority and approved design | Read original 00/05/06/09, DESIGN_SPECIFICATION, corrected S5 code report/review and guardian correction. All four plan files equal their immutable `4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983` Git blobs. Design hash is `607fb9b7ada44aebae7cd8d80c1fd163e83206607e751de455a46efbcd7a3272`. All six code-receipt file hashes, coder-report hash and design binding match current files. |
| Completed core and branch evidence | Executed production `terminal_gates` read-only on the actual selection: PASS, 178 authenticated evidence files, 12 accepted local core results and 80 consistent selected A identities. A_CORE, B_CORE, A_HISTORY, A_DOSE and A_TARGET have completed required records. A_HISTORY and A_DOSE each contain all 20 prescribed jobs, accepted completion receipts and full 50-epoch earliest-max Validation curves. Their separate review receipts are PASS and bind this design. |
| A continuation and simple-history selection | Independently recomputed five-seed per-cell/Macro AUC and LogLoss differences from the selected fixed-EMA endpoint CSVs; they match all A_CORE/A_HISTORY/A_DOSE `vs_a00` values and the relevant matched-Adam contrasts. Both A20−A10 means are positive, so A21/A22 were correctly triggered. Applying the registered protection stratum, .0001 band and complexity order selects A21 on MovieLens and A20 on KuaiRand. MovieLens A21−A20 is +.0000348362, within the registered band; the simpler A21 is preferred. Ordinary Adam is outside that band, and A22 fails the clean six-cell guard on MovieLens. |
| Target and common transfer form | MovieLens's eligible preferred-form increment over A10 is +.0001959191, exceeding KuaiRand A20's +.0001717663; both preferred forms have positive clean increments. The original target rule therefore selects MovieLens. Both dataset entries declare identical A21/collapsed/static/.05/S_dense parameters. All ten actual A21 job configs agree with these declarations. Dataset-specific rho, WD and source-model settings remain the registered values. Inherited Gaussian target_domain/kappa/lambda_aug fields differ in the full source configs but are inactive in the unaugmented non-A0/A1 PAIR objective; they do not create a different transfer method or Gaussian arm. |
| Dose and scope | Both .025 arms completed five seeds on both datasets with matching controls. Independent A40−A41 means are −.0002326276 ML and −.0043656098 KR; the lower dose does not replace the target's .05 choice. MovieLens's A21 remains more than .0001 above either eligible ordinary-Adam dose. Verified all 320 unique localization rows against the exact two-dataset/two-dose/five-seed/four-anchor/two-group/two-scale menu and all 80 source hashes. MovieLens reduced-dose STAR_SKIP/EXPERT intervention Macro changes are negative at both raw and matched scales, so there is no jointly confirmed localized damage. KuaiRand lacks an eligible selected history increment after dose comparison. A_SCOPE is correctly NOT_TRIGGERED. |
| B and joint closure | Authenticated the complete 40 pulse slots and independently recomputed terminal V_B population AUC reductions, averaging anchors within each seed. ML online true−JOINT is −.0000690346; KR online true−pseudo is −.0000561967. Held-out selected−static means are −.0000362054 ML and −.0000134340 KR. Both fail the required training-information gate and correctly record MECHANISM_WITHOUT_TRAIN_SIGNAL. B_SEARCH and AB_FROZEN are correctly NOT_TRIGGERED; no search or bridge is required after this failure. |
| Gaussian technical gap | Inspected actual source audit, original trigger, independent G_CONTROL report and receipt. Audit, trigger, design and independent report hashes match their bindings; auditor and reviewer differ. The exact control inventory is ML {2022,2024}, KR {}, rather than both datasets × all five seeds. The independent review documents a concrete noise-stream mismatch and KR source D2/D0 mismatch. Original 00 explicitly pauses this supplement on control mismatch. G_FROZEN remains TECHNICAL_BLOCKED with no results, while its trigger remains TRIGGERED. This is only a source/control prerequisite failure; no Gaussian performance conclusion follows. It does not exclude the unchanged unaugmented A21 candidate. |
| Exact final comparisons and baselines | Constructed the actual production matrix read-only: nine arms and eleven directed comparisons per dataset, 90 Validation endpoints in total (58 archived, 32 accepted local). Arms are A00/A10/A20/A30, all three published baselines, A21/A22. The matrix includes all five core contrasts, candidate−strongest published, candidate−A00, candidate−A10, A20−A21, A20−A22 and A21−A22. All selected internal identities match accepted A tables; all 28 reused core endpoints match S0 identities and metrics. Published baselines retain native selected online readout; internal arms retain selected fixed EMA. No augmented control is substituted. |
| Freeze plan and Test isolation | All 32 local proposed main-table endpoints passed `accepted_result`, including their complete 50-epoch Validation curves, earliest maximum, bound same-point loss and accepted selected-model hashes. All 40 history/dose accepted jobs record zero Test evaluations. Selection explicitly names Validation and declares no Test use. There is no FINAL_FREEZE or S5 scoring directory at review time. This reviewer called neither freeze nor evaluate, read no new Test, and did not load a Test dataset. Runtime must still obtain fresh selected-RAM identities and publish the immutable freeze before any permitted one-shot evaluation. |

Independently recomputed target/transfer evidence:

| Dataset | A21−A10 Macro AUC | Positive seeds | A21−A00 Macro AUC | Worst clean cell | A21−A10 Macro LogLoss |
|---|---:|---:|---:|---:|---:|
| MovieLens | +.0001959191 | 4/5 | +.0002782863 | −.0002531733 | −.0013107354 |
| KuaiRand | −.0019885110 | 1/5 | +.0010779891 | −.0027032819 | +.0013112495 |

The MovieLens target satisfies the registered positive-mean Validation eligibility and six-cell protection. KuaiRand's negative matched increment and clean cell trade-off are preserved; this is a transfer evaluation, not a claim of universal improvement. The scalar status FINAL_CANDIDATE denotes a configuration selected for final evaluation, not a claim that the .002/.003 final Test thresholds have been met.

## 3. Bugs/deviations found

CRITICAL: none. MAJOR: none. MINOR: none in the scientific selection handoff. The Gaussian missing-control matrix is an explicitly reviewed technical limitation, not an algorithmic negative. Previously documented final-delivery schema/cost obligations remain outside this selection approval.

## 4. Corrections made

None. Only this report and `S5_SELECTION_REVIEW.json` were created. The selection, code, decisions, baseline endpoints, G audit and trigger were left unchanged. No additional experiment was created.

## 5. Remaining risks

Positive mean Validation eligibility does not establish stable generalization or final algorithm qualification; the target signal is small and the transfer result contradicts a universal-benefit claim. Historical Test exposure must remain labelled during final reporting. Current selected-RAM availability for every pending state has not been reverified by this selection-only review; production freeze/evaluate must enforce those checks. Gaussian supplementation remains technically blocked. Final reporting and delivery review remain separate obligations.

## 6. Whether experiments are safe to run

PASS for the unchanged, reviewed unaugmented selection to proceed through its normal final-freeze and runtime gates under the foreground orchestrator's applicable authority. This receipt does not authorize new training, Gaussian jobs or immediate Test access. No freeze, Test scoring, guardian mutation, code change or publication was performed here.

Not checked: fresh live readback of every selected RAM state or any new Test result. Remaining limits: the incomplete Gaussian control matrix and unproven final Test qualification.
