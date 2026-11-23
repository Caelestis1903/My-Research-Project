# REVIEW REPORT — A_TARGET

## 1. Verdict: PASS

Independent inspection of the actual implementation, approved 11-22 files 03/04/06/09, DESIGN_SPECIFICATION.md, and CODER_IMPLEMENTATION_REPORT_A_TARGET.md found no blocking implementation defect. The reviewed source hash is `225beded64c94932a2d32d36184f0290a2de64bc04c368785c09f92020e89e95`.

This authorizes only the registered read-only localization of A21 (.05) and A40_HISTORY_L025 (.025) on their own retained actual-trajectory packets. It does not authorize scope training or Test evaluation. Parent completion, retained generation, packet identities, and stop/resource checks remain mandatory runtime gates.

## 2. Requirement-by-requirement verification

| Requirement | Verification |
|---|---|
| Approved specification and preserved baseline | The four original plan files match origin commit 4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983. All 52 frozen v3 source files match their manifest. No source code was edited during review. |
| Actual trajectory, both registered intensities, five seeds and four anchors | `parents`, `parent_result`, and `seed_run` require exactly ML/KR × seeds 2022–2026, completed 50-epoch parent results, E5/E20/E35/E50 anchors, parent reviewed source, current RAM generation, packet digest, data/group identities, cursor, ordered batch rows, and global step. Static PAIR is regenerated and checked. Each intensity uses its own parent packet and coefficient; there is no borrowed clean or cross-dose optimizer state. |
| Reproduce the actual optimizer/online/F update | `replay` loads a private model and complete deep-copied optimizer, applies recorded mixed/raw PAIR gradients, and demands exact model, optimizer (including shadows), and EMA hashes. Native missing gradients and per-parameter/private PAIR clocks are checked. No training graph is recomputed or altered by the probe. |
| COLL full shadow | `alternatives` uses actual post-step mR, actual v and parameter clock, and six independent M9 shadows. N_H=mR/(1-.9^step)+lambda Q/(1-.99^pair_clock) when PAIR exists; N_shadow=(mR+sum M9)/(1-.9^step). Empty PAIR omits Q injection. WD remains in mR and is applied once in the replay. This is the specified same-trajectory ordinary first-moment reference with a common actual denominator. |
| Six source replacements and group routing | Each replacement subtracts only lambda Q_c/(1-.99^pair_clock)-M9_c/(1-.9^step), preserving other sources. Q-sum and numerator closure are checked; reconstructed actual parameter updates must be bitwise equal. STAR_SKIP and EXPERT partition only the frozen S_dense set; embedding and other parameters retain the actual H update. No A22 real/WD residual is assigned to this A21 experiment. |
| Matched RMS | The reference and alternative L2 norms use the same group coordinates, so their ratio equals the specified RMS ratio. One scalar rescales the whole applied group displacement from the same prestate. Group-external parameters and buffers retain actual poststate. Both-zero, zero-reference, and zero-alternative cases follow the specification; unmatched cases stay explicit. Storage-precision norm error is checked. |
| Fixed panel and readout semantics | Frozen B panel hashes are checked. P_A/P_B training rows and V_A/V_B validation rows are disjoint by their registered partitions. Population probability AUC uses all positive-negative pairs of each fixed receiver population, ties=.5; sampled-pair transitions are separate. Six cells, Macro, natural LL, exact correction/break/tie tables, and descriptive fold/pair reliability are retained for online and fixed EMA. All branches use the same pre-step EMA and rho. No persistent-effect claim is implemented. |
| Action menu and gate inputs | Each anchor requires exactly 28 group/source/scale actions plus raw S_dense H_SHADOW. The combined artifact requires both intensities, 80 anchors and 320 whole-group gate rows. Only whole-group raw/norm-matched fixed-EMA V_B population AUC enters `gate_rows`; individual-source diagnostic maxima never select a source or enter the scope gate. `results.localization_evidence` aggregates the four anchors within seed and then the five seeds, requiring complete informative raw and matched records. |
| Isolation, gradients, RAM and Test | Probe functions use no_grad and private clones; source gradients are consumed exactly as recorded. Synthetic packet immutability passes; runtime RNG neutrality is checked. Loader/PAIR identity is checked without advancing training. All counterfactual model/optimizer tensors are transient RAM; only scalar/identity/evaluation JSON survives. This path has no Test loader or Test evaluation and adds no training replay or scientific seed. |
| Admission and foreground lifecycle | Both workspace stop controls are checked through `history_stage.stopping`; workers require their foreground supervisor PID. Admission uses each GPU's current memory <85%, current power <85%, host RAM <90%, plus usable-capacity physical fit with measured full-process residency. No predicted-power predicate, projected85% threshold, or worker-count cap is added. Sampling follows each launch. Pending first-complete-anchor protection is released after the allocation receipt; measured diagnostic residency updates the profile. Failure pauses and partial scalar results require digest-bound receipts before reuse. |

## 3. Bugs/deviations found

No CRITICAL, MAJOR, or MINOR implementation defect found in the reviewed scope. The full S_dense shadow is an explicitly labelled diagnostic reference and is excluded from gate inputs, consistent with the prescribed decomposition.

## 4. Corrections made

None. Only this report and the independent review receipt were written.

## 5. Remaining risks and evidence limits

The project-Python synthetic check passed independently: five optimizer steps, changing sources, empty PAIR, native missing ordinary gradients, independent mixed first-moment reconstruction, source-sum displacement closure, packet immutability, 29 action identities, 240 panel forwards, all three zero-norm cases, and 80 synthetic gate rows.

Read-only scalar inspection authenticated ten accepted A21 parent results and their 40 anchor receipts. At review time the ten .025 parent jobs were registered, but there were zero completed .025 parent anchor scalar receipts and no A_DOSE_RUN.json. Therefore this review does not claim any .025 packet was measured or its numerical results verified. The code correctly prevents its localization until that parent run is complete.

No actual scientific packet was scored, no GPU allocation calibration was rerun, and no scientific outcome was analyzed. Runtime exact replay hashes and initial full-anchor allocation protection remain necessary. The initial physical-fit profile uses the prior completed full-process training measurements and is refined from actual diagnostic residency; it is an engineering admission measurement, not proof of future allocation success.

## 6. Whether experiments are safe to run

Yes, for foreground A_TARGET diagnostics with the existing gates satisfied. A21 can proceed when current RAM/stop/resource checks pass. A40_HISTORY_L025 must wait for the authenticated complete A_DOSE parent run. `combine` must wait for all 80 accepted anchors. Scope training remains subject to the later scientific gate. Test remains sealed.
