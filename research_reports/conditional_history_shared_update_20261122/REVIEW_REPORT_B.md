# REVIEW REPORT — B_CORE and conditional B_PULSE

1. **Verdict: PASS WITH FIXES.** Both blocking deviations identified during review have been corrected by the coder and included in the final audit and engineering check. No blocking issues remain. This review covers the original 11-22 B observer, tables, pulse execution, gates and runner integration; it grants no scientific conclusion.

2. **Requirement-by-requirement verification**

| Requirement | Verification |
|---|---|
| Natural objective and baseline | The clean source forward and original scalar backward remain the live update. Six real and twelve pseudo gradients use sum/(2B), retain the original graph and gate detach, and are audited against the native gradient. Frozen v3 closure remains unchanged. |
| Coordinates | Actual-name mapping covers EMBED/S_DENSE/N. Domain embedding is excluded from shared EMBED; its three rows are recorded as D0/D1/D2. Published counts for both fixed models are enforced. |
| Native Adam interventions | Deep-copied model and native Adam load complete parameter groups and private state. Non-None zero gradients still step; native None remains None. WD is applied once by Adam. The full 52-intervention menu and 18 extra sweeps are enforced. |
| Diagnostic isolation | Model/Adam/F identities and RNG/loader identities are checked before live commit. Virtual JOINT equals actual JOINT exactly. Real first/tail engineering checks compare the observer with independent unchanged-source updates on both datasets. |
| Readouts and controls | Probability population AUC, ties, fixed-pair changes and natural LL are separate. Raw and common-radius updates preserve JOINT outside the scope; zero radius is NOT_INFORMATIVE. Online, small online, identical small displacement on F, actual F and full displacement on F are recorded. |
| True/pseudo identity control | Two independently keyed label-stratified six-bin partitions preserve per-bin positive/negative counts. Each has a separate six-way choice plus NO_ACTION. |
| Train selection and static control | Corrected selector uses P_A population Macro AUC, strict positive action and domain-major tie order. LOSO static sources use other four seeds at E5/E20; E35/E50 confirmation uses P_B/V_B, with no Validation source selection. |
| Parent windows and pulse | All E20/E35 parent windows are retained in RAM. A triggered dataset schedules all five seeds and both epochs for true/pseudo0. The first full model/Adam/F post is reused, K−1 subsequent updates recompute ordinary joint gradients, and parent row/RNG identities are verified. Terminal controls use full-model directions. NO_ACTION reuses JOINT and remains in statistics. |
| Gates and statistics | Complete 20-anchor and 20-pulse-slot dataset populations are required. Epoch effects are reduced within seed before five-seed means/SD/t4 intervals. Future JOINT/pseudo, held-out static, F residual, six-cell protection and action-value gates are implemented. |
| State, resources and boundaries | Deferred branches use guardian RAM with initial readback verification. Scientific source/check/owner identities gate dispatch. Admission uses current per-card memory/power and host RAM plus measured physical capacity. Conditional full-training methods and S5 remain outside this approval; Test is sealed. |

3. **Bugs/deviations found**

- **MAJOR, corrected:** Source selection initially used sampled fixed-pair correctness rather than the specified population probability AUC. This affected selected pulses and LOSO static sources. The observer, gate and table now share `selection_score`, which returns `P_A.macro_auc_delta`.
- **MAJOR, corrected:** The RAM anchor stored an RNG identity tuple, while JSON returned a list. The pulse dispatch equality check would reject an otherwise identical anchor. The anchor now stores a list initially.
- No remaining CRITICAL, MAJOR or MINOR implementation deviations were identified within this scope.

4. **Corrections and validation**

The coder made the two corrections; the reviewer made no implementation edits. Independently reran the updated basic check, including a case with population AUC gain +0.25 and sampled-pair change −0.5. Independently checked the C++ transition kernel against brute force on 760 randomized tie-heavy populations. A separate synthetic search-gate check confirmed five-seed reduction, rejection of an incomplete slot population, and retention of all-zero NO_ACTION effects.

The final `B_CORE_CHECK.json` passes on the real first/tail batches: ML [40960,21927], KR [40960,40052]. Both records enforce exact source/observer identities, 52 virtual steps, 18 sweeps and zero Test/scientific runs. Current owner generation was independently queried and matched. The check is bound to the final source hash below.

5. **Remaining risks**

The engineering fixture uses two native updates with synthetic anchor labels; it is not a scientific trajectory. Actual 40-anchor completeness, gradient audits, exact JOINT equality, window continuity and pulse stream checks remain enforced during production. Admission must continue to use current telemetry and the measured profiles. No sustained scientific training, full conditional B_SEARCH implementation or final Test evaluation was run or approved by this reviewer.

6. **Whether experiments are safe to run**

Yes: the ten registered B_CORE clean replays may run under the current-conversation foreground runner, with the existing resource, RAM and stop gates. B_PULSE may run only after its dataset gate triggers and must include every registered slot. A remains governed by its separate review. B_SEARCH/AB/G and S5 require their later implementation and review.

Final source SHA256: `8d13cc6cf2fd730563720087ce8beef51d2055e2e6654b2db777ec8b3d800cad`.

Design SHA256: `607fb9b7ada44aebae7cd8d80c1fd163e83206607e751de455a46efbcd7a3272`.

Check SHA256: `7350e22b0194c461f5e8e1cab811e5b218e2a855b58548b027356e36e669a963`.

Skipped scientific runs and downstream conditional-training review. No open blocker remains for this bounded B approval.
