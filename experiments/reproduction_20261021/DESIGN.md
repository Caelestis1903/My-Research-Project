# DESIGN SPECIFICATION — M3oE supplied-method reproduction, 2026-10-21

Status: DESIGN COMPLETE. This document authorizes implementation, not experimental conclusions. No training was run by the designer.

## 1. Research question

Can the supplied data-generation and gradient-amplitude modifications improve the original M3oE backbone reliably across MovieLens-1M and KuaiRand-Pure, seeds 2022, 2023, 2024, 2025? How do they compare with (a) the actual original learned-fusion implementation and (b) the controlled fixed-fusion baseline? Is an apparent gain attributable to the gradient scaling increment, the supplied augmentation/weighting bundle, domain size, or pre-existing domain/task difficulty?

The user's four seeds override start.md's five seeds. The concluding compact instruction of start.md prioritizes Stage 0 + 1, then evidence-guided Stage 2; Stage 3 combinations and hyperparameter searches are not unconditional tasks. The user-provided AGENTS workflow takes precedence over the older start.md restriction on delegation: DESIGN → IMPLEMENT → REVIEW → RUN → ANALYZE.

## 2. Hypotheses being tested

H-D: The supplied static copy/noise + specificity-weight bundle improves test macro AUC relative to fixed M3oE without a consistent adverse LogLoss/per-cell trade-off.

H-G: Given exactly the supplied bundle, domain-task gradient amplitude balancing adds a reproducible improvement. Its causal contrast is G−D, not G−B.

H-transfer: Any improvement persists on KuaiRand with the same, explicitly defined embedding perturbation semantics.

H-size: Improvement is preferentially associated with smaller domains rather than merely baseline difficulty. This remains an exploratory association with three domains per dataset; this matrix cannot identify size causally.

All hypotheses may be rejected. No presumption is made about curriculum, momentum, M3oE failure, or equality of domain size and difficulty.

## 3. Variables changed / required matrix

| Arm | Fusion / optimizer | Weighting and augmentation | Gradient balancing |
|---|---|---|---|
| O-original-ported | Original learned sigmoid fusion, original two-Adam update path | None | Off |
| B-fixed | Fixed exp_d=exp_t=bal_d=bal_t=1; single Adam | None | Off |
| D-supplied-ported | Same B backbone | Specificity-only weights, smallest-training-domain copy once, Gaussian embedding noise std=0.1 | Off |
| G-supplied-ported | Exactly same model, initialization, weights, augmentation, batches as D | Exactly D | All-parameter norms, tau=0.5 |

Mandatory first-round matrix: 2 datasets × 4 arms × 4 seeds = **32 formal training runs**. MovieLens seed2022 is first, followed by remaining MovieLens seeds and frozen-method KuaiRand runs. Pipeline/smoke runs do not count as formal runs. Early stopping under §4 is valid completion. Previously produced results are contextual references unless source, initialization, data, loss, selection and effective configurations are demonstrated identical; existing R2/S2/S3 outputs do not automatically replace any of the 32 runs.

D is exactly the gradient-off G0 control when the supplied sources are inspected: both trainers actually use specificity-only weighting, the same copy, and the same noise interface. Prove this with one matched batch and one optimizer update, then reuse D as G0; do not duplicate eight nominal G0 runs.

O is mandatory because the user explicitly requested original M3oE training. B is still required to isolate D/G because restoring learned fusion inside D/G would change the supplied methods. O−B is a complete protocol contrast, not a single-factor attribution to fusion alone.

## 4. Variables held constant

- Backbone: the actual `m3oe/MDMTRec/models/multi_domain/model.py` STAR + shared/domain/task experts + detached-input gates + six towers; no replacement architecture or R2 adapters. FCN dims [512,256,256,64], shared expert_num=4, sparse embedding_dim=16, 3 domains, 2 tasks.
- Adam lr=0.01, coupled weight_decay=1e-5, betas=(0.9,0.999), eps=1e-8; batch_size=40960, shuffled ordinary concatenated training data, drop_last=False, num_workers=0; no AMP, gradient clipping, scheduler, hidden LR changes, or batch downsizing. O auxiliary optimizer lr=0.001, otherwise same Adam defaults.
- Epoch limit=50, validation macro AUC once after each epoch, patience=10 consecutive non-improving validations, strict greater, earliest tie. Restore the actual best validation checkpoint at **both** early stop and epoch limit for all formal arms. Save the raw last-epoch metrics too. The legacy trainer restores best only when it breaks; this unified selection change is explicit and is not called exact historical reproduction.
- Same dataset byte identity, row split, task definitions, feature mapping, validation/test evaluation, and base-backbone initialization for a paired seed. Use an independent seeded loader generator so augmentation or diagnostic RNG consumption cannot change next-epoch sample order. Auxiliary-network and noise RNG are separately reproducible. Diagnostic probes restore all RNG and optimizer/model states and perform no optimizer step.
- Initialize canonical original backbone once per dataset/seed and copy matching backbone tensors into B/D/G before training (exclude learned-fusion weights and DBO-only tensors). D and G must share the complete initial state, including auxiliary layers. Record canonical and copied state hashes. This is an explicit unified initialization protocol; do not compare accidentally different backbone RNG draws caused by inserting auxiliary modules before STAR initialization.
- Early-stopped arms may have different trained epoch counts; report selected epoch, stopped epoch, steps, wall time, peak memory, and performance at common available epochs. Do not force additional steps or reuse test results to extend runs.

## 5. Mathematical / algorithmic definition

Let ell_i,t = BCE(pred_i,t, y_i,t), B_d be real batch samples in domain d, and D=3,T=2.

Original and fixed main objective is the actual historical minibatch group-mean **sum**:

`L_B = sum_t sum_d mean_{i in B_d} ell_i,t`.

There is no implicit division by six. Empty groups contribute no gradient. This is called `legacy_group_sum`, not the later audited S2 population-Equal objective.

Original learned fusion uses the four actual `Weights` modules, effective sigmoid values (raw initialization 1 corresponds to sigmoid(1), not fixed 1). Preserve original optimization semantics: the source uses `if ~darts_para_flag(name)`, whose bitwise Boolean negation is truthy in both cases, so the main Adam includes learned weights; the auxiliary Adam also includes `_weight_*`. Each epoch its extra pooled-BCE update is performed using one batch from a persistent **training** loader iterator, despite the source comment calling it validation. Preserve and disclose these behaviors for O; do not silently fix the predicate, move its update to validation data, or describe this as an idealized DARTS optimizer. Record both optimizers and effective learned fusion curves. Isolate the auxiliary loader RNG from main training order, and record this controlled-port difference.

For the supplied D/G model, c_i is attention entropy over padded field embeddings; specificity q_i is the first softmax output of the supplied two-output MLP. Actual weights are **specificity only**: `w_i = 0.1 + 0.5*q_i`, without detach on real samples. Complexity is computed/logged but does not multiply loss. Do not restore a complexity product based on stale prose.

Choose `d_star = argmin_d N_train,d` before training (lowest domain ID breaks ties). For each real batch copy every d_star row once, retain its two labels, append it, and mark appended rows. Add independent N(0, 0.1²) noise to the entire flattened embedding of appended rows **before** STAR and auxiliary computations. This covers all fields, including the domain embedding, but does not change routing domain IDs or any categorical IDs. It is the actual supplied model semantics and requires no movie_id→video_id mapping. KuaiRand must not gain video_id or perturb a hardcoded feature column. Real-sample weights use only the real prefix; appended samples receive detached mean(real w). Let B'_d include copied rows:

`L_D,d,t = mean_{i in B'_d} w_i * ell_i,t`; `L_D = sum_d,t L_D,d,t`.

G calculates six gradients `g_d,t = grad_theta L_D,d,t`, with all trainable model parameters in the norm (unused group coordinates count as zero). `G_d,t = ||g_d,t||_2`; for G>0, `s_d,t = 1-tau + tau*max(G)/G_d,t`; for G=0, scale=1. Final optimizer gradient is `sum_d,t s_d,t*g_d,t`. tau=0.5. This is magnitude balancing, not direction projection or full GradCraft. No differentiation through scales.

Retain the supplied G behavior of replacing unavailable parameter gradients by zeros and assigning the final zero gradient even to an entirely disconnected trainable parameter. This can create Adam state/weight-decay updates where D would leave grad=None. Do not silently repair this historical mathematical behavior. Perform a matched tau=0 vs balancing-off one-step check and report parameter-level gradient/update differences; tau=0 is not assumed equivalent. The supplied specificity-only loss leaves complexity disconnected, so distinguish complexity-only bookkeeping/update differences from any backbone/specificity prediction-relevant difference. If any prediction-relevant difference exists, record it as a competing explanation for G−D and require a narrowly scoped semantics-matched control before claiming pure amplitude attribution. Sample gradients must not be detached through w_i.

## 6. Exact affected modules / source provenance

Existing supplied files to archive byte-for-byte before editing:

- `experiments/data_generate/{model.py,layers.py,ctr_trainer.py}`.
- `experiments/grad/ctr_trainer.py`.
- Original model/trainer/basic/utils/examples under `m3oe/`.

Complete self-contained executable packages **inside each requested folder**, e.g. `experiments/data_generate/MDMTRec/{basic,models/multi_domain,trainers,utils}` and matching `experiments/grad/MDMTRec/...`, with examples/entrypoints. B is the actual original MDMTRec with learned fusion replaced by fixed scalar coefficients; D/G use the supplied **MDMTRec** class, never the unrelated PLE class at the file end. There is exactly one supplied MDMTRec definition in data_generate/model.py; grad supplies no model, so its compatible companion is an explicit port. Both use original M3oE backbone. Retain root supplied files as immutable archived snapshots or archive copies with manifest and retain root files as clear shims. Never overwrite an executing source directory.

Shared protocol/data/runner/audit code may live under `experiments/reproduction_20261021/`; avoid duplicating two independent preprocessing implementations. Default new artifacts under this directory; run receipts need absolute source paths and SHA256, not only Git commit. The original working checkout is unborn master with untracked files; do not assume a HEAD commit or commit the whole workspace. Parent orchestrator handles isolated Git delivery and pushing.

The provided `data_generate/model.py` **already** has `forward(x, test_flag=False, aug_mask=None, noise_std=0.1)` and a real Gaussian embedding-noise implementation. It is API-compatible with grad trainer, unlike the older V1 model described by start.md. Training returns `(pred, complexity, specificity)`, eval returns pred. Porting/copying it into grad is permissible, but historical identity as grad's original companion is unproven and must be marked `paired-from-supplied-model / ported`.

Critical known fix: grad trainer empty-group branch appends to undefined `logloss_list` rather than `task_losses`. Correct it to a graph-safe skipped/zero cell. Keep the explicit original G unused-gradient behavior described above, and do not allow an empty cell to create a NaN norm or an undefined variable.

No full historical VUG/Phase2/Phase3/adaptive-mask source has yet been established. Inspection found none in the supplied files or project.zip; ARCHITECTURE.md mentions them but prose is not source. Inspect old local Git histories/archive locations with a bounded search before declaring unavailable. Do not invent VUG/random replacement/soft labels/masking, call the supplied noise bundle “complete historical mask”, or claim numerical reproduction of mixed/unidentified historical maxima. If a genuinely complete and matched snapshot is recovered **before implementation freezes**, return to designer/reviewer for an explicit extra arm and protocol amendment. Do not silently replace D mid-matrix.

## 7. Implementation requirements: data and validation

MovieLens raw `/home/xzf/data/ml-1m/ml-1m.csv` (known recorded SHA256 f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4); use full natural distribution, no role-rotation downsampling. Domains ages {1,18}/{25}/{35,45,50,56}; click=rating>3, like=rating>4. Feature order [user_id,movie_id,gender,age,occupation,zip,cate_id,domain_indicator], first genre as cate_id; seven sparse embeddings and one minmax age.

KuaiRand raw `/home/xzf/data/kuairand/processed/kuai_rand.csv` (recorded SHA256 a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f); domains tab0/tab1/tab4→0/1/2; tasks stored is_click and long_view; use the 33 non-tab source feature fields plus final domain_indicator, i.e. current `run_kuaishou.py`/`KUAIRAND_FEATURE_COLUMNS` schema. No video_id or user_id input addition.

For formal paired comparisons freeze the audited `stage_S1_data.py` **training-only** encoding conventions: fit each categorical vocabulary on the real global training prefix, sorted deterministic mapping with OOV=0, fixed routing IDs 0/1/2; fit MovieLens age minmax only on that training prefix. The MovieLens helper must retain the entire natural training split, bypassing role-reversal sampling and its role-count assertions. KuaiRand may reuse `load_kuairand_natural`. Freeze feature order explicitly rather than relying on DataFrame incidental order. Label this `ported-train-only` unified protocol, because original examples fit full-file encoders. Full-file-encoded archival scores are not exact controls. All four arms use identical encoded arrays. No duplicate full-file protocol matrix is required when historical source/config identity is unavailable.

Split after domain/task construction by frozen original CSV row order at floor(0.8N), floor(0.9N), without random split. The plan describes these files as time-ordered, but current KuaiRand processed CSV has no interaction timestamp and current audited loader does not sort it; report `preserved source row order, upstream chronological provenance not independently reverified`, not a false verified temporal sort. MovieLens timestamp monotonicity can be checked cheaply without resorting. Do not sort by upload_dt: it is video metadata, not interaction time. Existing KuaiRand train counts expected {0:172782,1:1743898,2:89452}, val {0:2692,1:246232,2:1843}, test {0:2613,1:246284,2:1870}; verify current file identity and actual counts before training. Compute natural MovieLens counts from file; do not reuse R2's 150000/34500/3000 role-subsampled counts.

Before formal training: verify module paths, all four fusion/optimizer modes, shapes/device, training/eval output contract, one real batch forward/backward/step, one empty-domain batch for grad fix, tau0 matched update with explicit unused-parameter difference audit, D/G0 matched update, and one short actual pipeline run. A full one-epoch pass suffices as short run; no generic robustness suite or unrelated refactor is required. Short runs are labeled and cannot substitute for formal training.

Preserve mathematical operations; device-local result allocation is allowed as a minimal Linux/CUDA port, with patch receipt. Do not enable TF32/AMP or new optimizer options without disclosure. If original batch cannot fit, reduce concurrency first; any required batch change applies to all paired arms under a reviewed separate configuration.

## 8. Ablations and adaptive continuation rules

First complete and analyze the 32 formal runs. At that point the orchestrator records Stage 2 decision **before** any ablation runs and returns a design amendment for implementation/review. Mandatory gradient-off contrast is already G−D, requiring no extra run.

Only supplied mechanisms are eligible: remove specificity weighting (w=1); copy with noise_std=0; remove copy/noise while retaining specificity; all- vs shared-norm if mechanism observations implicate private norms. Never run “remove VUG” when no VUG exists. If a bundle has a same-direction practical macro AUC change in at least 3/4 seeds on a dataset (positive or negative), or pooled/macro/per-cell conclusions conflict, select at most two removal controls that distinguish the observed explanations. Practical screening threshold = 0.001 absolute macro AUC, recorded as a screening rule, not significance. A null/unstable bundle without a mechanism-specific question warrants no factorial expansion. Negative repeatable effects are equally eligible for diagnosis. Every selected dataset/control uses all four seeds, at most 16 extra runs for two controls × two datasets.

Stage 3 Equal/Sample/combination is conditional, not part of the first-round 32-run obligation. Consider it only if Stage 2 identifies a repeatable component or objective-dependent interpretation remains decisive. Reuse audited `experiments/stage_S2_objectives.py` population Equal `mean_i[(1/(3p_d))*mean_t BCE]` and Sample `mean_i,t BCE`; these differ from legacy_group_sum, so both require fresh matched B controls under the new scale and cannot reuse Stage1 B. p_d is frozen from **real** training rows, never augmented shares. Specify generated-loss coefficient separately and keep it identical across objectives. Any D+G interaction uses common B and four paired seeds: I=score(D+G)−score(D)−score(G)+score(B). Under current supplied definitions G already equals D+balancing, so “D+G” would double-count the same augmentation and is not a new valid arm. A genuine independent-component combination requires an explicit new specification. No tau sweep, optimizer/momentum reset study, curriculum or unrelated domains are required by the present task.

## 9. Expected logging requirements

Per formal run store `config.json`, `data_manifest.json`, `epoch_metrics.csv`, `test_metrics.csv`, `mechanism_metrics.csv`, `stdout.log`, `completion.json`, best checkpoint and matching optimizer state. Runtime/checkpoint files may be placed at `/home_new/xzf_m3oe_reproduction_20261021` with repository receipts pointing there. Config includes effective loss formula/scale, feature encoder/scaler scope, all seeds, package paths/hashes, source differences, versions/CUDA/GPU, original/fixed fusion flags, optimizer parameter identities, checkpoint rule, noise/std/target domain and tau. Manifest includes raw hash, all ordered split row IDs or compressed indices and hashes, encoded split/label/feature hashes, counts and positive rates for every cell. Do not log fabricated Git HEAD.

Train curves include optimized weighted loss **and unweighted real-example per-cell BCE**, validation six-cell AUC/LogLoss and macro, explicit pooled/task-averaged and sample-weighted aggregates, fusion trajectories, best epoch and status. Do not compare raw weighted objective magnitudes as if loss scales were identical. Test once on frozen best validation checkpoint after configuration is fixed; no test-driven selection.

Report AUC and LogLoss per six cells, macro over six, task macro over domains, domain macro over tasks, pooled per-task AUC averaged over tasks, pooled mean BCE, sample-weighted mean of per-cell AUC (a distinct metric), worst-cell AUC and its paired change. Multi-output sklearn log_loss can normalize rows as multiclass probabilities: avoid calling it on [B,2] independent Bernoulli outputs; calculate each binary task/cell loss or direct independent BCE. Record undefined AUC as undefined with class counts, not invented 0.5.

Mechanism observation at a fixed first ordinary training batch per epoch (or predeclared smaller fixed cadence) is sufficient: real/copy counts, specificity and complexity mean/std/min/max, w statistics, weight-vs-BCE correlation, auxiliary gradient norm; per-cell all/shared gradient norms, scales, cosine matrices, conflict rate, real-vs-generated gradient cosine; actual Adam delta norm and cosine with final raw gradient, m/v norms. Shared subset explicitly includes embeddings, shared STAR weights/bias/star_mlp/skip, shared experts; excludes private slots/domain/task experts/towers/gates/auxiliary networks. Treat this as a defined subset, not an assertion about mathematical exclusivity. For B/O collect the same representative unweighted group norm probes so a G mechanism can be interpreted against baseline. Diagnostic separate autograd passes do not consume noise RNG or mutate state. Complexity may be unused in loss; its absent gradient is an observation, not a defect.

Save model, optimizer(s), RNG, epoch/step and any scheduler at best only; an optional final resumable state is sufficient. Avoid every-epoch full optimizer copies. Source snapshots/manifests/diffs and compact CSV/JSON reports are repository deliverables; bulky datasets/checkpoints stay external with checksums and absolute/artifact locations. Never upload raw large datasets or secrets.

## 10. Acceptance criteria for reviewer / runner

Reviewer must independently inspect this specification, actual complete packages and coder report; report PASS / PASS WITH FIXES / FAIL. Blocking issues are wrong model, unidentified package imports, mismatched dataset/split/features, incorrect loss/fusion/optimizer/augmentation/gradient formula, absent O/B controls, wrong checkpoint selection, method/source mislabeling, test leakage, metric miscalculation, or failed critical smoke checks. Style, unused unrelated modules, and hypothetical untriggered edge cases do not block.

Coder report lists exact files, source snapshots and hashes, copied/model matching evidence, formulas, all compatibility changes, smoke results, deviations and effective commands. No research-result analysis by coder.

Runner is luna-runner per current AGENTS; it records raw logs/results only. Use the actual installed torch environment and reviewed commands. At most five processes per GPU and below 90% GPU memory; benchmark one run, then raise concurrency safely without altering scientific settings or interrupting unrelated users' processes. Max concurrency is conditional on memory, not a mandate to OOM; grad's retained graphs can cost more. Preserve every failure attempt with reason, use new attempt folder, mark exactly one valid completed attempt per dataset/arm/seed. Completion means earlystop or epoch50, validated metric files and matching best checkpoint; a job merely launched is not completion.

## 11. Conditions that falsify hypotheses / analysis requirements

H-D is weakened/rejected if paired gains vanish or reverse across seeds/dataset, LogLoss consistently worsens, or benefit comes only from a narrow cell with offsetting damage. H-G is rejected as an incremental benefit if G−D is null/negative even when G−B is positive; a package effect must not be renamed a scaling effect. H-transfer is rejected if the fixed semantic port has reproducible opposite/null effect on KuaiRand. Magnitude balancing cannot be said to resolve conflict when only scale changes and cosine signs persist. A shift in raw gradient norm is not evidence of balanced Adam updates.

Analysis unit is seed (n=4), not epochs/tasks/cells. Report all raw per-seed numbers, paired deltas, mean, sample SD, positive-seed count and descriptive paired t 95% intervals (df=3), with no strong generalization from four seeds or many post-hoc cell contrasts. Use validation-only continuation decisions; all eventual final test comparisons remain transparent and exploratory when multiple choices were considered.

Final orchestrator analysis must explicitly cover AUC, loss, each domain/task, seed consistency, size, difficulty measured from original/fixed baseline performance and label rates, model/protocol, and training stage. Identify (1) repeated stable phenomena; (2) contradictory experiments including archived evidence with incompatible protocol clearly distinguished; (3) evidence STRONG / MEDIUM / WEAK; (4) competing explanations; (5) whether size is plausibly the main explanation, without equating small and difficult; (6) whether small-domain improvement/large-domain decline trade-offs exist; (7) the most discriminating next falsification experiment. Paired signs in four seeds and another dataset can support repeatability, but current natural-domain design alone remains WEAK for causal domain-size claims. Report missing complete historical mask source, unavailable historical launch settings and any failed/unfinished run as limits; do not fill them with made-up outcomes.
