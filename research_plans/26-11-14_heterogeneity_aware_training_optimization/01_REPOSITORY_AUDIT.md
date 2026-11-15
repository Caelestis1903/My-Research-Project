# Main-Branch Repository Audit — Current Task Scope

**Audit date:** 2026-11-15  
**Repository:** Caelestis1903/My-Research-Project  
**Main HEAD at audit:** bcb7e2e61d1524a7efab2cb059f47fe042ec16b7  
**Scope:** main branch only, as explicitly required by the user for this task.

## 1. Scope declaration

This audit inspected the current main tree, the main-branch research context documents, the existing plan directory, and selected main-branch implementation files. **No remote branch list was queried, no remote research branch was opened, and no branch comparison was performed in this task.** Historical research not present as raw artifacts on main is used only to the extent it is summarized in main's accepted context/evidence documents.

The older 2026-11-14 audit snapshot and its branch inventory remain in 15_CRITICAL_REVIEW_AND_REFINEMENT.md as a historical planning record. It is not the audit method or scope for this current task. Codex must follow this main-only audit and must not use branch-only files as current instructions.

## 2. Current main context files

| Main file | Blob SHA | Role |
|---|---|---|
| AGENT.md | edda0f9e8466523756eb1caf5d327d50f0598e20 | Current engineering contract; includes 2026-11-15 precedence rules |
| EXP_AI_CONTEXT.md | e88b9f5d3deb4b7f01619b462f1694824289d054 | Latest AI handoff, including 11/09–11/11 EMA/PUB conclusions |
| experiments/Experimental_Global_Summary.md | 8cb128c481c5f7771bf6731ec1f0a9f230ebad8f | Consolidated historical findings |
| experiments/Experimental_Stable_Findings.md | 8c2178933dcacdc170a08153b138b42501378d8b | Stable evidence register, through current main entries |
| experiments/Experimental_Open_Questions.md | 38e3db93ef59585c45784d765edae5fe2d057575 | Current unresolved questions and boundaries |
| experiments/Experimental_Experiment_Index.md | f9435ba5c4223bda5ff81ed9c4d409a0cb2a23a8 | Experiment index and execution status |
| experiments/Experimental_Evidence_Ledger.md | 9f80542e19f4052a261b9572781ed7fdedfdb590 | Evidence-level register |
| MDMT_ALGORITHM_METRICS_LEDGER.md | 79673c4514b5aeb6d85bb730b7b82dfb0ff846b5 | Six-cell AUC/LogLoss and aggregate metric ledger |
| PROJECT_ARCHITECTURE.md | b42629830bfd55062d65b492b7ac78bc1ba1805c | Historical architecture and implementation map; newer context takes precedence |

The current main root contains AGENT.md, not AGENTS.md. Neither start.md nor AGENTS.md exists at the main root. Do not invent their contents or substitute files from another branch. The main tree already contains this plan directory; its existing files were inspected before any write.

## 3. Main implementation files directly inspected

| File | Blob SHA | Directly verified fact |
|---|---|---|
| experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py | 78a4c6006b2d4f89d1b4135ce30b9d5c4d5d9303 | Enumerates domain × target-task joint-label positive/negative boundaries; the initial same-other-label condition is included in the tested design space |
| experiments/conditional_information_momentum_20261102_v3/objectives.py | e17d722de5e9fc90d24aa10dfe16fdcba87dceaa | Applies target-task pairwise softplus auxiliary loss to sampled rows, in addition to clean supervised loss |
| experiments/conditional_information_momentum_20261102_v3/optimizers.py | c4136e403b7d70cce05afc322b5a87458f875a99 | Implements Adam and tested ECCM/collapsed/CV variants; this is not a blank optimizer-design space |
| experiments/conditional_information_momentum_20261102_v3/protocol.py | 243d979676b20130d089b8c7edab8bc154c7162d | Fixed1 model/protocol factory and source data/EMA utilities |
| experiments/conditional_information_momentum_20261102_v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/model.py | 6f80172ba8e2cc829abb1ccea77cabdfa2953246 | Fixed1 wrapper captures embedding output with a temporary forward hook; current Gaussian perturbation is inserted at this boundary |
| experiments/conditional_information_momentum_20261102_v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/reference.py | 6d1616efa1ce3e028ccd9de4a9b2daae302a184c | M3oE reference forward; embedding output, six towers and pre-sigmoid tower outputs are accessible; LayerNorm is used, with no BatchNorm or Dropout in this model |
| experiments/stage_S3_gradient_access.py | 3cd2a8789688373f9211ebe70541e0937bb99390 | Prior target-only/shared gradient access experiment |
| experiments/stage_S4_gradient_controls.py | 3f5dc0d270a8ab3d9959eb4db1efa5ddc95b9570 | Prior norm-matched gradient controls |
| experiments/tests/test_stage_S3_gradient_access.py | 7b6e3fe1a4305c6ca72133e03d5d1aac335cbb60 | Existing gradient-access test precedent |

## 4. Consequences for the research design

1. The proposed simple label condition “target labels differ, other-task labels equal” is not a new method: the main PAIR sampler enumerates all joint-label pairs with target-bit disagreement, which includes the same-other-label subset. The existing objective already applies target-task pairwise ranking.
2. The new candidate must therefore use a different information source: per-example target-task logit sensitivity and the other-task logit Jacobian, not another label-matching rule or pairwise loss.
3. The fixed1 model computes per-domain/task tower outputs from a shared embedding. The tower hook in the current pair implementation already exposes pre-sigmoid outputs. The current wrapper also demonstrates a safe embedding-hook insertion point.
4. The inspected reference model uses row-wise LayerNorm and has no BatchNorm/Dropout. This supports per-example gradient extraction from the sum of per-row logits, subject to a finite-difference and batch-independence test in the actual execution configuration.
5. The plan must still verify the active training entry point, DataParallel behavior if any, exact data/config identity, and whether a temporary embedding override is safe. Do not infer those from the vendored source alone.

## 5. Main evidence synthesis

- Gaussian augmentation is a useful matched control, not a universal winner: MovieLens D0 has positive AUC evidence; KuaiRand did not beat the strongest published baseline in the summarized coverage study.
- PAIR/conditional supervision has already tested joint-label boundaries and pairwise auxiliary ranking. Its complete-stack gains and incremental gains over fixed EMA must remain separate.
- Generic PCGrad, GATS task projection, ECCM variants and gradient magnitude balancing have been explored. Their results do not justify another relabeling of the same parameter-gradient operation.
- Fixed EMA has strong five-seed evidence, with 11/10 attribution locating most tested benefit in shared parameters, especially EMBED. It remains an internal control only.
- PUB migration, A_MATCH feedback and several adaptive EMA action rules have version-specific no-go outcomes. They are not the target of this plan.

## 6. Missing or non-current files

- start.md: absent from main root.
- AGENTS.md: absent from main root; AGENT.md is authoritative.
- No main-branch raw source report for every historical branch is claimed. The evidence register and AI handoff are the source of accepted historical summaries where raw reports are not in main.

## 7. Audit conclusion

The existing DT-PGNA design is materially different from the already implemented PAIR label condition and from parameter-gradient projection: it perturbs the input embedding using target-logit sensitivity projected into the nullspace of the other task's logit sensitivity, then trains the target task with the original BCE label. The strongest direct comparison is not another pairwise-loss variant but the three-way comparison between pair-directed TGA, projected DT-PGNA, and pairless per-example TGA-Example, with matched random and Gaussian controls.

This is sufficient to proceed to a bounded, controlled training-optimization study. It is not evidence that DT-PGNA already improves AUC or is already a publishable method.
