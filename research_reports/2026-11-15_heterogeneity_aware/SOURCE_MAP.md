# Read-only source map — 2026-11-15

Audit base: `79fbd441c6c87cff30838c65bf97ad6ec7f45abb`, local branch `26-11-15`.
This map precedes implementation. Existing historical packages/reports stay unchanged.

## Governing documents

The current plan is `research_plans/26-11-14_heterogeneity_aware_training_optimization/`, a directory on **main**, not a research branch. Its governing files are `16_CURRENT_MAIN_AUDIT_AND_EXECUTION_REFINEMENT.md`, `17_FINAL_CONSISTENCY_REVIEW.md`, and `experiment_manifest_v3.json`. All plan files were read. `AGENT.md`, the latest handoff in `EXP_AI_CONTEXT.md`, Global Summary, Stable Findings, Open Questions, experiment/evidence indices and metrics ledger supply accepted history. Main contains neither `AGENTS.md` nor `start.md`; the original workspace/user instructions remain applicable. The user's current instruction overrides the plan's development seed pool: use **2022–2026 only**.

## Verified implementation

Let `P=experiments/conditional_information_momentum_20261102_v3` and `V=P/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31`.

| Responsibility | Current main source | Verified behavior / reuse |
|---|---|---|
| Fixed1 model factory | `V/model.py::make_model` | 3 domains, 2 tasks, sparse dim16, `[512,256,256,64]`, four experts, all fixed fusion coefficients 1; no learned fusion/DARTS/BatchNorm/Dropout |
| Model input/output | `V/reference.py::MDMTRec.forward(x,test_flag=False)` | Input `[features,batch]`; embedding `[batch,input_dim]`; output `[batch,2]` sigmoid probabilities |
| Embedding insertion | `V/model.py::MDMTRec.forward` | Existing temporary embedding forward hook, removed in `finally`; new adapter must use this boundary without modifying history |
| Raw logits | `P/boundary_sampler.py::forward_logits` | Hook six towers, select tower `task*3+domain`, remove hooks in `finally`; reuse readout pattern |
| Sensitivity issue | `V/reference.py`, `gate_value=gate(emb.detach())` | Ordinary autograd w.r.t. embedding omits the gate-input derivative. Full numerical forward Jacobian requires a sensitivity-only reconnection. Training must keep source detach |
| Clean loss | `P/vendor/ema02/objectives.py`, `V/resolved_protocol.json` | **Sample-mean probability BCE across all rows and both tasks**; older root/plan Equal-style formulas do not describe this accepted natural protocol |
| Optimizer | `P/vendor/ema02/protocol.py::make_optimizer` | Single ordinary Adam, lr .01, betas(.9,.999), eps1e-8, coupled WD; foreach/fused/amsgrad off |
| EMA | Same file `update_ema` | After Adam once/batch; ML rho .9828205985452511, KR rho .995; initial EMA is initial model copy |
| Dataset identity | `P/protocol.py::load_protocol`, `P/vendor/ema02/data.py::load_dataset` | Current relocated CSV paths, natural split seed2022, train-only preprocessing, hash assertions; no prepared mdmt_next artifact needed |
| Data implementation | `V/audited_data.py` | Train/validation/test natural80/10/10; ML nested rating labels; KR tab0/1/4→domain0/1/2 |
| Training order | `P/run.py::worker` | Setup(seed), CPU generator(seed+11000), randperm/epoch; batch40960, drop_last false; 50epochs |
| Selection | Same worker | Earliest strict `validation.macro_auc > best`, online/EMA selected independently; state snapshots RAM only |
| Evaluation | `P/vendor/ema02/evaluate.py::{evaluate,metrics}` | Unperturbed inference, six cells, Macro/Traffic/Pooled AUC/LL, no conflation of pooled and macro |
| Memory state utilities | `V/protocol.py` | cpu_copy/rng_state/restore_rng/state hashing; `BytesIO` IPC available, no disk checkpoint needed |

## Data contract

| Dataset | Raw SHA256 | Train / validation / test rows | Train domain counts |
|---|---|---|---|
| MovieLens | `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` | 800167 / 100021 / 100021 | 155949,318599,325619 |
| KuaiRand | `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f` | 2006132 / 250767 / 250767 | 172782,1743898,89452 |

The primary audit verified both raw hashes. Derived identities must be checked against `V/resolved_protocol.json` before training. Loading/hash-checking the frozen test arrays for data identity is permitted; test predictions/metrics are sealed until formal freeze.

## Scope reaching implementation

Create only `experiments/heterogeneity_aware_20261115/` (adapter, perturbation/objective, runner, compact analysis and tests), this report directory, experiment outputs under `artifacts/heterogeneity_aware_20261115/`, and the required new metric-ledger entry after actual execution. Import current-main source utilities rather than copy/refactor historical models. Historical PAIR losses, ECCM and dynamic EMA are not active in this study.

No missing source/data blocker is established for the pilot. Seed-pool separation is overridden by the user, so later confirmation shares the five training seeds and must be described accordingly. Main has strongest-published-model aggregate endpoints in the metrics ledger; raw HyperGate seed endpoints have not been located on main. If formal confirmation is reached, report available aggregate comparison and the exact missing paired statistics, or run a matched published baseline if its current-main implementation/config is available. Do not invent paired HyperGate intervals or fetch branch-only files under this plan.
