# Active task — do not mark complete before all required work

User explicitly requires executing start.md, decoded GB18030. Entry point mandates full research_plans/2026-10-19 plan, no direction confirmation, no early stop. Latest local start §11 overrides older user AGENTS/agent_promt delegation: primary does design/implementation/self-review/analysis, ONLY luna_worker mechanically executes experiments in training stage. Do not spawn designer/reviewer; never misrepresent self-review as independent.

Workspace /home/xzf/Code/phase0-3 is an uncommitted snapshot with no HEAD/remote. Dedicated branch/worktree was created safely from origin/main55afd053d7a65ea782487d703f189615e296fd3e:

- directory /home/xzf/Code/phase0-3-20261019-git
- branch codex/conditional-paths-2026-10-19
- reviewed training commit aa8df217 (full SHA via git log)
- minimal pinned legacy N19 data/model imports copied, not a merge of historical branches
- /home/xzf/Code/phase0-3/research_plans/2026-10-19/execution symlinks to the actual execution folder
- old N19 /home/xzf/Code/phase0-3-n19-git at c440350681a7a3e41ab3f05eb1c4d31f419b989f is READ ONLY historical source
- Python /home/xzf/miniconda3/envs/phase0-3-env/bin/python, PyTorch2.9cu126, 2×RTX3090
- local states/probes/predictions /home_new/xzf_cm19_artifacts

All five parent plans fetched/read. DESIGN_SPECIFICATION → IMPLEMENTATION_REPORT → REVIEW.md done. Six full physical batch40960 engineering checks passed, finite, peak1250–2916MiB. Basic twelve model checks passed, matched N19 baseline forward to5.96e-8, author HAMUR associative forward difference9.24e-6, gradient max5.78e-5; WD0 Adam/AdamW identical ten steps.

Frozen manifest856 independent runs +72 mathematical aliases=928 nominal configurations. G0=160, G0M=32, G0MC=32 (required normalized private-task sharing controls), G1=120, G2=192 (+48 additive capacity aliases), G3=208, G4=96, G4LONG=16;24 WD0 optimizer aliases. Four seeds2022–2025. KR50epochs/ML100epochs, batch40960, LR.003/.001, no early stop, long ML2350updates. G4 matched exposures preserve original per-domain counts and original total updates with target-domain repeat sampling.

Formal RUN is ACTIVE. Worker /root/cm19_execution, role luna_worker, monitors two persistent sessions:

1. GPU training session21697, command python -B experiments/cm19_queue.py --gpus 0,1 --limit 5. Per-GPU independent90% admission and max5 processes. Do not start duplicate queue. Queue status research_plans/2026-10-19/execution/queue_formal.status.json, raw events queue_formal.events.jsonl; stdout logs/formal. At last checkpoint about18 of856 completed, ten running, no failures. Worker sends raw count updates every10min.
2. CPU offline raw diagnostics session38561, coordinator PID1325353, command python -B experiments/cm19_posttrain_queue.py.48 G0 fixed/STAR local-moment counterfactuals+78 representative model/variant/dataset cost profiles; at most2 CPU processes, CUDA hidden. Status offline_queue.status.json. May run concurrently; does not change checkpoints or read test. Worker must report failures, no autonomous code changes/retries.

No formal scientific results analyzed, no test metrics opened. Current stage cannot be marked COMPLETE. All856 runs must complete, including full negative groups. Any crash: primary categorizes relevance, repairs only affected science code, explicitly reviews, then restarts affected tasks without overwrite (archive incomplete outputs). Core cm19_train.py/cm19_models.py/legacy/ref files must remain unchanged unless critical fix; freeze checks every run's source hashes, so logging edits midtraining are also problematic. New analysis scripts are safe and separate.

Prepared AFTER formal training:

- cm19_collect.py --split development: requires all856 receipts, emits development_readout/{seed_cell_metrics,seed_macro_metrics,condition_summary,paired_comparisons,paired_seed_deltas,paired_cell_deltas,trajectories,costs,...}. Descriptive95% t intervals, df3, four paired seeds; weighted-cell metrics use original population weights; selected/final separate. It includes global WD0/AdamW strong references and private-task control comparisons. Large path aggregate is .csv.gz.
- Primary then scientifically analyze development (all stage results complete), decide conditional external confirmation only if useful/novel, write DEVELOPMENT_DISPOSITION.md. Do not tune on old test or arbitrarily extend grids.
- cm19_freeze.py requires all full budgets/configs/source hashes, development disposition, binds selected/final SHA256 in TEST_FREEZE.json.
- cm19_evaluate.py / cm19_test_queue.py: zero-update selected/final natural test predictions and model-specific neutral interventions; ONLY after freeze and explicit primary release. Worker can run `python -B experiments/cm19_test_queue.py --gpus 0,1 --limit 5` when released. No optimizer updates. Raw test predictions ART/.../test_selected.npz and test_final.npz, output metrics_test.csv/test_receipt.json.
- `cm19_collect.py --split test` then final scientific analysis.
- Need create figures (matplotlib PDF/PNG) and required final documents OPTIMIZER_MECHANISM.md, CROSS_MODEL_REPORT.md, PERFORMANCE_AND_NOVELTY.md, FINAL_REPORT.md, full OPEN_PROBLEMS, matrix/alias completion ledger and checkpoint/prediction hashes/local storage index. Existing MODEL_FIDELITY and LITERATURE_LEDGER / ANALYSIS_RULES guide interpretation.
- Update EXECUTION_STATUS.json with actual counts regularly; final all_directions_complete only after required training/test/analysis/delivery, not just queue completion.
- Commit new code/raw tabular reports/docs explicitly (avoid tracked old .pyc, huge states not in repo); push execution branch and create PR targeting main, as explicitly authorized in START_EXPERIMENT_PROMPT. Do not merge older N19 or main PRs. User wants full work, not a plan/final claim while jobs still active.

Interpretation boundaries already documented: new original cycles auxiliary training iterator like public author source (N19 first shuffled batch differs), fixed fusion intentional. Independent STAR implements paper-style PN+multi-layer factors+aux, differs from third-party TF choices. PEPNet uses available priors, KR lacks IDs. Author HAMUR has shared/all-ones factors, mixed-batch/statistic DN even at eval; this is public-code-style, not strict paper DN, associative contractions numerical tolerance explicit. Expert private replacements approximate matched capacity, frozen serialized inactive parameters distinguished from trainable count. Do not claim all STAR/main/shared experts dead because zero modulation sensitivity. Gate1/HAMUR residual0/expert0 are distinct estimands.

Analysis must independently cover stable phenomena, contradictions, STRONG/MEDIUM/WEAK scope, competing explanations, domain size vs difficulty, small/large trade-off, best next falsification experiment. Restoring functional contribution is not AUC improvement; final-only benefit is not selected-model benefit. Historical test population is exploratory, not new confirmation.

Literature primary findings to account for: AdamW1711.05101; Hoff2017 Hadamard L2↔sparsity;2026 Smoothing the Edges10994-026-06997-0; Gradient Starvation2011.09468; Rotational Equilibrium2305.17212; Star+2406.16568 (add/adaptive/concat/gating prior art). Shared scalar s and domain factors v_d with quadratic WD induce, under scale minimization, lambda*||effective w across domains||2 per coordinate; this algebra is known structured sparsity, not a novel theorem, and not a proof of finite-time Adam collapse. Existing additive G2 is a control, not an original algorithm.
