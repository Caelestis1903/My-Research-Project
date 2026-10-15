# Stage S5 DESIGN SPECIFICATION — Persistent Adam History × Current Update Source Factorization

**Author: Designer agent (K3 1M max). Transcribed verbatim by main agent (designer ran read-only).**

## 0. Scope and design decision

S5 only. Mechanism diagnosis, not method innovation. One setting: KuaiRand-Pure / M3OE / natural / Equal / target d1(tab1), non-target d0(tab0), d2(tab4); tasks t0=click, t1=long_view; seeds 2022–2025; 50 epochs; 16 fresh formal runs (4 arms × 4 seeds). S3/S4 conclusions are inherited and not re-litigated. S4's N/U arms are not repeated.

**Design decision: GO.** S4's validated read-only `AdamDelta` plus its U-arm "commit-then-overwrite" pattern suffice to manipulate persistent-history source and current-update source independently, with F/T reducing bit-exactly to the S3/S4 definitions. No new optimizer, no shadow optimizer object, no extra forward, no extra DataLoader is required. If the Coder cannot reuse `stage_S4_gradient_controls.py` **unmodified**, this design is NO-GO pending amendment.

Key code facts the design relies on:
- `experiments/stage_S4_gradient_controls.py` `adam_delta()` is a read-only, side-effect-free same-state counterfactual Adam delta calculator, validated against the installed CUDA Adam in `validate_adam_delta_against_clone()`; S4's U arm already used the "real `step()` + post-hoc overwrite with recomputed delta" pattern (`experiments/stage_S4_run.py:129-137`).
- M3OE forward (`m3oe/MDMTRec/models/multi_domain/model.py:160-193`) applies all `expert/domain_expert/task_expert` to the full batch, so the target loss graph reaches every shared parameter and `g_T` is never `None` (S4's T arm had no None-related failures); design fails closed on None anyway.
- One stochastic forward per batch; `g_F/g_T` both come from `autograd.grad` on the same graph, consuming no RNG and creating no DataLoader iterator.

## 1. Experiment objective

Single research question: when the source of the persistent Adam optimizer history (first moment `m`, second moment `v`, step counter `τ`) and the source of the current actually-applied shared-parameter update are manipulated independently, which factor does the d1 rescue (S3/S4: T−F ≈ +0.039 AUC on both tasks) follow? Secondary first-class question: which factor does the d0 damage (≈ −0.031/−0.032 selected AUC) and the d2 late-stage sign reversal follow?

Arm matrix (factor H = persistent history source; factor D = current applied update source):

| Arm | H (updates `m,v,τ`) | D (computes applied Δ) | Identity |
|---|---|---|---|
| F | FULL | FULL | = S3/S4 FULL baseline |
| T | TARGET | TARGET | = S3/S4 target-only arm (rescue positive control) |
| HT | TARGET | FULL | new |
| DH | FULL | TARGET | new |

## 2. Mathematical definition

Per train batch (main step, group `C`) and per epoch-end architecture batch (group `A_arch`), from **one** stochastic forward:

- `L` = full Equal objective, `L*` = target-only component (S3 `loss_components`, unchanged).
- `g_F = ∇_G L`, `g_T = ∇_G L*`, both via `torch.autograd.grad` on the same graph (`gradients_one_forward`), `g_P = ∇_P L` for private params (always FULL).
- Persistent state per parameter: `s = (m, v, τ)`. Pre-step values denoted `m, v, τ`; pre-step parameters `θ`.
- Coupled weight decay `λ=1e-5` enters both paths exactly as installed `torch.optim.Adam` does: `ĝ = g + λθ`.

Define the two per-arm source gradients `g_H` (history) and `g_D` (current) by the arm table above.

**Committed persistent transition (exactly one `optimizer.step()` per batch):** `g_H` is installed into `p.grad` for shared names; `g_P` for private names. The installed Adam performs, for every parameter with non-None installed grad:

```
m ← β1·m + (1−β1)·(g_H + λθ)
v ← β2·v + (1−β2)·(g_H + λθ)²
τ ← τ + 1
θ ← θ − lr/(1−β1^τ) · m / ( √( v/(1−β2^τ) ) + ε )
```

**Counterfactual applied delta (read-only, computed from the PRE-step state, discarded after use):**

```
Δ_D = AdamDelta(θ, s, g_D)   # = -lr/(1−β1^τ') · m̃ / ( √(ṽ/(1−β2^τ')) + ε )
m̃ = β1·m + (1−β1)·(g_D + λθ),  ṽ = β2·v + (1−β2)·(g_D + λθ)²,  τ' = τ+1
```

`m̃, ṽ` exist only inside the read-only computation and are **never written** to `optimizer.state`.

**Applied parameter move:**

- F, T: `H = D`, so `Δ_D ≡ Δ_H`; the parameter keeps exactly what `optimizer.step()` produced. **No overwrite.** This makes F/T bit-identical to the S3/S4 F/T update path.
- HT, DH: after `optimizer.step()` commits the `g_H` history transition, overwrite shared parameters only: `θ_G ← θ_before + Δ_D` (same pattern as S4 arm U, which overwrote with `θ_before + r_U·Δ_F`).

Private parameters always undergo the ordinary FULL Adam step from the same `optimizer.step()` call and are never overwritten. Architecture group `A_arch` (4 scalars, separate Adam, lr 0.001) follows the same H/D rule as `C`, matching S3/S4 where T applied target-only gradients to `A_arch`.

**Bias correction / step counter:** one counter per parameter, living only in the persistent optimizer, incremented exactly once per real step. `Δ_D` uses `τ' = τ_pre + 1`, i.e., the identical `t` that the committed history update uses. No second optimizer object, no shadow/counterfactual persistent state exists anywhere in the design.

**Reduction check:** for F (H=D=FULL) and T (H=D=TARGET) the committed transition and the applied move are ordinary Adam on the respective gradient stream — exactly the S3/S4 definitions.

**None-gradient semantics:** structurally, both losses reach every shared/arch parameter (M3OE applies all experts to the full batch), and S4's T arm completed without None-related failures. Therefore: if any shared/arch `g_F` or `g_T` is `None`, or if `g_F` is None while `g_T` is not, **fail closed** (run invalid). Per-step None counts are logged; any nonzero count aborts the run. Private gradient None is handled exactly as S3/S4.

## 3. Independent variables / Controlled variables

**Independent (the only intended differences between arms within a seed):** factor H (which gradient stream updates persistent `m/v/τ` on `C` and `A_arch`); factor D (which gradient is used as the current-step `g` in the applied `AdamDelta` on `C` and `A_arch`).

**Controlled (inherited frozen protocol, verified by pairing audit):** byte-identical KuaiRand data/split/view and row order; Equal coefficient table (`w_tab1=0.3834574423`, no renormalization after masking); M3OE forward source hash; parameter partition (C: 80 tensors/744,848 params; P: 54/869,406; A_arch: 4); semantic initial parameters per seed; batch size 40,960; shuffled concatenated loader, `drop_last=False`, `num_workers=0`; one stochastic forward per batch; batch order and exposure (batch fingerprints must be identical across the four arms within a seed); main Adam lr 0.01, arch Adam lr 0.001, betas (0.9, 0.999), coupled weight decay 1e-5, installed epsilon, no scheduler; one main step per batch, one arch step per epoch; private parameters always FULL; loss objective and target masking identical; 50 epochs; validation-only six-cell macro-AUC checkpoint selection with earliest tie; test deferred until after selection; checkpoint labels `selected/final/common{9,19,29,39,49}`; AUC and LogLoss on all six domain-task cells; two concurrent Python processes max.

## 4. Per-arm optimizer state flow (step by step)

Common per main batch (arch step analogous, group `A_arch`, arch optimizer):

1. Forward once → `L, L*`. `gradients_one_forward` → `g_F, g_T` (shared), `g_P` (private). No RNG consumption, no new forward.
2. Snapshot `θ_before` (shared) and private `θ_before_P` (detached clones).
3. **Read-only** from persistent state `s`: `Δ_F = AdamDelta(θ_before, s, g_F)`, `Δ_T = AdamDelta(θ_before, s, g_T)`, `Δ_P = AdamDelta(θ_before_P, s_P, g_P)`. Also log `‖m‖`, `‖v‖` (float64).
4. Install `p.grad ← g_H` (shared), `p.grad ← g_P` (private) as detached clones.
5. `optimizer.step()` — the **only** state mutation of the step. Persistent `m/v/τ` commit the `g_H` transition; parameters move by `Δ_H`.
6. Capture observed step delta `θ_after_step − θ_before`; assert it matches `Δ_H` (Invariant I2).
7. Arm-dependent finalization, then assertions and logging.

**F:** `g_H = g_D = g_F`. Step 5 is the whole update; no overwrite. Persistent = FULL-history; applied = FULL-current. Bit-identical to S4 arm F.

**T:** `g_H = g_D = g_T`. Same as F with target stream. Bit-identical to S4 arm T.

**HT (TARGET-history + FULL-current):** install `g_T` (step 4), so `optimizer.step()` commits the **target-stream** `m/v/τ` transition and moves params by `Δ_T`. Then overwrite shared params: `θ ← θ_before + Δ_F` (computed at step 3 from the pre-step target-history state and the FULL current gradient). The persistent state after the step is exactly what arm T's state update rule would produce from the same pre-state and `g_T`; the parameter instead received the FULL-current counterfactual move. The `Δ_T`-based move from `step()` is discarded from parameters only — never from state.

**DH (FULL-history + TARGET-current):** symmetric: install `g_F` (history commits FULL transition, params move by `Δ_F`), then overwrite shared params with `θ_before + Δ_T`.

Which state is what: the single per-group Adam `state` is the **persistent history** and is the only state ever updated (step 5, by `g_H` only). `AdamDelta`'s internal `m̃/ṽ` are a **counterfactual step calculator**, computed at step 3, applied at step 7, then dropped; it can never be updated and never writes back. No third state exists.

## 5. Required code changes (file- and function-level)

No modification to `stage_S3_*`, `stage_S4_*`, or `m3oe/` sources. `stage_S4_gradient_controls.py` is imported unchanged (its SHA-256 enters the S5 producer inventory).

1. **`experiments/stage_S5_protocol.py` (new).** Mirror `stage_S4_protocol.py`: `STAGE="S5"`, `ARMS=("F","T","HT","DH")`, `SEEDS=(2022,2023,2024,2025)`, `COMMON_EPOCHS=(9,19,29,39,49)`, `ARM_HISTORY={"F":"FULL","T":"TARGET","HT":"TARGET","DH":"FULL"}`, `ARM_CURRENT={"F":"FULL","T":"TARGET","HT":"FULL","DH":"TARGET"}`, `key(seed,arm)` → `"s5|equal|kuairand|natural|m3oe|seed{}|{}"`, `matrix()`, `waves()` → pairs `("F","T")` per seed then `("HT","DH")` per seed (8 two-process waves, F/T first for early replication review), `inventory()` = S3 inventory + S4 source files (unchanged, hash-pinned) + the four S5 files, `MOMENT_DUMP_EPOCHS=(0,1,9,19,29,39,49)`. Roots: `stage_S5_raw_results/`, `stage_S5_queue_manifest.json`, `stage_S5_outputs/`.
2. **`experiments/stage_S5_run.py` (new).** `S5Runner(S4Runner)`:
   - `__init__`: own arm→factor mapping; set `args.objective_weighting="equal"`; call the S3-chain init; keep `validate_adam_delta_against_clone(args.device)`; freeze S5 inventory and check `MODEL_SOURCE_SHA256`.
   - Override `_validate_args`: same frozen values as S4 (epoch 50, batch 40960, lr 0.01/0.001, wd 1e-5, expert_num 4, num_workers 0, device cuda:0, fcn_dims, shadow_patience 10), S5 arms/seeds/key, launch provenance fields required.
   - Override `_controlled_step(losses, fingerprint, x_batch, epoch, step, kind)`: implement §4 exactly, reusing `gradients_one_forward, adam_deltas, assert_vector_match, assert_match, cosine, vector_norm`. Hard-assert invariants I1–I7 (§9). Emit the S5 per-step row (§7).
   - Override `_access_formula`, `_condition_name` (`"s5_"+arm`), `_output_formal_protocol` (stage S5, arm, `history_source`, `current_source`).
   - Add `_dump_moment_state(epoch)`: at epochs in `MOMENT_DUMP_EPOCHS`, save `{name: (exp_avg, exp_avg_sq, step)}` for shared+arch params to `moment_state_epoch{e}.pth` (CPU, detached clones; read-only w.r.t. state).
   - Override `write_outputs`: mirror S4's (stage re-stamping of inherited CSVs, `gradient_step.csv`, `probe.csv` with S5 FIELDS, moment-dump files hashed into manifest, S5 protocol observed-vs-expected check, code inventory). Fail atomically on any mismatch.
   - `build_argparser()`: S4 parser with `--arm choices=S5 ARMS`.
   - `main()`: mirror S4's (interval/failure receipts).
3. **`experiments/stage_S5_orchestrator.py` (new).** Mirror the S4 A2 orchestrator: reviewer-PASS-gated freeze of queue + ≥21-entry source inventory; 8 waves × 2 concurrent processes with positive overlap evidence; append-only attempts; whole-wave single retry on explicit transient `OSError` only; `validate_run` extended: REQUIRED artifacts + the 7 moment dumps, S5 `gradient_step.csv`/`probe.csv` presence and field contract, zero recorded invariant violations, cross-arm batch-trace equality within seed (reuse S3 `validate_batch_trace` + S4 PAIR_FIELDS).
4. **`experiments/stage_S5_report.py` (new).** Collector mirroring S4's, emitting §8 files, including the cross-run moment-cosine audit table (§7) and contrast set `T−F, HT−F, DH−F, HT−T, DH−T, HT−DH`.
5. **`experiments/tests/test_stage_S5_controls.py`, `test_stage_S5_orchestration.py` (new).** Required tests:
   - Toy two-step hand-computed HT and DH trajectories vs implementation (parameters, `m/v/τ`, bias-correction `t`) on CPU and CUDA.
   - F and T bit-equivalence against a direct `loss.backward()+step()` reference on an identical model copy (pattern of `run_s3_update_smoke`).
   - Overwrite touches only intended names; private params and private Adam state equal to a FULL reference; exactly one step counted per batch for both optimizers.
   - `AdamDelta` clone-parity revalidation (inherited) plus a regression that `Δ_D` is computed from pre-step state (mutating state before `Δ_D` must change the result — guards ordering bugs).
   - RNG digest unchanged across a controlled step at probe steps; moment dumps do not mutate state (hash state before/after dump).
   - Orchestrator: freeze refusal without reviewer PASS, source-drift rejection, pairing/batch-trace mismatch rejection, process overlap, transient-retry classification, collector isolation.

## 6. Required configuration changes

None to data, model, objective, optimizer hyperparameters, seeds, or checkpoint policy. New only: S5 queue manifest + state, `stage_S5_raw_results/` (empty at freeze), `stage_S5_outputs/`, run dirs `stage_S5_raw_results/{arm}/seed{seed}/joint`, formal keys `s5|equal|kuairand|natural|m3oe|seed{2022..2025}|{F,T,HT,DH}`. No S3/S4 artifact is a formal S5 arm; S4 F/T metrics may appear only in a labelled historical-context table.

## 7. Required metrics/logging

Inherited per-epoch six-cell (d0/d1/d2 × t0/t1) train/val/test AUC + LogLoss, selected/final/common checkpoints, batch trace — unchanged.

Per-step `gradient_step.csv` (main + architecture rows), S5 fields: S4-compatible identity/loss/gradient fields (`batch_fingerprint, domain_counts, L_full, L_target, L_non, decomposition_residual, g_full_norm, g_target_norm, g_non_norm, gradient_cosine`) plus:
- `history_source, current_source` (arm factors);
- `m_norm_persistent, v_norm_persistent` (pre-step, float64);
- `delta_full_norm, delta_target_norm` (both read-only counterfactual norms, as S4);
- `delta_history_norm, delta_applied_norm, delta_observed_step_norm`;
- `applied_cosine_full, applied_cosine_target` (cos of final applied Δ vs `Δ_F` and vs `Δ_T`; must be 1.0 for the D-source within tolerance);
- `installed_cosine_history` (≥0.99999 hard);
- `history_formula_residual, applied_formula_residual, private_full_residual` (all ≤ `max(1e-6, 1e-4·reference)`);
- `adam_state_step_min/max`, `optimizer_steps`, `architecture_optimizer_steps`;
- `none_count_full, none_count_target` (any nonzero → abort);
- `zero_target_gradient_step` flag (record; if >1% of main steps in any arm/seed, mark arm non-identifying).

`probe.csv`: first main step of each epoch + each architecture step (100 rows/run), adding `delta_cosine` = cos(`Δ_F`,`Δ_T`), moment norms, and `rng_digest_before/after` around the controlled step (must be equal).

Moment dumps: `moment_state_epoch{0,1,9,19,29,39,49}.pth` per run. Collector emits `stage_S5_moment_audit.csv`: per seed, per dump epoch, float64 cosines `cos(m_HT,m_T), cos(m_HT,m_F), cos(m_DH,m_F), cos(m_DH,m_T)` and same for `v`, plus norms.

## 8. Expected output files

Per run: the 17 S4-required artifacts (config, manifest, partition, batch_trace, gradient_step, probe, metrics_epoch/test/summary, predictions_test.npz, objective coefficients, both optimizer states, rng state, initialization audit, nonfinite inventory, stdout, process_interval) + 7 `moment_state_epoch*.pth` + `failure.json` on failure.

Post-run (under `experiments/stage_S5_outputs/`): `stage_S5_runs.csv` (16 rows); `stage_S5_raw_results.csv` (16×6×7 = 672 rows); `stage_S5_paired_results.csv` (6 contrasts × 4 seeds × 6 cells × 7 labels = 1,008 rows); `stage_S5_gradient_steps.csv`; `stage_S5_probe.csv` (1,600 rows); `stage_S5_moment_audit.csv`; `stage_S5_aggregates.csv`; `stage_S5_summary_manifest.json`; `stage_S5_run_report.md` (raw facts only). Mechanism interpretation (Patterns A–E) is Main Agent's later analysis, not the runner's.

## 9. Audit invariants and failure conditions

Tolerance everywhere: `max(1e-6, 1e-4 × reference_norm)` (S4's `TOLERANCE_ABS/REL`), float64 for scalar norms.

Hard per-step invariants (violation ⇒ run invalid, fail-closed, no silent retry):
- **I1 History-stream identity:** installed shared grad vector-matches `g_H`; `installed_cosine_history ≥ 0.99999`.
- **I2 History formula:** observed post-`step()` pre-overwrite delta matches `AdamDelta(pre-state, g_H)`. Proves the persistent transition consumed exactly `g_H` under installed Adam semantics.
- **I3 Applied-source identity:** final applied shared delta matches `AdamDelta(pre-state, g_D)`. For HT this is the FULL counterfactual; for DH the TARGET one; for F/T it coincides with I2.
- **I4 Private integrity:** private deltas match `AdamDelta(private pre-state, g_P)`; private parameters never overwritten; private state equals a FULL reference in tests.
- **I5 Step counter:** exactly one `optimizer.step()` per batch per optimizer; `adam_state_step` advances by exactly 1 for all live params; counters equal across arms within seed (pairing audit).
- **I6 No counterfactual pollution:** `optimizer.state` is written only by step 5; tests prove `Δ_D` uses pre-step state and that overwrite/moment-dump paths never mutate state.
- **I7 RNG/DataLoader trajectory:** probe-step RNG digests before/after the controlled step equal; no additional forward, DataLoader, or iterator created by S5 code; architecture-batch path retains `preserve_rng_state` with restored-digest evidence.
- **I8 Batch identity:** batch-fingerprint sequences identical across the four arms within each seed (pairing audit; mismatch ⇒ `pairing_valid=false`, no analysis use).
- **I9 None-gradient:** any None in shared/arch `g_F`/`g_T` aborts the run.

Cross-run moment gate (reviewer-checked before accepting runs as identifying): at dump epochs 0 and 1, `cos(m_HT,m_T) > cos(m_HT,m_F)` **and** `cos(m_DH,m_F) > cos(m_DH,m_T)` (same for `v`). Rationale: identical init + identical batches make HT/T histories identical at step 0 and near-identical through early epochs, while F's history consumes `g_F = g_T + g_non`. Failure of this gate ⇒ the two factors were not demonstrably decoupled ⇒ **stage invalid**, reviewer block, no mechanism interpretation. Later epochs are informational (trajectories legitimately diverge).

Experiment-level failure/invalidity conditions: any I1–I9 violation; source/inventory drift after freeze; missing/corrupt artifact or moment dump; protocol observed≠expected; loss of two-process overlap evidence; more than one whole-wave retry; F/T fresh reproduction failure (mean `R_T < +0.003` on either d1 task, or <3/4 seeds positive) ⇒ mechanism classification **unresolved** (scientific null, not a code failure; matrix still completes). Pattern-E-looking results (HT≈DH≈T) must trigger Reviewer leakage audit before any interpretation.

## 10. Potential implementation pitfalls

1. **State-timing bug (highest risk):** computing `Δ_D` after `optimizer.step()` instead of before — this silently uses one-step-ahead moments and `τ+2` bias correction, and would contaminate both HT and DH in a way performance metrics cannot detect. Guard: I3 assertion + the ordering regression test.
2. **Overwrite leakage:** the HT/DH overwrite must cover exactly the group names (`C` for main, `A_arch` for arch step) and nothing else; touching private params or `optimizer.state` breaks I4/I6. Reuse S4 U-arm's exact loop shape.
3. **Confusing "FULL-current" with F's delta:** HT's applied delta is `AdamDelta(target-history state, g_F)`, **not** F's `Δ_F` (which uses FULL history). The numerator is 0.9·(target-history m) + 0.1·(FULL current) — that is the honest Adam decomposition of "current gradient source"; reviewers must verify the spec was read this way.
4. **Bitwise cross-run comparisons:** CUDA foreach dispatch is not guaranteed bitwise-reproducible across processes; all cross-run checks (batches excepted — fingerprints are data-level) use tolerances, never `torch.equal`.
5. **RNG side effects:** any added diagnostic that creates a DataLoader iterator or draws from global RNG changes the training trajectory (known project pitfall). This design adds none: autograd-only counterfactuals, state reads, CPU dumps. Keep it that way; probe-step digest assertions (I7) catch regressions.
6. **Moment-dump mutation/IO:** dumping must use detached CPU clones; dumping at epoch end must not reorder the training loop or checkpoint selection; hash dumps into the manifest.
7. **None-gradient assumption:** design fails closed on any None; if M3OE routing ever produces one, stop for amendment rather than improvising skip semantics.
8. **Inventory binding:** S5 modules subclass S4/S3 code; the producer inventory must pin the unchanged hashes of `stage_S4_gradient_controls.py`, `stage_S4_run.py`, `stage_S3_run.py`, and the M3OE forward (`a7702a64…` as of the S4 A2 review; recompute at freeze) in addition to the four new S5 files.
9. **Zero-target-gradient steps:** `g_T=0` is not undefined here (decay-coupled delta still defined), but >1% such steps makes an arm non-identifying — record, don't clamp.
10. **Subclass drift:** `S4Runner.__init__` hardcodes S4 arm mapping/protocol; S5 must override every arm-dependent method (`_validate_args`, `_controlled_step`, `_access_formula`, `_condition_name`, `_output_formal_protocol`, `write_outputs`, FIELDS) — a silent inherit of any one of them mixes S4 N/U semantics into S5. Tests must assert the arm→factor map on all four arms.

---

## Designer delivery summary (for main agent)

- **HT state flow:** per step compute `g_F/g_T` from one forward; install `g_T` into shared `.grad` and call the single `optimizer.step()` so persistent `m/v/τ` strictly follow the target stream (params transiently move by `Δ_T`); then overwrite shared params with `θ_before + Δ_F`, where `Δ_F` is the read-only counterfactual delta computed BEFORE the step from the same persistent (target-history) state + FULL current gradient. State is updated only by `g_T`; applied move comes only from `g_F`; both use the same `τ+1` bias correction; no shadow optimizer.
- **DH state flow:** fully symmetric — install `g_F` for `step()` (persistent history follows FULL stream), then overwrite params with `θ_before + Δ_T` (pre-step FULL-history state + TARGET current gradient counterfactual).
- **F/T identical to S3/S4:** when H=D, `Δ_D ≡ Δ_H`, no overwrite; update path bit-identical to S4 F/T.
- **Largest implementation risk:** the state-timing of `Δ_D` — it must be computed from pre-step state; computing it after `optimizer.step()` silently uses wrong moments and bias-correction step, invisible to performance metrics. Triple guard: I3 per-step hard assertion, a dedicated ordering regression test (mutating state before `Δ_D` must change its result), and the epoch 0/1 cross-run moment-cosine hard gate (proving HT history ≈ T, DH history ≈ F). Secondary point: HT's "FULL-current" is the FULL counterfactual under target-history state (numerator still 90% target history), not F's delta — Reviewer must audit under this reading to avoid false leakage calls.
