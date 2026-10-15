# Stage S6-R2 — KuaiRand DH Selected-Solution Analysis

**Audience:** main agent. **Scope:** retrospective robustness + parameter-swap evaluation of KuaiRand DH. MovieLens included only as a negative control (per spec §9).

**Auto-generated data sections (§§1–6, 11)** produced by `s6r2_report.py` from runner outputs; **verdict sections (§0, §§7–10)** written by the main agent after completion of all KuaiRand + MovieLens runs (seeds 2022–2025).

---

## 0. Executive verdict (main agent)

**The KuaiRand DH selected-solution advantage is a ROBUST-SELECTION phenomenon on the test side, but it is a *generalization* phenomenon, not a *validation-optimum* phenomenon — and this distinction is the stage's central finding.**

1. **Test side robust.** Under canonical R0, ΔNonTarget = **+0.0068 (4/4 seeds positive, 3/4 ≥ +0.005)**, ΔMacro = +0.0044 (4/4 positive), ΔTarget = −0.0004 (§2, kuairand R0). The same qualitative result holds under three independent non-circular validation-selection rules in the A-grid sensitivity variant: R1 (target-selected) ΔN = +0.0041, R3 (worst-domain) ΔN = +0.0060, R4 (sample-weighted) ΔN = +0.0041 — each 3/4 seeds positive, each with ΔT > −0.003 (§2 gate summary, test_grid rows). This rules out H2 (selection-objective specificity) and H6 (narrow peak) as primary explanations.

2. **Validation side null-to-negative.** At the very same selected checkpoints, the *validation* gap is reversed: R0 val ΔN = **−0.0039** (1/4 positive), val ΔM = −0.0027. Under every rule R0–R4 the val-layer ΔN is ≤ +0.0007 with ≤2/4 seeds positive. F reaches the higher validation peaks; DH's selected checkpoints simply lose much less when moving from validation to test on non-target cells. **A practitioner watching only validation would never see this advantage** — and would in fact prefer F.

3. **Timing vs policy (Part B).** On the validation layer the decomposition is anchor-dependent to the point of sign reversal: for N, PolicyAtF = −0.0111 while PolicyAtDH = +0.0082 (both 4-seed means), with timing terms of comparable or larger magnitude (§3). The same-epoch policy difference is therefore not a stable quantity on validation; epoch-alignment dominates. On test, cross-epoch decomposition is impossible with saved artifacts (all canonical epochs off-grid), but the same-epoch grid trajectory Δ(e) is positive at 16/20 seed×epoch points, up to +0.018 — suggesting a real but heterogeneous same-stage test-side policy effect underneath the timing component.

4. **Parameter localization impossible (Part D).** Both cross-partition hybrids collapse catastrophically (macro 0.52–0.58 vs parents 0.67–0.68; retention ρ_N ∈ [−15.4, −8.0] for usable seeds), while F_hybrid ≡ F_full (ρ ≈ 0, the 4 architecture tensors are inert). Per spec §8 this is flagged as **dominant cross-partition co-adaptation**: the selected solutions of F and DH are internally co-adapted objects; the gain cannot be assigned to either the shared or the private partition.

5. **MovieLens negative control confirms the cross-dataset distinction.** No selection rule produces a stable MovieLens selected benefit (all test ΔN ∈ [−0.002, +0.003], R3 grid even negative), despite large same-epoch late-training advantages (grid Δ(e) up to +0.033). MovieLens selects at epochs 7–13, long before the late DH advantage exists. MovieLens = late-training phenomenon that selection never captures; KuaiRand = selected-solution phenomenon. The prior interpretation stands.

**Gate decision (§9): GO — not STOP.** Conditions 1–4 are broadly satisfied; mechanism localization is genuinely ambiguous. Per spec §11, exactly **one minimal mechanism experiment** is licensed (dense-checkpoint evaluation replication; no method design). Details in §9–§10.

## 1. Data availability & integrity

All inputs are read-only (`experiments/stage_S6_raw_results`, `experiments/stage_S6ML_raw_results`). No training was performed.

### Integrity report (machine-readable)

| dataset | arm/seed | canonical epoch | R0 matches flag? | npz vs csv max diff | final vs ce49 max |
|---|---|---|---|---|---|
| kuairand | F_seed2022 | 30 | yes | 0.00e+00 | 0.00e+00 |
| kuairand | F_seed2023 | 47 | yes | 0.00e+00 | 0.00e+00 |
| kuairand | F_seed2024 | 41 | yes | 0.00e+00 | 0.00e+00 |
| kuairand | F_seed2025 | 24 | yes | 0.00e+00 | 0.00e+00 |
| kuairand | DH_seed2022 | 40 | yes | 0.00e+00 | 0.00e+00 |
| kuairand | DH_seed2023 | 30 | yes | 0.00e+00 | 0.00e+00 |
| kuairand | DH_seed2024 | 48 | yes | 0.00e+00 | 0.00e+00 |
| kuairand | DH_seed2025 | 27 | yes | 0.00e+00 | 0.00e+00 |
| movielens | F_seed2022 | 7 | yes | 0.00e+00 | 0.00e+00 |
| movielens | F_seed2023 | 8 | yes | 0.00e+00 | 0.00e+00 |
| movielens | F_seed2024 | 8 | yes | 0.00e+00 | 0.00e+00 |
| movielens | F_seed2025 | 12 | yes | 0.00e+00 | 0.00e+00 |
| movielens | DH_seed2022 | 8 | yes | 0.00e+00 | 0.00e+00 |
| movielens | DH_seed2023 | 13 | yes | 0.00e+00 | 0.00e+00 |
| movielens | DH_seed2024 | 8 | yes | 0.00e+00 | 0.00e+00 |
| movielens | DH_seed2025 | 7 | yes | 0.00e+00 | 0.00e+00 |

Full machine-readable entries: `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/s6r2_integrity_report.json`.

**Part D integrity (KuaiRand only):**
- *kuairand*: 16 seed runs; partition / state-dict parity / anchor reproduction results in JSON.

---

## 2. Selection-rule robustness (Part A)

Five pre-registered rules R0–R4 (R0 = canonical six-cell macro, R1 = target, R2 = non-target, R3 = worst-domain, R4 = val sample-weighted). Tie-break: earliest epoch. Test data is **never** used for selection. Test metrics for R0 are exact (selected checkpoint saved); other rules often need test data at non-saved epochs and are reported as **NA** with `test_available=0`. The A-grid variant restricts selection to `{9,19,29,39,49} ∪ {e_R0}` so every rule has a test value; it is explicitly flagged as a sensitivity variant.

### Dataset: kuairand

### Rule R0

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.6768 | 0.6729 | -0.0039 ± 0.0081 | 1/4 (n=4/4) | 3 | 3 |
| val | T | 0.7107 | 0.7104 | -0.0003 ± 0.0032 | 2/4 (n=4/4) | 1 | 0 |
| val | M | 0.6881 | 0.6854 | -0.0027 ± 0.0064 | 2/4 (n=4/4) | 3 | 2 |
| test | N | 0.6536 | 0.6604 | +0.0068 ± 0.0046 | 4/4 (n=4/4) | 3 | 3 |
| test | T | 0.7118 | 0.7114 | -0.0004 ± 0.0031 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.6730 | 0.6774 | +0.0044 ± 0.0029 | 4/4 (n=4/4) | 3 | 1 |

### Rule R1

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.6640 | 0.6637 | -0.0003 ± 0.0172 | 1/4 (n=4/4) | 3 | 3 |
| val | T | 0.7138 | 0.7129 | -0.0009 ± 0.0009 | 1/4 (n=4/4) | 0 | 0 |
| val | M | 0.6806 | 0.6801 | -0.0005 ± 0.0112 | 1/4 (n=4/4) | 3 | 2 |
| test | N | NA | NA | NA (no test for seed) | - | - | - |
| test | T | NA | NA | NA (no test for seed) | - | - | - |
| test | M | NA | NA | NA (no test for seed) | - | - | - |

### Rule R2

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.6768 | 0.6734 | -0.0034 ± 0.0082 | 1/4 (n=4/4) | 3 | 3 |
| val | T | 0.7107 | 0.7090 | -0.0017 ± 0.0035 | 1/4 (n=4/4) | 2 | 0 |
| val | M | 0.6881 | 0.6853 | -0.0028 ± 0.0064 | 2/4 (n=4/4) | 3 | 2 |
| test | N | 0.6508 | 0.6553 | +0.0045 ± 0.0064 | 2/2 (n=2/4) | 1 | 1 |
| test | T | 0.7115 | 0.7099 | -0.0016 ± 0.0044 | 1/2 (n=2/4) | 1 | 0 |
| test | M | 0.6710 | 0.6735 | +0.0025 ± 0.0028 | 2/2 (n=2/4) | 1 | 0 |

### Rule R3

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.6743 | 0.6734 | -0.0009 ± 0.0080 | 2/4 (n=4/4) | 3 | 3 |
| val | T | 0.7102 | 0.7090 | -0.0013 ± 0.0032 | 1/4 (n=4/4) | 1 | 0 |
| val | M | 0.6863 | 0.6853 | -0.0010 ± 0.0061 | 2/4 (n=4/4) | 3 | 3 |
| test | N | NA | NA | NA (no test for seed) | - | - | - |
| test | T | NA | NA | NA (no test for seed) | - | - | - |
| test | M | NA | NA | NA (no test for seed) | - | - | - |

### Rule R4

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.6640 | 0.6646 | +0.0007 ± 0.0160 | 1/4 (n=4/4) | 3 | 3 |
| val | T | 0.7138 | 0.7129 | -0.0009 ± 0.0008 | 1/4 (n=4/4) | 0 | 0 |
| val | M | 0.6806 | 0.6807 | +0.0001 ± 0.0105 | 1/4 (n=4/4) | 3 | 2 |
| test | N | 0.6475 | 0.6503 | NA | 1/1 (n=1/4) | 0 | 0 |
| test | T | 0.7152 | 0.7145 | NA | 0/1 (n=1/4) | 0 | 0 |
| test | M | 0.6701 | 0.6717 | NA | 1/1 (n=1/4) | 0 | 0 |

**A-grid sensitivity variant (selection restricted to {9,19,29,39,49} ∪ {e_R0}; NOT the primary rule — spec §4.3):** test layer shown only; the corresponding val-layer objective values are in `s6r2_partA_grid_epochs_*.csv`.

### Rule R0

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.6536 | 0.6604 | +0.0068 ± 0.0046 | 4/4 (n=4/4) | 3 | 3 |
| test | T | 0.7118 | 0.7114 | -0.0004 ± 0.0031 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.6730 | 0.6774 | +0.0044 ± 0.0029 | 4/4 (n=4/4) | 3 | 1 |

### Rule R1

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.6515 | 0.6556 | +0.0041 ± 0.0071 | 3/4 (n=4/4) | 3 | 1 |
| test | T | 0.7146 | 0.7136 | -0.0009 ± 0.0003 | 0/4 (n=4/4) | 0 | 0 |
| test | M | 0.6725 | 0.6750 | +0.0024 ± 0.0046 | 3/4 (n=4/4) | 1 | 1 |

### Rule R2

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.6536 | 0.6604 | +0.0068 ± 0.0046 | 4/4 (n=4/4) | 3 | 3 |
| test | T | 0.7118 | 0.7114 | -0.0004 ± 0.0031 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.6730 | 0.6774 | +0.0044 ± 0.0029 | 4/4 (n=4/4) | 3 | 1 |

### Rule R3

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.6540 | 0.6600 | +0.0060 ± 0.0050 | 3/4 (n=4/4) | 3 | 3 |
| test | T | 0.7134 | 0.7110 | -0.0024 ± 0.0026 | 1/4 (n=4/4) | 2 | 0 |
| test | M | 0.6738 | 0.6770 | +0.0032 ± 0.0039 | 3/4 (n=4/4) | 3 | 1 |

### Rule R4

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.6515 | 0.6556 | +0.0041 ± 0.0071 | 3/4 (n=4/4) | 3 | 1 |
| test | T | 0.7146 | 0.7136 | -0.0009 ± 0.0003 | 0/4 (n=4/4) | 0 | 0 |
| test | M | 0.6725 | 0.6750 | +0.0024 ± 0.0046 | 3/4 (n=4/4) | 1 | 1 |

### kuairand — gate summary (ΔDH−F)

| rule | metric | layer | n_seeds | mean | SD | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|---|
| R0 | M | test | 4 | +0.0044 | 0.0029 | 4 | 3 | 1 |
| R0 | M | test_grid | 4 | +0.0044 | 0.0029 | 4 | 3 | 1 |
| R0 | M | val | 4 | -0.0027 | 0.0064 | 2 | 3 | 2 |
| R0 | N | test | 4 | +0.0068 | 0.0046 | 4 | 3 | 3 |
| R0 | N | test_grid | 4 | +0.0068 | 0.0046 | 4 | 3 | 3 |
| R0 | N | val | 4 | -0.0039 | 0.0081 | 1 | 3 | 3 |
| R0 | T | test | 4 | -0.0004 | 0.0031 | 2 | 1 | 0 |
| R0 | T | test_grid | 4 | -0.0004 | 0.0031 | 2 | 1 | 0 |
| R0 | T | val | 4 | -0.0003 | 0.0032 | 2 | 1 | 0 |
| R1 | M | test | 0 | +nan | nan | 0 | 0 | 0 |
| R1 | M | test_grid | 4 | +0.0024 | 0.0046 | 3 | 1 | 1 |
| R1 | M | val | 4 | -0.0005 | 0.0112 | 1 | 3 | 2 |
| R1 | N | test | 0 | +nan | nan | 0 | 0 | 0 |
| R1 | N | test_grid | 4 | +0.0041 | 0.0071 | 3 | 3 | 1 |
| R1 | N | val | 4 | -0.0003 | 0.0172 | 1 | 3 | 3 |
| R1 | T | test | 0 | +nan | nan | 0 | 0 | 0 |
| R1 | T | test_grid | 4 | -0.0009 | 0.0003 | 0 | 0 | 0 |
| R1 | T | val | 4 | -0.0009 | 0.0009 | 1 | 0 | 0 |
| R2 | M | test | 2 | +0.0025 | 0.0028 | 2 | 1 | 0 |
| R2 | M | test_grid | 4 | +0.0044 | 0.0029 | 4 | 3 | 1 |
| R2 | M | val | 4 | -0.0028 | 0.0064 | 2 | 3 | 2 |
| R2 | N | test | 2 | +0.0045 | 0.0064 | 2 | 1 | 1 |
| R2 | N | test_grid | 4 | +0.0068 | 0.0046 | 4 | 3 | 3 |
| R2 | N | val | 4 | -0.0034 | 0.0082 | 1 | 3 | 3 |
| R2 | T | test | 2 | -0.0016 | 0.0044 | 1 | 1 | 0 |
| R2 | T | test_grid | 4 | -0.0004 | 0.0031 | 2 | 1 | 0 |
| R2 | T | val | 4 | -0.0017 | 0.0035 | 1 | 2 | 0 |
| R3 | M | test | 0 | +nan | nan | 0 | 0 | 0 |
| R3 | M | test_grid | 4 | +0.0032 | 0.0039 | 3 | 3 | 1 |
| R3 | M | val | 4 | -0.0010 | 0.0061 | 2 | 3 | 3 |
| R3 | N | test | 0 | +nan | nan | 0 | 0 | 0 |
| R3 | N | test_grid | 4 | +0.0060 | 0.0050 | 3 | 3 | 3 |
| R3 | N | val | 4 | -0.0009 | 0.0080 | 2 | 3 | 3 |
| R3 | T | test | 0 | +nan | nan | 0 | 0 | 0 |
| R3 | T | test_grid | 4 | -0.0024 | 0.0026 | 1 | 2 | 0 |
| R3 | T | val | 4 | -0.0013 | 0.0032 | 1 | 1 | 0 |
| R4 | M | test | 1 | +0.0016 | nan | 1 | 0 | 0 |
| R4 | M | test_grid | 4 | +0.0024 | 0.0046 | 3 | 1 | 1 |
| R4 | M | val | 4 | +0.0001 | 0.0105 | 1 | 3 | 2 |
| R4 | N | test | 1 | +0.0028 | nan | 1 | 0 | 0 |
| R4 | N | test_grid | 4 | +0.0041 | 0.0071 | 3 | 3 | 1 |
| R4 | N | val | 4 | +0.0007 | 0.0160 | 1 | 3 | 3 |
| R4 | T | test | 1 | -0.0007 | nan | 0 | 0 | 0 |
| R4 | T | test_grid | 4 | -0.0009 | 0.0003 | 0 | 0 | 0 |
| R4 | T | val | 4 | -0.0009 | 0.0008 | 1 | 0 | 0 |



### Dataset: movielens

### Rule R0

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.7263 | 0.7240 | -0.0023 ± 0.0021 | 0/4 (n=4/4) | 1 | 1 |
| val | T | 0.7398 | 0.7398 | -0.0000 ± 0.0008 | 2/4 (n=4/4) | 0 | 0 |
| val | M | 0.7308 | 0.7293 | -0.0016 ± 0.0016 | 0/4 (n=4/4) | 1 | 0 |
| test | N | 0.7343 | 0.7343 | -0.0000 ± 0.0031 | 2/4 (n=4/4) | 1 | 0 |
| test | T | 0.7307 | 0.7307 | +0.0001 ± 0.0026 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.7331 | 0.7331 | +0.0000 ± 0.0029 | 2/4 (n=4/4) | 1 | 0 |

### Rule R1

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.7231 | 0.7222 | -0.0009 ± 0.0061 | 1/4 (n=4/4) | 3 | 2 |
| val | T | 0.7404 | 0.7409 | +0.0005 ± 0.0016 | 1/4 (n=4/4) | 0 | 0 |
| val | M | 0.7289 | 0.7284 | -0.0005 ± 0.0040 | 1/4 (n=4/4) | 2 | 1 |
| test | N | 0.7301 | 0.7326 | NA | 1/1 (n=1/4) | 0 | 0 |
| test | T | 0.7343 | 0.7338 | NA | 0/1 (n=1/4) | 0 | 0 |
| test | M | 0.7315 | 0.7330 | NA | 1/1 (n=1/4) | 0 | 0 |

### Rule R2

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.7263 | 0.7240 | -0.0023 ± 0.0021 | 0/4 (n=4/4) | 1 | 1 |
| val | T | 0.7398 | 0.7397 | -0.0001 ± 0.0007 | 1/4 (n=4/4) | 0 | 0 |
| val | M | 0.7308 | 0.7292 | -0.0016 ± 0.0016 | 0/4 (n=4/4) | 1 | 0 |
| test | N | 0.7343 | 0.7341 | -0.0002 ± 0.0029 | 2/4 (n=4/4) | 1 | 0 |
| test | T | 0.7307 | 0.7306 | -0.0001 ± 0.0024 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.7331 | 0.7329 | -0.0002 ± 0.0027 | 2/4 (n=4/4) | 1 | 0 |

### Rule R3

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.7252 | 0.7221 | -0.0031 ± 0.0037 | 1/4 (n=4/4) | 2 | 1 |
| val | T | 0.7388 | 0.7371 | -0.0018 ± 0.0055 | 1/4 (n=4/4) | 4 | 2 |
| val | M | 0.7298 | 0.7271 | -0.0027 ± 0.0038 | 1/4 (n=4/4) | 3 | 1 |
| test | N | NA | NA | NA (no test for seed) | - | - | - |
| test | T | NA | NA | NA (no test for seed) | - | - | - |
| test | M | NA | NA | NA (no test for seed) | - | - | - |

### Rule R4

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| val | N | 0.7263 | 0.7239 | -0.0025 ± 0.0020 | 0/4 (n=4/4) | 1 | 1 |
| val | T | 0.7398 | 0.7400 | +0.0002 ± 0.0010 | 2/4 (n=4/4) | 0 | 0 |
| val | M | 0.7308 | 0.7292 | -0.0016 ± 0.0016 | 0/4 (n=4/4) | 1 | 0 |
| test | N | 0.7357 | 0.7350 | -0.0007 ± 0.0034 | 1/3 (n=3/4) | 1 | 0 |
| test | T | 0.7295 | 0.7289 | -0.0006 ± 0.0027 | 1/3 (n=3/4) | 1 | 0 |
| test | M | 0.7336 | 0.7330 | -0.0007 ± 0.0032 | 1/3 (n=3/4) | 1 | 0 |

**A-grid sensitivity variant (selection restricted to {9,19,29,39,49} ∪ {e_R0}; NOT the primary rule — spec §4.3):** test layer shown only; the corresponding val-layer objective values are in `s6r2_partA_grid_epochs_*.csv`.

### Rule R0

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.7343 | 0.7343 | -0.0000 ± 0.0031 | 2/4 (n=4/4) | 1 | 0 |
| test | T | 0.7307 | 0.7307 | +0.0001 ± 0.0026 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.7331 | 0.7331 | +0.0000 ± 0.0029 | 2/4 (n=4/4) | 1 | 0 |

### Rule R1

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.7348 | 0.7333 | -0.0015 ± 0.0039 | 2/4 (n=4/4) | 2 | 1 |
| test | T | 0.7310 | 0.7296 | -0.0015 ± 0.0018 | 1/4 (n=4/4) | 1 | 0 |
| test | M | 0.7335 | 0.7321 | -0.0015 ± 0.0031 | 2/4 (n=4/4) | 2 | 0 |

### Rule R2

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.7343 | 0.7341 | -0.0002 ± 0.0029 | 2/4 (n=4/4) | 1 | 0 |
| test | T | 0.7307 | 0.7306 | -0.0001 ± 0.0024 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.7331 | 0.7329 | -0.0002 ± 0.0027 | 2/4 (n=4/4) | 1 | 0 |

### Rule R3

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.7343 | 0.7324 | -0.0019 ± 0.0036 | 1/4 (n=4/4) | 1 | 1 |
| test | T | 0.7307 | 0.7264 | -0.0043 ± 0.0063 | 1/4 (n=4/4) | 2 | 2 |
| test | M | 0.7331 | 0.7304 | -0.0027 ± 0.0044 | 1/4 (n=4/4) | 2 | 1 |

### Rule R4

| layer | metric | F (mean) | DH (mean) | ΔDH − F (4-seed mean ± SD) | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|
| test | N | 0.7343 | 0.7343 | -0.0000 ± 0.0031 | 2/4 (n=4/4) | 1 | 0 |
| test | T | 0.7307 | 0.7307 | +0.0001 ± 0.0026 | 2/4 (n=4/4) | 1 | 0 |
| test | M | 0.7331 | 0.7331 | +0.0000 ± 0.0029 | 2/4 (n=4/4) | 1 | 0 |

### movielens — gate summary (ΔDH−F)

| rule | metric | layer | n_seeds | mean | SD | n_pos | n_ge_003 | n_ge_005 |
|---|---|---|---|---|---|---|---|---|
| R0 | M | test | 4 | +0.0000 | 0.0029 | 2 | 1 | 0 |
| R0 | M | test_grid | 4 | +0.0000 | 0.0029 | 2 | 1 | 0 |
| R0 | M | val | 4 | -0.0016 | 0.0016 | 0 | 1 | 0 |
| R0 | N | test | 4 | -0.0000 | 0.0031 | 2 | 1 | 0 |
| R0 | N | test_grid | 4 | -0.0000 | 0.0031 | 2 | 1 | 0 |
| R0 | N | val | 4 | -0.0023 | 0.0021 | 0 | 1 | 1 |
| R0 | T | test | 4 | +0.0001 | 0.0026 | 2 | 1 | 0 |
| R0 | T | test_grid | 4 | +0.0001 | 0.0026 | 2 | 1 | 0 |
| R0 | T | val | 4 | -0.0000 | 0.0008 | 2 | 0 | 0 |
| R1 | M | test | 1 | +0.0015 | nan | 1 | 0 | 0 |
| R1 | M | test_grid | 4 | -0.0015 | 0.0031 | 2 | 2 | 0 |
| R1 | M | val | 4 | -0.0005 | 0.0040 | 1 | 2 | 1 |
| R1 | N | test | 1 | +0.0025 | nan | 1 | 0 | 0 |
| R1 | N | test_grid | 4 | -0.0015 | 0.0039 | 2 | 2 | 1 |
| R1 | N | val | 4 | -0.0009 | 0.0061 | 1 | 3 | 2 |
| R1 | T | test | 1 | -0.0005 | nan | 0 | 0 | 0 |
| R1 | T | test_grid | 4 | -0.0015 | 0.0018 | 1 | 1 | 0 |
| R1 | T | val | 4 | +0.0005 | 0.0016 | 1 | 0 | 0 |
| R2 | M | test | 4 | -0.0002 | 0.0027 | 2 | 1 | 0 |
| R2 | M | test_grid | 4 | -0.0002 | 0.0027 | 2 | 1 | 0 |
| R2 | M | val | 4 | -0.0016 | 0.0016 | 0 | 1 | 0 |
| R2 | N | test | 4 | -0.0002 | 0.0029 | 2 | 1 | 0 |
| R2 | N | test_grid | 4 | -0.0002 | 0.0029 | 2 | 1 | 0 |
| R2 | N | val | 4 | -0.0023 | 0.0021 | 0 | 1 | 1 |
| R2 | T | test | 4 | -0.0001 | 0.0024 | 2 | 1 | 0 |
| R2 | T | test_grid | 4 | -0.0001 | 0.0024 | 2 | 1 | 0 |
| R2 | T | val | 4 | -0.0001 | 0.0007 | 1 | 0 | 0 |
| R3 | M | test | 0 | +nan | nan | 0 | 0 | 0 |
| R3 | M | test_grid | 4 | -0.0027 | 0.0044 | 1 | 2 | 1 |
| R3 | M | val | 4 | -0.0027 | 0.0038 | 1 | 3 | 1 |
| R3 | N | test | 0 | +nan | nan | 0 | 0 | 0 |
| R3 | N | test_grid | 4 | -0.0019 | 0.0036 | 1 | 1 | 1 |
| R3 | N | val | 4 | -0.0031 | 0.0037 | 1 | 2 | 1 |
| R3 | T | test | 0 | +nan | nan | 0 | 0 | 0 |
| R3 | T | test_grid | 4 | -0.0043 | 0.0063 | 1 | 2 | 2 |
| R3 | T | val | 4 | -0.0018 | 0.0055 | 1 | 4 | 2 |
| R4 | M | test | 3 | -0.0007 | 0.0032 | 1 | 1 | 0 |
| R4 | M | test_grid | 4 | +0.0000 | 0.0029 | 2 | 1 | 0 |
| R4 | M | val | 4 | -0.0016 | 0.0016 | 0 | 1 | 0 |
| R4 | N | test | 3 | -0.0007 | 0.0034 | 1 | 1 | 0 |
| R4 | N | test_grid | 4 | -0.0000 | 0.0031 | 2 | 1 | 0 |
| R4 | N | val | 4 | -0.0025 | 0.0020 | 0 | 1 | 1 |
| R4 | T | test | 3 | -0.0006 | 0.0027 | 1 | 1 | 0 |
| R4 | T | test_grid | 4 | +0.0001 | 0.0026 | 2 | 1 | 0 |
| R4 | T | val | 4 | +0.0002 | 0.0010 | 2 | 0 | 0 |



---

## 3. Algebraic decomposition (Part B)

### Val-layer decomposition

Identities |ΔSel − (Policy + Timing)| < 1e-9 are asserted programmatically. Anchors: F-anchor decomposes the gap into `PolicyAtF` (same F checkpoint) + `TimingAlongDH` (DH from e_F to e_DH); DH-anchor is the symmetric form.

### Dataset: kuairand

**Val layer decomposition (4-seed mean ± SD of ΔSel / Policy / Timing, identity residual max-abs across all cells):**

| anchor | metric | ΔSel (mean ± SD) | Policy (mean ± SD) | Timing (mean ± SD) | max |residual| |
|---|---|---|---|---|---|
| F | M | -0.0027 ± 0.0064 | -0.0081 ± 0.0085 | +0.0054 ± 0.0024 | 0.0e+00 |
| F | T | -0.0003 ± 0.0032 | -0.0019 ± 0.0020 | +0.0016 ± 0.0052 | 0.0e+00 |
| F | N | -0.0039 ± 0.0081 | -0.0111 ± 0.0136 | +0.0073 ± 0.0061 | 0.0e+00 |
| F | D0 | -0.0111 ± 0.0121 | -0.0209 ± 0.0259 | +0.0098 ± 0.0139 | 0.0e+00 |
| F | D2 | +0.0034 ± 0.0062 | -0.0014 ± 0.0042 | +0.0047 ± 0.0025 | 0.0e+00 |
| F | d0t0 | -0.0115 ± 0.0180 | -0.0178 ± 0.0306 | +0.0063 ± 0.0136 | 0.0e+00 |
| F | d0t1 | -0.0106 ± 0.0109 | -0.0239 ± 0.0231 | +0.0133 ± 0.0148 | 0.0e+00 |
| F | d1t0 | -0.0003 ± 0.0036 | -0.0018 ± 0.0019 | +0.0015 ± 0.0053 | 0.0e+00 |
| F | d1t1 | -0.0004 ± 0.0029 | -0.0021 ± 0.0021 | +0.0017 ± 0.0050 | 0.0e+00 |
| F | d2t0 | +0.0021 ± 0.0066 | -0.0018 ± 0.0043 | +0.0040 ± 0.0039 | 0.0e+00 |
| F | d2t1 | +0.0046 ± 0.0060 | -0.0009 ± 0.0044 | +0.0055 ± 0.0025 | 0.0e+00 |
| DH | M | -0.0027 ± 0.0064 | +0.0053 ± 0.0072 | -0.0080 ± 0.0045 | 0.0e+00 |
| DH | T | -0.0003 ± 0.0032 | -0.0004 ± 0.0014 | +0.0000 ± 0.0038 | 0.0e+00 |
| DH | N | -0.0039 ± 0.0081 | +0.0082 ± 0.0108 | -0.0120 ± 0.0063 | 0.0e+00 |
| DH | D0 | -0.0111 ± 0.0121 | +0.0110 ± 0.0165 | -0.0220 ± 0.0081 | 0.0e+00 |
| DH | D2 | +0.0034 ± 0.0062 | +0.0054 ± 0.0087 | -0.0020 ± 0.0061 | 0.0e+00 |
| DH | d0t0 | -0.0115 ± 0.0180 | +0.0004 ± 0.0180 | -0.0119 ± 0.0110 | 0.0e+00 |
| DH | d0t1 | -0.0106 ± 0.0109 | +0.0215 ± 0.0202 | -0.0321 ± 0.0095 | 0.0e+00 |
| DH | d1t0 | -0.0003 ± 0.0036 | -0.0005 ± 0.0018 | +0.0002 ± 0.0038 | 0.0e+00 |
| DH | d1t1 | -0.0004 ± 0.0029 | -0.0002 ± 0.0011 | -0.0002 ± 0.0037 | 0.0e+00 |
| DH | d2t0 | +0.0021 ± 0.0066 | +0.0033 ± 0.0097 | -0.0011 ± 0.0065 | 0.0e+00 |
| DH | d2t1 | +0.0046 ± 0.0060 | +0.0074 ± 0.0081 | -0.0028 ± 0.0059 | 0.0e+00 |

**Test-layer decomposition availability (per §3.3):**

> Existing artifacts are insufficient for test-layer cross-epoch decomposition.

- {'endpoints_only_no_cross': 88}
- Per §2.6, all 8 (arm,seed) combinations have canonical selected epoch off-grid; test-layer cross-epoch decomposition requires both e_F and e_DH to be in {9,19,29,39,49} ∪ {selected}, which is only possible when two arms happen to share the same canonical epoch (degenerate, cross-terms ≡ 0).

### Same-epoch Δ(e) on the common checkpoint grid (test layer)

**Non-target mean Δ(e) = (DH − F) at saved grid epochs:**

| seed | e=9 | e=19 | e=29 | e=39 | e=49 |
|---|---|---|---|---|---|
| 2022 | -0.0140 | +0.0002 | +0.0120 | +0.0182 | +0.0102 |
| 2023 | +0.0111 | +0.0024 | +0.0111 | +0.0032 | +0.0028 |
| 2024 | +0.0025 | +0.0041 | +0.0006 | -0.0132 | +0.0043 |
| 2025 | -0.0163 | +0.0045 | +0.0118 | +0.0052 | -0.0039 |


### Dataset: movielens

**Val layer decomposition (4-seed mean ± SD of ΔSel / Policy / Timing, identity residual max-abs across all cells):**

| anchor | metric | ΔSel (mean ± SD) | Policy (mean ± SD) | Timing (mean ± SD) | max |residual| |
|---|---|---|---|---|---|
| F | M | -0.0016 ± 0.0016 | -0.0018 ± 0.0016 | +0.0002 ± 0.0004 | 0.0e+00 |
| F | T | -0.0000 ± 0.0008 | +0.0001 ± 0.0010 | -0.0001 ± 0.0008 | 0.0e+00 |
| F | N | -0.0023 ± 0.0021 | -0.0027 ± 0.0021 | +0.0004 ± 0.0003 | 0.0e+00 |
| F | D0 | -0.0020 ± 0.0027 | -0.0032 ± 0.0037 | +0.0012 ± 0.0010 | 0.0e+00 |
| F | D2 | -0.0000 ± 0.0008 | +0.0001 ± 0.0010 | -0.0001 ± 0.0008 | 0.0e+00 |
| F | d0t0 | -0.0016 ± 0.0037 | -0.0031 ± 0.0044 | +0.0015 ± 0.0018 | 0.0e+00 |
| F | d0t1 | -0.0025 ± 0.0036 | -0.0033 ± 0.0035 | +0.0008 ± 0.0011 | 0.0e+00 |
| F | d1t0 | -0.0059 ± 0.0053 | -0.0045 ± 0.0066 | -0.0014 ± 0.0034 | 0.0e+00 |
| F | d1t1 | +0.0006 ± 0.0087 | +0.0001 ± 0.0062 | +0.0006 ± 0.0026 | 0.0e+00 |
| F | d2t0 | +0.0000 ± 0.0010 | -0.0002 ± 0.0009 | +0.0003 ± 0.0005 | 0.0e+00 |
| F | d2t1 | -0.0000 ± 0.0011 | +0.0004 ± 0.0019 | -0.0004 ± 0.0013 | 0.0e+00 |
| DH | M | -0.0016 ± 0.0016 | +0.0017 ± 0.0055 | -0.0032 ± 0.0047 | 0.0e+00 |
| DH | T | -0.0000 ± 0.0008 | +0.0005 ± 0.0021 | -0.0005 ± 0.0018 | 0.0e+00 |
| DH | N | -0.0023 ± 0.0021 | +0.0022 ± 0.0072 | -0.0046 ± 0.0063 | 0.0e+00 |
| DH | D0 | -0.0020 ± 0.0027 | -0.0010 ± 0.0022 | -0.0011 ± 0.0007 | 0.0e+00 |
| DH | D2 | -0.0000 ± 0.0008 | +0.0005 ± 0.0021 | -0.0005 ± 0.0018 | 0.0e+00 |
| DH | d0t0 | -0.0016 ± 0.0037 | +0.0005 ± 0.0012 | -0.0021 ± 0.0028 | 0.0e+00 |
| DH | d0t1 | -0.0025 ± 0.0036 | -0.0025 ± 0.0033 | -0.0000 ± 0.0020 | 0.0e+00 |
| DH | d1t0 | -0.0059 ± 0.0053 | -0.0010 ± 0.0089 | -0.0049 ± 0.0093 | 0.0e+00 |
| DH | d1t1 | +0.0006 ± 0.0087 | +0.0119 ± 0.0234 | -0.0113 ± 0.0153 | 0.0e+00 |
| DH | d2t0 | +0.0000 ± 0.0010 | +0.0001 ± 0.0015 | -0.0000 ± 0.0010 | 0.0e+00 |
| DH | d2t1 | -0.0000 ± 0.0011 | +0.0009 ± 0.0031 | -0.0009 ± 0.0026 | 0.0e+00 |

**Test-layer decomposition availability (per §3.3):**

> Existing artifacts are insufficient for test-layer cross-epoch decomposition.

- {'endpoints_only_no_cross': 66, 'available': 22}
- Per §2.6, all 8 (arm,seed) combinations have canonical selected epoch off-grid; test-layer cross-epoch decomposition requires both e_F and e_DH to be in {9,19,29,39,49} ∪ {selected}, which is only possible when two arms happen to share the same canonical epoch (degenerate, cross-terms ≡ 0).

### Same-epoch Δ(e) on the common checkpoint grid (test layer)

**Non-target mean Δ(e) = (DH − F) at saved grid epochs:**

| seed | e=9 | e=19 | e=29 | e=39 | e=49 |
|---|---|---|---|---|---|
| 2022 | -0.0009 | +0.0078 | +0.0091 | +0.0148 | +0.0259 |
| 2023 | +0.0128 | +0.0334 | +0.0306 | +0.0158 | -0.0068 |
| 2024 | +0.0009 | +0.0070 | +0.0234 | +0.0232 | +0.0219 |
| 2025 | +0.0002 | +0.0089 | +0.0109 | +0.0289 | +0.0170 |


---

## 4. Selection-window analysis (Part C)

### Dataset: kuairand

| arm | seed | canonical epoch | V(e*) | drop_1 | drop_2 | max_drop | classification | edge-clipped |
|---|---|---|---|---|---|---|---|---|
| DH | 2022 | 40 | 0.6869 | +0.0013 | +0.0033 | +0.0064 | moderate | no |
| DH | 2023 | 30 | 0.6848 | +0.0029 | +0.0042 | +0.0058 | narrow_spike | no |
| DH | 2024 | 48 | 0.6864 | +0.0048 | +0.0034 | +0.0072 | moderate | yes |
| DH | 2025 | 27 | 0.6835 | +0.0042 | +0.0030 | +0.0044 | moderate | no |
| F | 2022 | 30 | 0.6824 | +0.0032 | +0.0033 | +0.0050 | moderate | no |
| F | 2023 | 47 | 0.6945 | +0.0080 | +0.0052 | +0.0088 | narrow_spike | no |
| F | 2024 | 41 | 0.6924 | +0.0058 | +0.0047 | +0.0061 | narrow_spike | no |
| F | 2025 | 24 | 0.6830 | +0.0054 | +0.0055 | +0.0065 | narrow_spike | no |


### Dataset: movielens

| arm | seed | canonical epoch | V(e*) | drop_1 | drop_2 | max_drop | classification | edge-clipped |
|---|---|---|---|---|---|---|---|---|
| DH | 2022 | 8 | 0.7275 | +0.0004 | +0.0020 | +0.0027 | moderate | no |
| DH | 2023 | 13 | 0.7289 | +0.0022 | +0.0022 | +0.0034 | moderate | no |
| DH | 2024 | 8 | 0.7314 | +0.0015 | +0.0030 | +0.0039 | moderate | no |
| DH | 2025 | 7 | 0.7293 | +0.0010 | +0.0018 | +0.0035 | moderate | no |
| F | 2022 | 7 | 0.7314 | +0.0018 | +0.0029 | +0.0031 | moderate | no |
| F | 2023 | 8 | 0.7293 | +0.0033 | +0.0024 | +0.0051 | moderate | no |
| F | 2024 | 8 | 0.7320 | +0.0025 | +0.0050 | +0.0064 | narrow_spike | no |
| F | 2025 | 12 | 0.7306 | +0.0012 | +0.0018 | +0.0022 | moderate | no |


---

## 5. Parameter-swap analysis (Part D, KuaiRand only)

**Partition declaration (spec §2.4/§3.5):** the swap uses `parameter_manifest.json` semantics (shared / architecture / private). Note the documented divergence: `gradient_partition.json` places `domain_expert.*` in its C (shared-side) group, whereas the manifest counts `domain_expert.*` as private. Main results use the manifest; the gradient-partition variant is at most an optional sensitivity note.

| seed | hybrid | test M | test T | test N | test D0 | test D2 | val M | val T | val N |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | C_hybrid | 0.5677 | 0.6411 | 0.5309 | 0.5347 | 0.5272  0.5867 | 0.6430 | 0.5586 |
| 2022 | DH_full | 0.6832 | 0.7136 | 0.6680 | 0.6600 | 0.6760  0.6869 | 0.7136 | 0.6735 |
| 2022 | D_hybrid | 0.5477 | 0.5872 | 0.5279 | 0.4938 | 0.5620  0.5636 | 0.5864 | 0.5522 |
| 2022 | E_hybrid | 0.5674 | 0.6411 | 0.5306 | 0.5344 | 0.5268  0.5865 | 0.6429 | 0.5584 |
| 2022 | F_full | 0.6757 | 0.7113 | 0.6579 | 0.6550 | 0.6608  0.6824 | 0.7118 | 0.6677 |
| 2022 | F_hybrid | 0.6757 | 0.7113 | 0.6579 | 0.6550 | 0.6609  0.6825 | 0.7118 | 0.6678 |
| 2023 | C_hybrid | 0.5797 | 0.5881 | 0.5755 | 0.5628 | 0.5883  0.5737 | 0.5840 | 0.5686 |
| 2023 | DH_full | 0.6743 | 0.7099 | 0.6565 | 0.6457 | 0.6673  0.6848 | 0.7079 | 0.6733 |
| 2023 | D_hybrid | 0.5196 | 0.5421 | 0.5083 | 0.5142 | 0.5024  0.5192 | 0.5435 | 0.5071 |
| 2023 | E_hybrid | 0.5796 | 0.5880 | 0.5755 | 0.5628 | 0.5882  0.5737 | 0.5839 | 0.5686 |
| 2023 | F_full | 0.6699 | 0.7146 | 0.6475 | 0.6320 | 0.6630  0.6945 | 0.7125 | 0.6855 |
| 2023 | F_hybrid | 0.6700 | 0.7146 | 0.6477 | 0.6323 | 0.6630  0.6946 | 0.7125 | 0.6856 |
| 2024 | C_hybrid | 0.5450 | 0.5625 | 0.5363 | 0.5303 | 0.5424  0.5588 | 0.5649 | 0.5558 |
| 2024 | DH_full | 0.6794 | 0.7122 | 0.6629 | 0.6658 | 0.6600  0.6864 | 0.7113 | 0.6740 |
| 2024 | D_hybrid | 0.5553 | 0.5612 | 0.5523 | 0.5427 | 0.5618  0.5511 | 0.5553 | 0.5490 |
| 2024 | E_hybrid | 0.5451 | 0.5625 | 0.5364 | 0.5304 | 0.5425  0.5589 | 0.5649 | 0.5559 |
| 2024 | F_full | 0.6744 | 0.7131 | 0.6551 | 0.6510 | 0.6591  0.6924 | 0.7123 | 0.6825 |
| 2024 | F_hybrid | 0.6745 | 0.7131 | 0.6552 | 0.6512 | 0.6591  0.6925 | 0.7123 | 0.6826 |
| 2025 | C_hybrid | 0.5842 | 0.5877 | 0.5824 | 0.5943 | 0.5705  0.5669 | 0.5840 | 0.5583 |
| 2025 | DH_full | 0.6727 | 0.7098 | 0.6541 | 0.6437 | 0.6645  0.6835 | 0.7087 | 0.6709 |
| 2025 | D_hybrid | 0.5791 | 0.6128 | 0.5623 | 0.5194 | 0.6052  0.5713 | 0.6134 | 0.5502 |
| 2025 | E_hybrid | 0.5840 | 0.5876 | 0.5822 | 0.5941 | 0.5703  0.5668 | 0.5840 | 0.5582 |
| 2025 | F_full | 0.6721 | 0.7083 | 0.6541 | 0.6312 | 0.6770  0.6830 | 0.7063 | 0.6713 |
| 2025 | F_hybrid | 0.6722 | 0.7083 | 0.6541 | 0.6312 | 0.6770  0.6830 | 0.7063 | 0.6713 |

**Retention ρ on N (test, seed-level):**

| seed | hybrid | Δ(hybrid − F_full) | denom (DH_full − F_full) | ρ_N |
|---|---|---|---|---|
| 2022 | C_hybrid | -0.1270 | +0.0101 | -12.576 |
| 2022 | D_hybrid | -0.1300 | +0.0101 | -12.878 |
| 2022 | E_hybrid | -0.1273 | +0.0101 | -12.608 |
| 2022 | F_hybrid | +0.0000 | +0.0101 | +0.002 |
| 2023 | C_hybrid | -0.0720 | +0.0090 | -7.969 |
| 2023 | D_hybrid | -0.1392 | +0.0090 | -15.413 |
| 2023 | E_hybrid | -0.0720 | +0.0090 | -7.975 |
| 2023 | F_hybrid | +0.0002 | +0.0090 | +0.017 |
| 2024 | C_hybrid | -0.1187 | +0.0079 | -15.047 |
| 2024 | D_hybrid | -0.1028 | +0.0079 | -13.025 |
| 2024 | E_hybrid | -0.1186 | +0.0079 | -15.033 |
| 2024 | F_hybrid | +0.0001 | +0.0079 | +0.015 |
| 2025 | C_hybrid | -0.0717 | +0.0000 | -6505.820 |
| 2025 | D_hybrid | -0.0918 | +0.0000 | -8327.234 |
| 2025 | E_hybrid | -0.0718 | +0.0000 | -6519.696 |
| 2025 | F_hybrid | +0.0001 | +0.0000 | +5.585 |

**Interpretation boundary (spec §8):** parameter swapping is a *localisation* diagnostic, not a causal decomposition. Independently trained parameter partitions may be co-adapted; hybrids that fail badly are evidence of cross-partition co-adaptation rather than absence of an effect.

---

## 6. MovieLens negative control

Per spec §9, only Parts A and B were replicated on MovieLens. The canonical selected epochs are very early (7–13). Compare the tables above (`section 2`, `section 3`).

---

## 7. Counterexamples

1. **Seed 2023 target regression under R0/R2 (test):** ΔT = −0.0047, below the project practical threshold −0.003. The aggregate "target preserved" claim rests on the 4-seed mean (−0.0004); one seed in four violates it.
2. **Seed 2025 is the weak seed:** R0 test ΔN ≈ +0.00001 (positive but numerically zero); under R1/R4 (grid) ΔN = −0.0039 and under R3 (grid) ΔN = −0.0015 — i.e. the alternative-rule robustness is carried by seeds 2022/2023/2024, and seed 2025 actively disagrees.
3. **Validation-layer reversal (the largest counterexample to a naive reading):** at the R0-selected checkpoints, val ΔN is *negative* in 3/4 seeds (mean −0.0039) and val ΔM negative in 2/4. No rule R0–R4 shows a positive val-layer ΔN with >2/4 seeds. Any statement "DH is better" is false on validation; it holds only on test.
4. **Part D seed 2025 retention is undefined in practice:** denom_N = +1.1e-05, producing |ρ| > 6000; seed 2025 contributes no usable localization signal, and all ρ values are ≪ −1 (hybrids lose 8–15× the entire DH−F gap) — co-adaptation artefact, reported as such per spec §8 rather than as "negative gain".
5. **Part B degenerate case (MovieLens seed 2024):** e_F = e_DH = 8, so the test-layer decomposition rows are trivially satisfied with cross-terms ≡ 0; they carry no evidential weight.
6. **Part C irregularities:** DH seed 2023 is itself a narrow_spike (drop_1 = 0.0029, drop_2 = 0.0042); DH seed 2024 is edge-clipped (canonical epoch 48 of 49, window truncated). The "DH peaks are not narrow" claim is a 3/4-seed statement, not 4/4.
7. **A-full test NA gaps:** R1/R3 have zero and R2/R4 have only 1–2 seeds with exact test values; multi-rule robustness on test necessarily leans on the A-grid sensitivity variant (selection space restricted to 6 epochs).

---

## 8. Competing explanations (H1–H6 evidence-strength)

- **H1 — robust better solution: MEDIUM.** For: test ΔN = +0.0068 (4/4) under R0 and +0.0041/+0.0060/+0.0041 (each 3/4) under R1/R3/R4 grid; ΔT > −0.003 on mean everywhere; ΔM positive 4/4 under R0 (§2). Against: the "better solution" is invisible on validation — val ΔN ≤ 0 under every rule (§2 val rows) — and seed 2025 disagrees under three alternative rules (§7.2). DH reaches a *more test-robust* solution, not a *higher-validation* solution; H1 survives only in that narrowed form.

- **H2 — selection-objective specificity: WEAK.** The benefit is not exclusive to R0 or the non-target-oriented R2: it persists under target-only (R1), worst-domain (R3) and sample-weighted (R4) selection on the grid variant, with the target penalty staying small (ΔT_grid ∈ [−0.0024, −0.0004]). Counter-evidence is only that exact (non-grid) test values are NA for R1/R3 (§7.7).

- **H3 — checkpoint timing: MEDIUM.** For: on validation the same-epoch policy term flips sign with the anchor (N: PolicyAtF = −0.0111 vs PolicyAtDH = +0.0082) while timing terms are large (TimingAlongF = −0.0120, 4/4 negative) — epoch-alignment, not a stable policy gap, drives the val-layer ΔSel (§3 kuairand). F's val peaks are also narrower (3/4 narrow_spike vs DH 1/4, §4), so part of DH's selected advantage is "DH sits on broader, later plateaus; F's spikes decay". Against: the test-layer same-epoch grid Δ(e) is positive at 16/20 points (§3 grid table), so timing cannot be the *whole* test-side story; the definitive test-layer attribution is artifact-blocked (§10).

- **H4 — shared-state localization: WEAK (no support).** C_hybrid (DH shared/architecture + F private) collapses to macro 0.545–0.584, ρ_N ∈ [−15.0, −8.0]; the gain is not preserved by the DH shared-side state (§5). E_hybrid ≈ C_hybrid, so adding F architecture changes nothing.

- **H5 — private/co-adaptation: STRONG as a co-adaptation finding; indeterminate as a gain explanation.** D_hybrid (F shared + DH private) also collapses (macro 0.520–0.579) and F_hybrid ≡ F_full (ρ ≈ 0.00–0.03). Both directions failing catastrophically is exactly the spec §8 signature of dominant cross-partition co-adaptation: F-selected and DH-selected solutions are internally co-adapted, and parameter swapping cannot localize the gain. This is a reproducible mechanistic fact about the solutions, but it does not by itself explain why DH's co-adapted state generalizes better on test non-target cells.

- **H6 — random/narrow peak: WEAK.** 4/4 seeds positive under R0 test with 3/4 ≥ +0.005; the effect replicates across four selection rules; DH selected windows are 3/4 moderate plateaus (drop_1 mean ≈ 0.0033 vs F ≈ 0.0056, §4) — if anything it is F whose selections are spike-prone. An isolated-spike origin is inconsistent with the multi-rule, multi-seed persistence.

---

## 9. GO / STOP decision (spec §11)

Conditions (1–6) of §11 are reproduced verbatim below; each must be checked by the main agent against the tables in §§2–5:

1. canonical KuaiRand selected ΔNonTarget remains practical (≥ +0.003) and ≥3/4 seeds positive;

2. target change remains above −0.003;

3. macro gain remains practically positive or the trade-off is otherwise clearly favorable;

4. at least two reasonable validation-selection rules show a qualitatively similar result, excluding reliance on non-target-only selection;

5. the selected gain is not explainable purely as a single-epoch spike;

6. decomposition provides at least one reproducible clue about same-epoch policy effect, timing, or parameter partition localization.


[MAIN-AGENT — gate applied:]

1. **PASS.** Canonical R0 test ΔN = +0.0068 ≥ +0.003, 4/4 seeds positive (§2 kuairand R0 test row).
2. **PASS (mean; one-seed caveat).** ΔT = −0.0004 > −0.003 on the 4-seed mean; seed 2023 individually at −0.0047 (§7.1).
3. **PASS.** ΔM = +0.0044, 4/4 positive, 3/4 ≥ +0.003.
4. **PASS (qualified).** R1, R3, R4 — three independent rules, none non-target-only — all show ΔN ≥ +0.004 with 3/4 seeds positive and ΔT > −0.003. Qualification: their exact re-selected epochs lack saved test checkpoints, so this rests on the A-grid sensitivity variant; and on the *validation* layer no rule shows any DH advantage (§7.3, §7.7).
5. **PASS.** DH selected optima are 3/4 moderate plateaus, not spikes; F's are the narrower ones (§4 kuairand).
6. **PASS.** Two reproducible clues: (a) val-layer decomposition shows anchor-dependent policy sign with dominant timing terms (§3); (b) Part D shows complete cross-partition co-adaptation (both hybrids collapse, ρ_N ≪ −1) while architecture parameters are provably inert (F_hybrid ≡ F_full) (§5).

**Verdict: GO — conditions 1–4 broadly satisfied → do NOT stop the KuaiRand DH selected-solution line. Mechanism localization is ambiguous (val–test reversal unexplained; parameter swapping blocked by co-adaptation), so per spec §11 exactly ONE minimal mechanism experiment is licensed, with no method design.**

**Licensed next experiment (minimal, evaluation-oriented):** a *checkpoint-dense evaluation replication* of the existing KuaiRand S6 F/DH configuration — identical model, objective, data ratio, seeds 2022–2025, hyperparameters; the only change is saving the full state_dict every epoch (or every 2 epochs) so that test-layer cross-epoch decomposition (Part B on test) and exact (non-grid) test evaluation of R1/R3/R4-selected checkpoints become possible. This is a measurement replication, not a new method; it directly closes the two artifact gaps that blocked definitive attribution in this stage. No algorithm modification, no new objective, no hyperparameter search is authorized by this gate.

---

## 10. What remains unknown

1. **Test-layer policy/timing attribution at the actual selected epochs.** All 16 canonical selected epochs lie off the saved checkpoint grid {9,19,29,39,49}; the test-layer cross-epoch decomposition is impossible from existing artifacts (§3, "Existing artifacts are insufficient"). Whether the +0.0068 test ΔN is mostly same-epoch policy or mostly epoch-timing remains unanswered on the layer where the phenomenon actually lives.
2. **The origin of the val→test generalization difference.** The stage establishes *that* DH's selected checkpoints lose less from validation to test on non-target cells (val ΔN = −0.0039 vs test ΔN = +0.0068 at the same checkpoints), but not *why* — candidate mechanisms (shared-representation smoothness, private-head calibration, objective-induced flatness) are untested.
3. **Parameter localization.** Cross-partition co-adaptation is total (§5); no swap-based localization of the gain is possible with this partition scheme. Whether a finer or functional (e.g. representation-level) probe could localize the effect is open.
4. **Exact multi-rule test robustness.** R1/R3/R4 robustness is demonstrated only on the 6-point A-grid variant; exact test values at the truly re-selected epochs are NA (§7.7).
5. **Seed 2025.** Its near-zero canonical ΔN and negative alternative-rule ΔN are unexplained; 3/4-seed robustness claims should be re-examined if more seeds ever become available.
6. **Horizon dependence.** All conclusions are within the 50-epoch S6 training budget; nothing is known about longer horizons.
7. **MovieLens mechanism.** The negative control confirms selection never captures ML's late DH advantage, but why ML selects at epochs 7–13 (very early peaks) while KuaiRand selects at 24–48 is itself unexplained.

---

## 11. Auto-generated outputs (file map)

- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partA_*.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partB_*.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partC_window_kuairand.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partD_swap_kuairand.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/kuairand/s6r2_partD_swap_retention_kuairand.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/movielens/s6r2_partA_*.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/movielens/s6r2_partB_*.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/movielens/s6r2_partC_window_movielens.csv`
- `D:/CODE/phase0-3/experiments/stage_S6_R2_raw_results/s6r2_integrity_report.json`
