# Stage S1 Main Analysis — Residual Negative Transfer Reproduction

## Decision

**STOP — Residual Negative Transfer as a primary research problem.**

Under the confirmed sample-proportional / target-aligned objective, joint MDMT training is overwhelmingly *beneficial* across all three backbones and both datasets (mean per-cell ΔAUC `+0.0384`, median `+0.0237`, 288 seed-cells). The residual degradation that remains is small (`0.000x–0.001` at seed-mean level), seed-sign-unstable, backbone-dependent, dataset-non-replicating, and numerically indistinguishable from the single-domain reference's own seed variance. Every STOP criterion in the Stage 1 charter is met; every GO criterion fails. **Stage 2 (Residual NT Causal Diagnosis) is NO-GO.** Per the charter, no Safe MTL method design may be entered to preserve this direction.

## Evidence base and protocol validity

- 192/192 formal runs valid (`status="OK"`, `validation_reasons=""`): 2 datasets × 4 settings (movielens d0_large/d1_large/d2_large, kuairand natural) × 3 backbones (m3oe/mmoe/ple) × 4 run types (single_domain d0/d1/d2 + joint) × 4 seeds (2022–2025).
- 288 paired seed-cells (72 aggregate groups × 4 seeds); each aggregate group = (backbone, dataset, setting, physical_domain, task) at the validation-selected checkpoint.
- Objective: sample-proportional only; equal weighting excluded from the main analysis per charter.
- Execution incidents were infrastructure-only (two console-logoff kills, one orchestrated pause) and were handled by the reviewed fail-fast recovery: affected attempts archived under `failed_attempts/` and retrained from scratch. No orphan or mixed-code artifacts entered the formal audit.
- All numbers below are quoted from `experiments/stage_S1_paired_cells.csv`, `stage_S1_paired_aggregates.csv`, `stage_S1_stable_cells.csv`, `stage_S1_reference_baseline_variation.csv`, and `stage_S1_summary_manifest.json` (2026-09-21 15:52).

## 1. Sample correction 后究竟还有多少 NT？（How much NT remains）

**Effectively none at the aggregate level; only sparse seed-level candidates.**

Seed-cell level (n=288):

- Raw negative ΔAUC: 43 cells (14.9%);
- practical NT (|Δ| ≥ 0.003): 12 cells (4.2%);
- strong NT (|Δ| ≥ 0.005): 5 cells (1.7%);
- practical LogLoss NT: 44 cells (15.3%), but with frequent AUC↔LogLoss sign disagreement (see §5).

Seed-mean level (n=72 aggregate groups) — the level the GO charter speaks to:

- **Zero** groups with mean ΔAUC ≤ −0.003. The most negative group mean is **−0.00292** (m3oe / movielens d0_large / domain 0 / task 1);
- 9/72 groups have negative mean ΔAUC at all; the other 8 besides the above are between −0.00001 and −0.00135;
- **`stable_practical_cell = 0` for all 72 groups** — no cell is stably, practically damaged even once the loosest stability definition is applied;
- the single worst seed-cell in the whole matrix is −0.00737 (m3oe / d2_large / domain 2 / task 1 / seed 2025), but its own group mean is −0.00027 with only 2/4 seeds negative — a seed fluctuation, not a stable effect.

For scale: the same matrix shows mean ΔAUC `+0.0423` (m3oe), `+0.0360` (mmoe), `+0.0369` (ple), with per-cell gains up to `+0.1429`. Residual degradation is a small tail against a large positive-transfer main effect.

## 2. 哪些 domain/task 稳定受损？（Which cells are stably damaged）

**None.** No group satisfies ≥3/4 seeds negative *and* |Δ| above the practical threshold:

- Only 1/72 groups has 4/4 seeds with negative ΔAUC (m3oe / d0_large / domain 0 / task 1), and its mean is −0.00292, below the 0.003 practical line; only 2/4 of its seeds cross practical NT and 1/4 crosses strong NT.
- The remaining negative-mean groups show 2–3/4 negative seeds with means between −0.00001 and −0.00135.
- `concordant_practical_nt_rate` never exceeds 0.5 anywhere; `concordant_strong_nt_rate` never exceeds 0.25.

The recurring *candidates* (conservative wording per charter — residual NT candidates, not established sharing-induced NT):

| Group | mean ΔAUC | neg seeds | reference seed SD (STL) |
|---|---|---|---|
| m3oe / kuairand / d1 / t0 | −0.00073 | 3/4 | 0.00072 |
| m3oe / movielens d0_large / d0 / t1 | −0.00292 | 4/4 | 0.00283 |
| m3oe / movielens d0_large / d0 / t0 | −0.00135 | 3/4 | 0.00266 |
| m3oe / movielens d2_large / d2 / t0,t1 | −0.00073 / −0.00027 | 2/4 | 0.00408 / 0.00846 |

In every candidate the |mean Δ| is at or below the single-domain reference's own seed-level standard deviation (ratio ≈ 0.1–1.0).

## 3. 是否存在 backbone-independent pattern？

**No.**

- The only group negative in all three backbones is kuairand / domain 1 / task 0: m3oe −0.00073, mmoe −0.00020, ple −0.00001. Direction is shared but magnitude is 0.0000x–0.0007 — at or below each backbone's reference seed SD (0.00072 / 0.00107 / 0.00085) and 4–300× below the practical threshold.
- The strongest single candidate (movielens d0_large / domain 0, both tasks, m3oe) **flips sign under backbone change**: mmoe +0.00158/+0.00106, ple +0.00051/+0.00157 on the identical cell.
- movielens d2_large / domain 2 / task 1 is negative for m3oe (−0.00027) and mmoe (−0.00077) but positive for ple (+0.00164).

A pattern that disappears or reverses when the backbone changes is exactly a charter STOP signal.

## 4. 是否存在 dataset-independent pattern？

**No.**

- No (domain-role, task) combination reproduces negative Δ across MovieLens and KuaiRand with any magnitude. The m3oe weakest-domain candidates (d0 in d0_large; d2 in d2_large; d1 in kuairand) do not align: kuairand d1/t1 is positive (+0.00048) where movielens d0/t1 is negative (−0.00292), and mmoe/ple show no corresponding movielens effect at all.
- Positive transfer, by contrast, replicates everywhere: mean ΔAUC `+0.0420` (movielens) and `+0.0275` (kuairand), positive in all 72/72 groups' dominant direction.

## 5. NT 是否足够大、稳定、可研究？

**No — it fails on all three axes.**

- **Magnitude:** max seed-mean |Δ| 0.00292 < 0.003 practical threshold; most negatives are 0.000x. STOP charter: "多数只有 0.000x–0.001" — met verbatim.
- **Stability:** 0/72 stable practical cells; seed signs flip in all but one group (mechanical contradiction inventory records 3,075 sign-flip/disagreement entries across seeds, metrics, and training stages). "seed 正负频繁翻转" — met.
- **Identifiability vs noise:** every negative candidate sits within ±1× of its single-domain reference seed SD (§2 table); the worst single seed-cell (−0.00737) is inside its reference's seed range (SD 0.00846, range 0.0201). "与 STL 自身方差处于同一数量级" — met.
- LogLoss does not rescue the signal: the largest mean LogLoss gaps in negative-AUC candidates are +0.0024 to +0.0043 against reference LogLoss seed SDs of 0.0056–0.0102, and mmoe/kuairand/d1 shows slightly negative ΔAUC with *improved* LogLoss — metric disagreement, not convergent degradation.

## 6. Stage 2 是否 GO？

**NO-GO. STOP Residual Negative Transfer as a primary research problem.**

GO required (charter): practical effect (|Δ| ≥ 0.003, strong ≥ 0.005) plus preferably ≥3/4-seed concordance, ≥2 backbones, ≥2 datasets/settings with an interpretable consistent pattern. The matrix delivers none of these at seed-mean level. All five STOP bullets are satisfied. Per the charter, this negative result is a valid and terminal scientific outcome: objective-induced NT was the operative problem; once controlled by the sample-proportional objective, no stable residual NT remains to diagnose, and no Safe MTL method line may be opened on this basis.

## Stable phenomena, contradictions, and competing explanations

Stable repeated phenomena:

1. **STRONG:** joint MDMT training under the sample-proportional objective yields large positive transfer in all 3 backbones × both datasets (group means up to +0.124; overall mean +0.0384; 63/72 groups positive).
2. **STRONG:** no stable practical residual NT cell exists (0/72; max seed-mean |Δ| 0.00292).
3. **STRONG:** residual candidates, where they exist at all, are numerically bounded by single-domain reference seed variance.
4. **MEDIUM:** residual candidates concentrate on the weaker/auxiliary physical domain of a setting (kuairand d1, movielens d0 in d0_large, d2 in d2_large), but with sub-noise magnitude and backbone-dependent sign.

Contradictory/qualifying evidence:

- 12/288 practical and 5/288 strong seed-level AUC NT cells, and 44/288 practical LogLoss NT cells, exist as isolated seed events; they do not aggregate into any stable group-level pattern.
- m3oe / d0_large / domain 0 / task 1 is negative in 4/4 seeds — the matrix's most consistent candidate — yet below the practical magnitude line and non-replicating across backbones; it is recorded here as the top residual NT candidate should future work (e.g., a third dataset) revisit the question with higher seed counts.

Competing explanations ranked by current support:

1. **Reference seed noise / estimation variance:** fully sufficient for every observed negative candidate (|Δ| ≤ reference seed SD everywhere).
2. **Backbone-specific optimization idiosyncrasy:** consistent with the sign flips between m3oe and mmoe/ple on identical cells.
3. **True stable residual sharing-induced NT:** unsupported — no group clears magnitude and stability simultaneously.

## Downstream disposition

- Stage 2 (Residual NT Causal Diagnosis): **not entered** — gated on Stage 1 GO.
- Stage 3+ / Safe MTL method design: **not entered** — charter forbids method work absent a reproduced problem.
- The user's standing requirement (issued 2026-09-21) that future multi-seed validation dispatch two concurrent training processes instead of fully serial execution is recorded as a mandatory design constraint for any future stage that *is* entered; it was deliberately not applied to the S1 tail (single remaining seed; mid-flight code change barred by the reviewed protocol).
- S1 artifacts (192 run dirs, 13 report tables, queue/lock/reclaim manifests, failed_attempts archives) are frozen for audit; no further action is required or permitted on them without a new reviewed stage.
