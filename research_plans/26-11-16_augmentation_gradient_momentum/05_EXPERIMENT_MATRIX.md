# Experiment Matrix

Stage 0: read-only identity audit and Astra capability preflight; no training.
Stage 1: CPU correctness + real full-batch GPU smoke on ML/KR; no performance claims.
Stage 2, all seeds 2022–2026, 50 epochs:

| Arm | gamma | Datasets | New runs |
|---|---:|---|---:|
| TV-GRM-005 | .05 | ML/KR | 10 |
| TV-GRM-010 | .10 | ML/KR | 10 |
| TV-GRM-020 | .20 | ML/KR | 10 |
| TV-GRM-SHUFFLE | .10 | ML/KR | 10 |
| CLEAN | 0 | ML/KR | 0 if exact reuse |
| NDP-A | historical direct-mixture context | ML/KR | 0; do not rerun |

Total 40 new full trajectories (2 datasets × 4 arms × 5 seeds). Historical references are not independent confirmation.

Gamma search is already included in five-seed matrix. No broad alpha/lambda search. Test only after candidate/config/source/data/selection freeze. If no candidate has positive validation Macro AUC mean and is not dominated by SHUFFLE/NDP-A, Test stays sealed. Ablations only if a positive AUC candidate exists and a specific mechanism uncertainty could change the claim.
