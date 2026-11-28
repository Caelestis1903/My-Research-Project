# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

`three_directions_20261128/C0` completed as a single foreground CPU audit of 20 retained A1 fixed-EMA trajectories. The run performed zero new training and recorded zero Test accesses.

## 2. Git and worktree state

The study worktree was detached at `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc`, with the existing modified vendored `objectives.py` and untracked study sources, reports, and artifacts. The executed C0 SHA-256 matched the reviewed source. [Implementation review](C0_IMPLEMENTATION_REVIEW.md)

- C0 source SHA-256 `a9b36fab67dd27687892442a648e3d4c685f8cea4c0d833752655aa421abaeda`.
- A1 roster SHA-256 `cb8b0cd85f3b128233b5c60be9941efda68cee21595080b211e1d0498992eecb`.
- C0 manifest SHA-256 `09dee9475202c9de93ebdea64203706f8d9eca487b8c0590f7206215dbc15b92`.
- The A1 roster, stage status and all 40 completion receipts matched their hashes captured during the run.

## 3. Configuration

Foreground command from `/home/xzf/Code/phase0-3_three_directions_20261128/m3oe/examples`, using project Python 3.13.5 and PyTorch 2.9.0+cu126.

```bash
PYTHONPATH=/home/xzf/Code/phase0-3_three_directions_20261128 PYTHONDONTWRITEBYTECODE=1 \
  /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  -m experiments.three_directions_20261128.c0_audit \
  --artifacts /home/xzf/Code/phase0-3_three_directions_20261128/artifacts/three_directions_20261128
```

CPU FP32 clean inference used up to 40,960 Train rows per batch. Each dataset and seed used one shared candidate set for B0 and P0, with up to 4,096 pairs per active condition and 2,048 draws per replay event. The run read E05, E10 and the original Validation-selected fixed EMA. The manifest records immutable Train cache fingerprints, full configuration and parameter scopes. [Manifest](../../artifacts/three_directions_20261128/C0/MANIFEST.json)

## 4. Seeds

2022, 2023, 2024, 2025 and 2026, as specified for the two datasets and two A1 arms. [Design specification](C0_EXECUTION_DESIGN.md)

## 5. Run status

- Exit code 0 and `COMPLETE`. All 20 expected inputs received `COMPLETE` dispositions. The recorded boundaries span 2026-11-28T12:32:19+00:00 through 2026-11-28T12:39:06+00:00. [Final status](../../artifacts/three_directions_20261128/C0/STATUS.json)
- All ten original and current study stop paths were clear at launch. The audit recorded 64 clear boundary checks. Host memory usage was 14.434% before launch.
- Verified both Train caches, 10 candidate and 20 margin archive SHA-256 receipts, and the linkage of all 60 fresh RAM readbacks to their frozen A1 completion, RAM journal, Validation and margin records.

## 6. Raw metrics

AUC and LogLoss are the original six-cell Validation macro metrics. The last two columns show weighted Train strict reversal rates conditioned on historical correctness.

| Dataset | Arm | Seed | Selected epoch | E5 AUC | E5 LogLoss | E10 AUC | E10 LogLoss | Selected AUC | Selected LogLoss | E5 reversal | E10 reversal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | B0 | 2022 | 26 | 0.746490220 | 0.535399509 | 0.750596666 | 0.522971390 | 0.760026467 | 0.526644258 | 0.035246892 | 0.031335311 |
| MovieLens-1M | B0 | 2023 | 18 | 0.745297916 | 0.553094743 | 0.751883182 | 0.522153446 | 0.759713740 | 0.523377127 | 0.034421750 | 0.027965511 |
| MovieLens-1M | B0 | 2024 | 20 | 0.746350887 | 0.545078353 | 0.750890637 | 0.523482418 | 0.759673484 | 0.525143249 | 0.032185626 | 0.026451200 |
| MovieLens-1M | B0 | 2025 | 28 | 0.746366854 | 0.538700335 | 0.749919420 | 0.523742065 | 0.759084400 | 0.527042800 | 0.034880236 | 0.032064361 |
| MovieLens-1M | B0 | 2026 | 22 | 0.746277517 | 0.556313432 | 0.752193899 | 0.522042619 | 0.759060300 | 0.527621432 | 0.037201715 | 0.029546271 |
| MovieLens-1M | P0 | 2022 | 19 | 0.746740194 | 0.529062233 | 0.753203938 | 0.521402801 | 0.759211193 | 0.526941283 | 0.033163938 | 0.025324378 |
| MovieLens-1M | P0 | 2023 | 20 | 0.745957686 | 0.549462561 | 0.752694943 | 0.521412948 | 0.759603141 | 0.523276655 | 0.035801591 | 0.028493045 |
| MovieLens-1M | P0 | 2024 | 21 | 0.746755441 | 0.529938209 | 0.751511649 | 0.522797468 | 0.759111129 | 0.528780783 | 0.034749731 | 0.027265209 |
| MovieLens-1M | P0 | 2025 | 22 | 0.746659013 | 0.532077463 | 0.751172118 | 0.523466457 | 0.759240082 | 0.526936283 | 0.034816995 | 0.029094859 |
| MovieLens-1M | P0 | 2026 | 22 | 0.746549553 | 0.540641760 | 0.753361234 | 0.520586195 | 0.759449466 | 0.528967866 | 0.037602111 | 0.029216395 |
| KuaiRand-Pure | B0 | 2022 | 33 | 0.597917017 | 0.458112552 | 0.636229081 | 0.458477007 | 0.692638160 | 0.430600892 | 0.154358414 | 0.113975651 |
| KuaiRand-Pure | B0 | 2023 | 50 | 0.598923037 | 0.458953676 | 0.640485223 | 0.448828163 | 0.677978728 | 0.432979571 | 0.141953840 | 0.111655138 |
| KuaiRand-Pure | B0 | 2024 | 50 | 0.592692587 | 0.467650511 | 0.639785285 | 0.440828747 | 0.695993511 | 0.426759923 | 0.145647656 | 0.112229518 |
| KuaiRand-Pure | B0 | 2025 | 50 | 0.601249251 | 0.454902207 | 0.646204498 | 0.456417892 | 0.686161380 | 0.429820889 | 0.144957202 | 0.107970091 |
| KuaiRand-Pure | B0 | 2026 | 47 | 0.596568521 | 0.451961237 | 0.641417684 | 0.450436009 | 0.689500363 | 0.428141739 | 0.154917659 | 0.116479870 |
| KuaiRand-Pure | P0 | 2022 | 22 | 0.603657820 | 0.456334484 | 0.651976592 | 0.441742684 | 0.689463428 | 0.434047252 | 0.122727519 | 0.085638413 |
| KuaiRand-Pure | P0 | 2023 | 24 | 0.604078903 | 0.458866581 | 0.656284386 | 0.448251048 | 0.688976233 | 0.435543173 | 0.129823302 | 0.088311301 |
| KuaiRand-Pure | P0 | 2024 | 21 | 0.605141420 | 0.456070434 | 0.658232036 | 0.451660012 | 0.686549470 | 0.435374515 | 0.130434122 | 0.077394636 |
| KuaiRand-Pure | P0 | 2025 | 24 | 0.618906328 | 0.453117671 | 0.657120909 | 0.456238787 | 0.683447637 | 0.434493869 | 0.121668435 | 0.079640988 |
| KuaiRand-Pure | P0 | 2026 | 29 | 0.605271340 | 0.450149361 | 0.658030183 | 0.455037826 | 0.689273509 | 0.431114286 | 0.128941142 | 0.085716536 |

- Full original per-domain, per-task, six-cell and macro AUC/LogLoss at each checkpoint, plus the aligned Train support and exposure rows. [Validation](../../artifacts/three_directions_20261128/C0/VALIDATION.json)
- Full candidate counts, weighted rates, margin statistics, confidence bins, replay exposure, eligible seed counts, horizons and paired summaries. [Pair metrics](../../artifacts/three_directions_20261128/C0/PAIR_METRICS.json) · [Overlap](../../artifacts/three_directions_20261128/C0/OVERLAP.json) · [Seed summaries](../../artifacts/three_directions_20261128/C0/SUMMARY.json)
- Audit elapsed time 407.089 s. Peak process RSS 2,178,883,584 bytes. Fresh RAM reads 288,740,680 bytes. CPU forwarded 5,958,114 Train rows across 180 batches. Scored 1,310,720 candidates across trajectory inputs. Replayed 13,610 logical steps and 27,873,280 draws. Numeric outputs occupied 57,346,879 bytes. [Cost receipt](../../artifacts/three_directions_20261128/C0/COST.json)

## 7. Log locations

- Complete combined stdout and stderr. [Foreground log](../../artifacts/three_directions_20261128/C0/FOREGROUND.log)
- All input dispositions, fresh snapshot receipts, source and data identities, and candidate and margin archive links. [Status](../../artifacts/three_directions_20261128/C0/STATUS.json) · [Manifest](../../artifacts/three_directions_20261128/C0/MANIFEST.json)
- Per-job JSON files, shared candidate archives and aligned snapshot margin archives are under `C0/jobs/`, `C0/candidates/` and `C0/margins/`.

## 8. Runtime errors

None. PyTorch emitted one TF32 API deprecation warning. No retry was required. [Foreground log](../../artifacts/three_directions_20261128/C0/FOREGROUND.log)

## 9. Missing runs

None. All 20 registered trajectories were inspected, with all three named snapshots freshly read per trajectory.
