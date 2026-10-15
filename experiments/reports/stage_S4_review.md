# REVIEW REPORT — Stage S4 Amendment A1

1. Verdict: **PASS**. Blocking issues: **0**. This independently reviews the binding A1 design, coder report, actual S4 modules/tests, and retained abort evidence. No A1 queue was frozen or formal training launched during review.

2. Requirement-by-Requirement Verification

| Requirement | Independent finding |
| --- | --- |
| S4 scope and matrix | A1 has exactly F/T/N/U for seeds 2022–2025 under Equal/KuaiRand/natural/M3OE. Normalized A1 and old task records are equal after removing only `run_dir`; all eight `(F,T)` then `(N,U)` waves and their order match. Each seed has four arms. |
| Old abort identity | Queue SHA `693b46697ba40bf6e9cd03c6e0d4688a3ee7aa80a5d6ce2a442bed5c846fb221`, state SHA `8ebff1bc35dd7fa5a182cfb3d56b88f7e336bedb66cd93fba588c5bc188c622e`, and abort-report SHA `52b6d5924fb60c6453e669d6318a11cc064cf8858c770b77440ff30e33a5d30e` match A1. State has 0/8 completed waves and a blocked first attempt; all 16 old canonical directories and old collector files are absent. Four attempt files have deterministic inventory digest `4bae95e4e35bb27fceaf3f70fe2ab21f57d24110f9471a1951a701f3ff1a0e18`. |
| New queue and source | A1 queue, raw root, and output root are absent. Freeze requires the new raw root empty, exact old matrix/waves, and hashed old report/attempt evidence. It freezes 21 nonempty producer hashes including M3OE forward `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`. Current inventory digest: `b82283018cc5d165b8f08ca4582a1537f7d6b10750de6c5ded68857aff13f7d1`. Existing A1 queue validation rejects source drift. The old queue cannot bind the repaired producer. |
| Numerical correction | Vector check is `||observed-predicted||₂ <= max(1e-6, 1e-4 ||predicted||₂)` with exact `None`/no-update semantics. N logs its predicted same-state Adam delta norm; validation uses it as N's formula reference. CUDA M3OE regression covers C and A_arch, first/warmed states, zero/None gradients, coupled decay, and installed foreach dispatch. A residual above `1e-6` within the relative bound passes; one beyond it fails. |
| Collector repair | Collector defaults to and enforces `experiments/stage_S4_A1_outputs`, writing its report at `experiments/stage_S4_A1_outputs/reports/stage_S4_A1_run_report.md`. It accepts only A1 queue/raw root. Fixture collection twice with intervening queue validation passed; legacy-root use and old-report tampering were rejected. |
| Original gates | One-forward loss decomposition, S3 parameter partition, N/U norm matching, U FULL Adam state and private FULL deltas, fixed probes, finite/source/metric/checkpoint checks, four-arm pairing, two-process overlap and whole-wave retry remain enforced. All 17 local control/orchestration tests passed independently in `EDDAenv` with CUDA available. |

3. Scientific & Algorithmic Integrity Audit

- **Mathematical and tensor correctness:** F applies FULL gradients; T applies target-only gradients to C and A_arch; N scales FULL shared gradients before Adam; U commits one FULL-gradient Adam step and scales only its shared move to the same-state target-only delta norm. Private main gradients and deltas remain FULL. Norm audits use float64; training tensors keep their dtype. N/U matching retains its separate design tolerance.
- **Gradient flow and loss participation:** Shared/private gradients come from one stochastic forward via `autograd.grad`. Target gradients determine T, N, U and same-state probes. Probes do not add a forward or optimizer step. Per-step finite and Adam clone checks block corrupt updates.
- **Controlled variables and baseline preservation:** A1 changes only queue/raw/output provenance and reviewed numerical/collector repairs. The old queue, state, attempts and report retain their hashes. Pairing validates source, data/splits/views, coefficients, initialization, partition, optimizer/exposure/batch traces, checkpoints and prediction-level metrics. No S3 artifact becomes a formal S4 arm.
- **Fail-fast:** Undefined direction, nonfinite values, missing evidence, source drift, broken probes, unmatched metrics, incomplete blocks and unproven process overlap fail. Only explicit transient child `OSError` errnos allow one fresh whole-wave retry. No silent config fallback or tensor reshaping was found. Broad catches in validation/promotion reject or roll back and re-raise.

4. Over-Engineering & Code Bloat Audit

The A1 predecessor receipt, output restriction and matrix comparison are necessary to prevent aborted evidence entering formal results. Repairs are local to S4 modules/tests. No unnecessary S1–S3 refactor, speculative abstraction or new algorithm was found.

5. Bugs / Deviations Found

- **MAJOR, corrected before this review:** The previous collector wrote into the legacy output location, invalidating repeated predecessor checks and overwriting the old abort report. The fixed A1 output root and repeatability regression resolve it. No open critical or major issue was found.

6. Corrections Made

Reviewer made no code changes. Updated this report after independently inspecting source/evidence. Ran 17/17 local tests and read-only verified old hashes, 0/16 old canonical artifacts, 16-task/eight-wave equality, absent A1 paths and the current source inventory.

7. Remaining Risks

Two-process A1 CUDA training has not completed; GPU capacity and scientific outcomes remain unknown. The runner must stop on any technical gate failure. A producer source change after freeze requires fresh design/review. The aborted first wave supplies no valid AUC/loss result.

8. Whether Experiments Are Safe to Run

**Yes, for isolated A1 only.** Run from `D:\CODE\phase0-3`, in order:

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_orchestrator.py --a1 --reviewer-pass --freeze-only
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_orchestrator.py --a1 --reviewer-pass
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_report.py
```

The run command schedules exactly two concurrent Python training processes per wave. At freeze, check the 21-entry inventory digest above and the five S4 hashes: protocol `0b95303373d37afa02e80abd82a83bd9ead5c7b980c0a4e4e555ee7b4031a858`, controls `e695bbfafafb9b238d5cb691fde5113bc7e87f2b7699e766fb84d71167090645`, run `8017e93510f4e25cf0d2ce7ea7c1208fe4a73ea04b4cf94038621580407169c3`, orchestrator `0f44b29dc95e9b4aef00b7fd0ea04301dcef05bf95c1dcafc7ea2d78e7e4bf2c`, collector `c891d456c3548719b569e1da2b7601d5427540c9c0b523043a2fe6b15e5af593`. Any mismatch stops the run. Collect only after 16/16 canonical runs and 4/4 valid four-arm blocks.
