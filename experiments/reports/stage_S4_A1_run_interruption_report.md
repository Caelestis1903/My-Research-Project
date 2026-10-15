# EXPERIMENT EXECUTION REPORT — Stage S4 A1 interruption

## 1. Experiment ID

Stage S4 Amendment A1. The frozen matrix contains 16 formal keys, seeds 2022–2025, arms F/T then N/U, and eight ordered two-process waves. Only the first wave was attempted.

## 2. Git/worktree state

The repository has no Git metadata at the workspace root, so a Git revision is unavailable. The frozen producer inventory is recorded in `experiments/stage_S4_queue_manifest_A1.json`.

## 3. Queue and source identity

- Queue: `experiments/stage_S4_queue_manifest_A1.json`
- Queue SHA-256: `C6A8F1322213BD001F6994D42FF009E460E869E6D0624903DE08E12088600D66`
- Producer inventory digest: `b82283018cc5d165b8f08ca4582a1537f7d6b10750de6c5ded68857aff13f7d1`
- Frozen S4 source SHA-256 values:
  - `stage_S4_gradient_controls.py`: `e695bbfafafb9b238d5cb691fde5113bc7e87f2b7699e766fb84d71167090645`
  - `stage_S4_orchestrator.py`: `0f44b29dc95e9b4aef00b7fd0ea04301dcef05bf95c1dcafc7ea2d78e7e4bf2c`
  - `stage_S4_protocol.py`: `0b95303373d37afa02e80abd82a83bd9ead5c7b980c0a4e4e555ee7b4031a858`
  - `stage_S4_report.py`: `c891d456c3548719b569e1da2b7601d5427540c9c0b523043a2fe6b15e5af593`
  - `stage_S4_run.py`: `8017e93510f4e25cf0d2ce7ea7c1208fe4a73ea04b4cf94038621580407169c3`
  - M3OE forward model `m3oe/MDMTRec/models/multi_domain/model.py`: `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1`
- The queue binds the prior abort receipt: predecessor queue SHA `693b46697ba40bf6e9cd03c6e0d4688a3ee7aa80a5d6ce2a442bed5c846fb221`, state SHA `8ebff1bc35dd7fa5a182cfb3d56b88f7e336bedb66cd93fba588c5bc188c622e`, and abort-report SHA `52b6d5924fb60c6453e669d6318a11cc064cf8858c770b77440ff30e33a5d30e`.

## 4. Configuration and commands

Runtime: `C:\Users\86199\.conda\envs\EDDAenv\python.exe`; device `cuda:0`; maximum live training processes: 2.

Freeze command:

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_orchestrator.py --a1 --reviewer-pass --freeze-only
```

Formal run command:

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_orchestrator.py --a1 --reviewer-pass
```

No collector command was run.

## 5. Run status

- Wave `s4_FT_seed2022`, attempt 1: F and T launched concurrently.
- Launch PIDs: F `26688`; T `22144`.
- Orchestrator recorded `max_live_observed=2`; both children exited with code 1. Neither peer was terminated by the orchestrator.
- Per-child wall time recorded in `failure.json`: F `9155.1919s`; T `9155.0754s`. Run manifests recorded approximately `9151.8s` for each before output-finalization failure.
- Attempt retry classification: `blocked`; reason `no single explicit transient child fault receipt`.
- Queue state: `completed=[]`; one attempt record; accepted waves `0/8`.
- Canonical runs: `0/16` promoted. The attempt artifacts are unpromoted and do not count as valid formal runs.
- No retry, collection, or analysis was performed.

## 6. Runtime error

Both child logs record the same finalization exception after writing attempt artifacts:

```text
Traceback (most recent call last):
  File "D:\CODE\phase0-3\experiments\stage_S4_run.py", line 341, in <module>
    raise SystemExit(main())
  File "D:\CODE\phase0-3\experiments\stage_S4_run.py", line 324, in main
    runner.write_outputs(info, time.time() - started)
  File "D:\CODE\phase0-3\experiments\stage_S4_run.py", line 255, in write_outputs
    super().write_outputs(fit_info, wall_seconds)
  File "D:\CODE\phase0-3\experiments\stage_S3_run.py", line 432, in write_outputs
    protocol = formal_protocol_for(self.fresh_role)
  File "D:\CODE\phase0-3\experiments\stage_S3_protocol.py", line 145, in formal_protocol_for
    raise ValueError("unknown Stage S3 fresh role")
ValueError: unknown Stage S3 fresh role
```

`failure.json` records `exception_type=ValueError`, `exception_errno=null`, and `exception="unknown Stage S3 fresh role"` for both F and T. The implementation error prevented attempt validation and promotion.

## 7. Preserved evidence locations

Attempt root: `experiments/stage_S4_A1_raw_results/.attempts/s4_FT_seed2022/attempt1/`.

- Complete child console logs: `F.stdout.log`, `T.stdout.log`; each contains the traceback above.
- Per-arm failure receipts: `F/failure.json`, `T/failure.json`.
- Per-arm process receipts: `F/process_interval.json`, `T/process_interval.json` (both status `FAILED`).
- Per-arm attempt outputs include preliminary run manifests, metric CSVs, prediction arrays, optimizer/RNG state, gradient-step logs and checkpoints. These remain in the failed attempt and were not collected or accepted.
- Queue state and frozen manifest remain at `experiments/stage_S4_queue_manifest_A1.json.state.json` and `experiments/stage_S4_queue_manifest_A1.json`.

## 8. Missing runs and collection status

All 16 canonical runs are missing; only the F/T seed-2022 failed attempt directories exist. The collector was not run because the required `16/16` canonical-run and four-arm-block validation gates were not met. No scientific metrics are reported here.
