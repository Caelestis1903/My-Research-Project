# EXPERIMENT EXECUTION REPORT — Stage S4

## 1. Experiment ID

S4 — `equal / kuairand / natural / m3oe`; frozen queue contains 16 fresh keys (`F/T/N/U × seeds 2022–2025`) in eight prescribed two-process waves.

## 2. Git/worktree state

The workspace has no `.git` directory at its root, so a Git revision/state could not be recorded. No canonical S4 run directory existed before the run. The frozen queue and failed attempt evidence are retained under `experiments/stage_S4_queue_manifest.json` and `experiments/stage_S4_raw_results/.attempts/`.

## 3. Config and environment

- Command to freeze: `C:\Users\86199\.conda\envs\EDDAenv\python.exe experiments\stage_S4_orchestrator.py --reviewer-pass --freeze-only`
- Command to run: `C:\Users\86199\.conda\envs\EDDAenv\python.exe experiments\stage_S4_orchestrator.py --reviewer-pass`
- Frozen entry point and source inventory: queue manifest; S4 source inventory contains five modules. Source hashes at freeze/run:
  - `stage_S4_protocol.py`: `D8C9C1571682154B5C8F1E040BB9B8B8710F9019F25CA67502AD5B2B42791864`
  - `stage_S4_gradient_controls.py`: `E55025677097554FC532E94DA5AECF3E19AAAD7927230FC99A502AE5C9D46BBB`
  - `stage_S4_run.py`: `E7B415AF56F6D6999E2AEB70F47E1525390C47F7E159C60629BBB39424D2E2EC`
  - `stage_S4_orchestrator.py`: `C2E10756A0E436E0F1FAD0D209D0309F19D2148FDF4795487F566D0A95C5CF1A`
  - `stage_S4_report.py`: `2E125C511D9856BED390650826D3873591C33AC2F6F8613CA54888BFB1C2C29F`
- Python `3.8.20`; PyTorch `2.0.0`; CUDA available, one device: NVIDIA GeForce RTX 4060 Laptop GPU.
- Formal device/batch settings remained those frozen in S4. No configuration changes were made.

## 4. Seeds

Frozen seeds: 2022, 2023, 2024, 2025. The run stopped during the first wave, seed 2022. No later seed or arm was launched.

## 5. Run status

**STOPPED — technical implementation assertion in first wave; 0/8 waves promoted, 0/16 canonical runs.**

Wave `s4_FT_seed2022` launched F PID 14436 and T PID 2136. The queue observed a maximum of two live training processes. F ran from `1790231786892842900` to `1790231877976400300` ns and exited with code 1. The launcher terminated the peer T after F failed; T exited with code 1 and produced no process interval receipt. Both were launched within approximately 1 ms, but a positive actual process-interval overlap was not recorded, so the formal overlap gate did not pass. The launcher did not start another wave.

## 6. Raw metrics

No run reached a valid completed artifact set. AUC, loss, per-domain and per-task metrics: **not available**. No collector was run.

## 7. Log and evidence locations

- Frozen queue: `experiments/stage_S4_queue_manifest.json`
- Queue state/receipt: `experiments/stage_S4_queue_manifest.json.state.json`
- F attempt directory: `experiments/stage_S4_raw_results/.attempts/s4_FT_seed2022/attempt1/F/`
- F console: `experiments/stage_S4_raw_results/.attempts/s4_FT_seed2022/attempt1/F.stdout.log`
- T attempt directory: `experiments/stage_S4_raw_results/.attempts/s4_FT_seed2022/attempt1/T/`
- T console: `experiments/stage_S4_raw_results/.attempts/s4_FT_seed2022/attempt1/T.stdout.log` (empty; peer terminated before writing console output)
- No canonical run folders were promoted.

## 8. Runtime errors

F failed in `stage_S4_run.py` during `_controlled_step`, at the `shared Adam formula` assertion:

`AssertionError: S4 norm mismatch shared Adam formula: 1.2112876425028966e-06 != 0.0`

Failure receipt type was `AssertionError` with `exception_errno: null`. The queue classified it as `blocked` (`fault is not a classified transient execution error`), so no retry was permitted. The T process was terminated by whole-wave fail-fast. There was no CUDA capacity or source-hash error reported.

## 9. Missing runs and collector status

- Completed valid waves: 0/8.
- Promoted formal runs: 0/16.
- Missing formal keys: all 16 keys in the frozen manifest; the two first-wave attempts are retained as failed/incomplete evidence.
- `stage_S4_runs.csv`, `stage_S4_raw_results.csv`, `stage_S4_paired_results.csv`, `stage_S4_gradient_steps.csv`, `stage_S4_probe.csv`, `stage_S4_aggregates.csv`, and `stage_S4_summary_manifest.json`: not produced.
- Collector command was not run because the formal run matrix did not complete.
