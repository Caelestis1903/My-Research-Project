# EMA causal attribution

Run from the pinned source checkout with `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`. The normative specification is `research_plans/26-11-10_ema_causal_attribution/00_EXECUTION_RUNBOOK.md`.

`run.py` exposes `audit`, `prepare`, `smoke`, `run`, `resume`, `status`, `analyze`, `freeze`, `evaluate`, `report`, and `release`. Every public command accepts `--plan-dir` and `--out`. `run_all.sh ATTEMPT_DIRECTORY` executes the sequence and waits for the orchestrator's scientific findings before Test access. Put the independent reviewer verdict in `ATTEMPT_DIRECTORY/IMPLEMENTATION_REVIEW.md` before `run`; the verdict line is `PASS` or `PASS WITH FIXES` after blocking fixes have been completed.

The guardian is a separate process. It retains model/Adam/EMA/history state in RAM, receives immutable anchor packets once, and keeps only O-selected/F-selected/O-at-F-selected after each five-seed block. Killing a terminal does not release the guardian. `resume` uses the live last epoch transaction; guardian loss triggers same-seed deterministic replay, once per job. No checkpoint or parameter file is written. Local prediction/metric output has a 2 GiB cap and model payload has a 4 GiB cap.

`run` finishes all live B/C/D/H operations before releasing block anchors. Results are in `blocks/DATASET__OBJECTIVE/ARTIFACT_INDEX.json`; logical tables map to compressed JSON and hashes. Ordinary epoch rows are under `runs/JOB/epochs`. `analyze` consumes only these committed observations.

Before `freeze`, the orchestrator writes `MECHANISM_FINDINGS.json` with `author_role: "orchestrator"`, `main_direction: "A"|"B"|"C"|"D"`, and `findings` containing Q1–Q5, their claims, evidence strength and linked evidence. The freeze binds these scientific findings and all current observation JSON receipts. Test has exactly five logical states per job. `report` generates the numerical report skeleton from findings and real endpoints; the orchestrator completes the research narrative before `release`.

The exact transition kernel compiles `pairs_kernel.cpp` with the existing C++ compiler into `/tmp/ema_causal_code`; that directory contains compiled code only. Engineering tests cover exact ties, state math, real model supports/cuts, derivatives, bridge leakage and RAM transaction failure. Real-data smoke covers full batch, tail batch, online/gradient/Adam/F/RNG/loader purity, a failed worker after committing RAM state, recovery followed by the next real update, and full-population causal readout costs.
