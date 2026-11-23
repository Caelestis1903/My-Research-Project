# Coder implementation report for the D2 review repairs

## Changes

| Finding | Implemented repair |
| --- | --- |
| D2-R1 | The worker tracks publication of `ALLOCATION.json` independently of process-residency measurements. The first baseline allocation callback emits one fresh memory measurement and guardian acknowledgment even after resumed anchor diagnostics. The scheduler uses the existing startup readiness predicate. |
| D2-R2 | Both S0's pinned donor source manifest and the independent review input bind the imported 11-22 `state.py`. The Git blob is `fb2b09ec8db04fd100c63c82d6714aa3d54f8828` and the SHA-256 is `93d10e84df94acbee5dae41c405c6b264b2b1337a6b27fe4cd6d3c604c8fcf74`. |
| D2-R3 | The standalone `s0` command exposes `prepare` and `cpu`. The staged `cli smoke --gpu <index>` entry point retains the CUDA fidelity helper and applies the review, stop and resource gates. This supersedes the earlier invocation in `CODER_CORE.md`. |
| D2-R4 | Each physical attempt atomically journals its completed diagnostic clones, native update steps and readouts in `attempts/<attempt_id>/FORK_COSTS.json`. The worker receipt summarizes these in `diagnostic_counts`. The scheduler recovers that summary when a process exits before publishing its attempt receipt. `COSTS.csv` includes unaccepted diagnostic phases as `fork_attempt` rows and counts accepted diagnostic phases once through the existing `fork` rows. |

The 19 interventions, batch mappings, timing of native Adam and EMA updates, scientific comparisons, and whole-anchor acceptance remain as registered in the design and erratum. Physical accounting captures completed phases, including the first native step when a later step of the same fork fails. The attempt wall time still includes time spent inside a phase that did not complete.

## Checks

Run these commands from the `phase0-3_mdmt_coordination_20261123` worktree.

```bash
export PYTHONDONTWRITEBYTECODE=1
D2_PYTHON=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
conda run --no-capture-output -n phase0-3-env python -m pytest -q -p no:cacheprovider experiments/mdmt_shared_private_coordination_20261123/tests
"$D2_PYTHON" -m experiments.mdmt_shared_private_coordination_20261123.s0 --help
"$D2_PYTHON" -m experiments.mdmt_shared_private_coordination_20261123.cli check
"$D2_PYTHON" -m experiments.mdmt_shared_private_coordination_20261123.cli review-input
```

- All 19 CPU tests passed in 9.06 seconds. The resumed-E10 fixture drives the worker to the E11 allocation callback twice and verifies a single fresh receipt, separate peak residency retention, and admission eligibility on a fresh card sample. The interrupted diagnostic fixture runs the real native code for a complete fork and the first step of the next fork, then verifies that three optimizer updates and four full Validation readouts survive the failure and publish once as retry work. The complete fixture matches physical and scientific counts across all 19 forks.
- The `s0` help listed `prepare` and `cpu`. The regression fixture verifies that the former CUDA action exits with an argument error.
- The project Python compiled all 12 D2 Python files. An inline check compared the loaded donor module to the pinned Git blob and SHA-256 and simulated a hash mismatch. `cli.review_input` rejected the simulated source drift.
- `cli check` passed parameter support and the exact native Adam comparison on the real first and natural tail batches for both datasets. The final `cli review-input` exited successfully. The data and panel hashes match the prior review. The parameter-support CSV binds the refreshed source manifest.

## Independent review handoff

The implementation hash is `87137de4a13274e6f79cb86d5d309a9c62278ce679bc7271f5187b587f86f861`. The [review input](../../artifacts/mdmt_shared_private_coordination_20261123/S0/REVIEW_INPUT.json) has SHA-256 `7f389ae8d9957fd2ae415331da6a870a8216642d3f10cc456ee641d094f197cb`. It binds the [source manifest](../../artifacts/mdmt_shared_private_coordination_20261123/S0/SOURCE_MANIFEST.json) with SHA-256 `db809b596df4a7c1a0006e7876fdc9cf7d6b4899e34eb17235eee7c3c601d2c4`.

The CPU checks exercise resumed control flow and cost persistence with synthetic fixtures and mocked RAM transport. Actual CUDA first/tail fidelity, live guardian recovery and the study run remain gated on independent review. Test remains sealed. PyTorch emitted one warning about the frozen donor TF32 setter during the CPU suite.

## D2-R4 restart repair

The final publisher enumerates durable `attempts/*/FORK_COSTS.json` journals directly. Unaccepted phases from a prior attempt without a receipt now appear in `COSTS.csv` as `fork_attempt` rows with `WORKER_RECEIPT_MISSING` status. Attempt-level timing and memory stay unknown for that attempt. Accepted fork keys still contribute once through the accepted `fork` rows, including when the accepted key came from the prior attempt.

The new CPU fixture reproduced the missing row with the real publisher before the fix. With the fix, the prior attempt contributes its one completed Adam step and measured phase duration. The fixture also checks accepted-fork deduplication and unknown attempt-level measurements. All 20 D2 CPU tests passed in 8.97 seconds with the existing donor TF32 deprecation warning. The donor `state.py` SHA-256 above matches the pinned machine receipt.

The refreshed implementation hash is `f45b68ef687179f756f1397314b47d46359fbd8ecbdd9312ef6da3704d819202`. The [review input](../../artifacts/mdmt_shared_private_coordination_20261123/S0/REVIEW_INPUT.json) has SHA-256 `ec0e196c5cebda442a307a86f4458a1d8f21a2542bce607ee743fda154339e2d`. The scheduler and runtime test are the only changed bound inputs. Prior S0 evidence hashes, including canonical CPU first and natural tail fidelity, match the preceding review.
