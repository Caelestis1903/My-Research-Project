# REVIEW ADDENDUM — F0 runner startup correction v3

Date: 2026-10-12 13:48 UTC. Independent reviewer: `/root/f0_ml_reviewer`.

## Verdict: PASS WITH FIXES

The NameError startup correction itself is PASS. No blocking defect was found in the bounded change. Overall execution-control verdict remains PASS WITH FIXES because the three nonblocking forensic limitations in `stage_F0_execution_isolation_review_v2.md` remain. The earlier review missed the undefined parser-default constant; its passing verdict must not be used for this changed snapshot without this addendum and fresh gate binding.

This review was read-only except for adding this report. No Python, tests, training, checkpoint loading, signals, source edits, gate edits or hold changes occurred.

| Inspected identity | Independently verified SHA256 |
|---|---|
| Runner | `aee9ae3207bb07260615e545dbbf3206626d639d3a90a67c6720efdd2d880818` |
| Runner tests | `abcdba06987886141df7ccc99e674f8993060c0d5d94da3182be675d56da0564` |
| Coder report | `833ba6788874c46f9613b87ac2d041aba35ae9be0c1303522af3e9a9f359639b` |
| Pending execution gate | `a26aa7db7b1e5b6375a0b3281a332b918efc4a155b21a6b37b6cf2976873b5e6` |
| 14-file control inventory | `2d88a8ae863cd7402e9a5eae9b85ff1ea0dce2451450c22b08617cd31f0d414e` |
| 28-file runtime inventory | `d55eae40cb9a8c4ff45dc4c2bc757c04d7700c7cb813e5d5601e6816bfbf6fc5` |
| Current active hold | `95898ecaf954ffe85ec355619edabaf7e018421f6edd774ad182ad8b997e6114` |
| Unchanged prior scientific search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |

## Requirement verification

- Runner now explicitly imports `EXECUTION_GATE` from the standard-library control module (line 82). `argparse` evaluates the defined default at line 2004, including when the caller overrides the option. The observed undefined-name cause is corrected without changing experiment mathematics.
- CLI reserved-slot fields default to the task wrapper's exported FD/index/owner/role. Parser converts the integer fields; explicit supervisor/trajectory flags still override these defaults. Main adopts the existing FD lease and refuses missing/incomplete prelaunch identity instead of acquiring another slot after Python startup. The original inherited-lease implementation closes only its duplicate and retains the parent's flock.
- Actual startup ordering is lease adoption → new execution-gate validation → global hold validation → frozen held set → queue read. Gate validator rejects PENDING review before reading the old search gate, queue, dataset, GPU preflight or claim state. Imports occur before this runtime gate; this is not a claim that torch is never imported by the runner probe.
- New `test_real_runner_entrypoint_uses_gate_default_and_stops_before_claim` launches the actual script through the prelaunch task wrapper. It uses a matching control inventory plus PENDING review, nonexistent queue/search paths and absent output root. Assertions require the precise gate rejection, no NameError, no queue-path error and no output creation. This exercises real parser/default/lease/main startup rather than a monkeypatched runner function. It adds one focused case, giving coder-reported **54 passed**. Reviewer read the test and source; the pass result was not independently rerun.
- All 14 actual per-file hashes match the pending gate; independently recomputed canonical JSON control/runtime digests match the handoff and runtime amendment. Gate-pinned coder-report hash, full old-gate hash and current hold hash also match their actual files. These checks used shell/Perl, not Python. Other control inputs retain the previous reviewed hashes; this correction does not reopen the eight original blockers.
- Actual gate remains `reviewer_verdict: PENDING`, amendment pending, `launch_authorized: false`, `training_started: false`; current global hold is active. Historical release/review entries remain historical and do not release this bootstrap incident. Scientific canonical code identity and prior search-gate bytes remain unchanged.

## Bugs, corrections and remaining risks

No new CRITICAL/MAJOR issue was identified in this startup correction. Reviewer made no source correction. Carry forward the three MINOR v2 limits: timeout log hashes can be capture-time rather than final-file digests; failure audit needs a separate verified last-complete checkpoint inventory before recovery; PID-only historical claim association needs nonce/start/claim evidence before attribution. Synthetic CPU optimizer unit tests are correctly distinguished from research/CUDA training in the coder report.

The new fixture verifies the rejection branch, not a fully approved successful production/CUDA startup. No native-abort root cause or scientific benefit is established. Future unexplained errors must remain held and be adjudicated separately; no automatic retry or recovery is approved here.

## RUN boundary

**RUN is still held.** The orchestrator may bind this passing addendum/current hashes and update the new gate only after an explicit audited release of the current bootstrap hold and current prelaunch/process/resource checks. Preserve all protected key holds and original scientific identities. This report does not authorize protected recovery, formal training, test evaluation or an automatic worker restart.
