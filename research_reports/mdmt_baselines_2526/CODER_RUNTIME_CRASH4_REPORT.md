# Runtime Crash 4 implementation report

Status: **MAJOR review finding corrected; focused review pending.** The initial independent review FAIL is preserved in its review report. The CPU synthetic self-check and manifest-schema preflight pass. The GDB checkpoint reproduction itself has not run; implementation performed no real-data evaluation, training, or E access.

## External diagnostic

`runtime_crash4_tool.py` reuses the Crash 3 external `_load_stack("D1")` helper. It sets the four CPU-library limits and PyTorch intra/inter-op counts to1 before importing the experiment package, then requires the current package bytes and the partial run's retained implementation snapshot to match the approved R2 source receipt.

The script checks the existing formal AUTOIFS/KuaiRand/2022 partial run, approved config/source commits, prepared-data manifest, and both checkpoint embedded identities. Existing `load_checkpoint` loads each model state with its strict default and restores AutoIFS ticket/temp. It also checks selected reuse epoch2/step5064 and fixed-final reuse epoch3/step5486 metadata, ticket/temp, and the audited 26 tensor/2,436,811 element state shape. The report includes all three preserved checkpoint paths; the rewind checkpoint is presence-checked and deliberately not loaded.

It reads only V/select labels. It evaluates the full V role once with the selected model, keeps that model resident, then evaluates once with a fresh fixed-final model; both use cuda:1 and the configured 4096-row batch with its final short batch. The reproduction loop retains the evaluator's NumPy-copy, float32 `as_tensor`, domain-forward, detach/CPU/NumPy, row order, and metric function. It flushes stage boundaries to `stages.jsonl` and writes logits/metrics only under a new diagnostic directory. Available reuse epoch2/3 validation summaries are compared exactly; unavailable summaries are explicitly labelled unavailable. The partial run manifest and canonical evaluation index are never written.

The diagnostic outputs include `diagnostic.json`, `stages.jsonl`, `v_row_ids.npy` with its exact shape/count/order, both views' logits and metrics, and reuse epoch2/3 exact-comparison results. It references the sibling GDB transcript and runner execution receipt at `runtime_crash4/gdb_transcript.txt` and `runtime_crash4/gdb_execution.json`. The script does not launch GDB; the runner invokes it once.

## Runner command

The diagnostic is intended to run exactly once in a fresh inferior under the installed debugger. D1 limits are established by the script before NumPy/PyTorch imports. The transcript captures faulthandler and native debugger output. A debugger exit status of zero does not mean the inferior completed normally; inspect the transcript's signal and inferior termination lines.

```bash
mkdir -p research_reports/mdmt_baselines_2526/runtime_crash4
ulimit -c 0
PYTHONFAULTHANDLER=1 /usr/bin/gdb -q -nx -batch \
  -ex 'set pagination off' \
  -ex 'set confirm off' \
  -ex 'set debuginfod enabled off' \
  -ex 'set disable-randomization off' \
  -ex 'handle SIGSEGV stop print nopass' \
  -ex run \
  -ex 'thread apply all bt 40' \
  -ex 'info sharedlibrary' \
  -ex quit \
  --args /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  research_reports/mdmt_baselines_2526/runtime_crash4_tool.py diagnose \
  --output-dir research_reports/mdmt_baselines_2526/runtime_crash4/diagnostic \
  > research_reports/mdmt_baselines_2526/runtime_crash4/gdb_transcript.txt 2>&1
```

The diagnostic output path must be new. The command makes one debugger invocation and the script makes exactly one V sweep per checkpoint view; it has no retry or stress loop.

## Verification performed

Using `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- `ast.parse` of the new script: **PASS**.
- `runtime_crash4_tool.py selfcheck`: **PASS_CPU_SYNTHETIC_ONLY**; all36 synthetic rows' logits and complete metrics exactly matched the existing evaluator, and flushed markers covered all eight loop/aggregation/metric stages in the expected order across six batches. The metadata-only preflight reads `DATA_MANIFEST.json` and resolves V through its recorded `role_map` (`V` → `validation`), yielding **250,767** rows without opening arrays, labels, or checkpoints.

No real checkpoint was loaded by this implementation check. The exact R2 check, checkpoint identity checks, full V results, GDB transcript, and native termination remain for the single runner invocation after independent review. No package, config, R2 receipt, partial/successful run, or checkpoint was changed; no dependency or hash was used.

## Focused correction after initial review

The initial independent review found that the diagnostic indexed `data.manifest["roles"]["V"]`, while the stored manifest records that component as `roles.validation`. I replaced that direct key with the existing manifest `role_map` resolution and added the metadata-only 250,767-row preflight to `selfcheck`. The original FAIL report is unchanged; no arrays, labels, checkpoints, or run files were read or edited for this correction.
