# Runner attestation — v1.4 prelaunch correction preparation

Preparation ID: `f0-v14-prelaunch-correction-v2-20261013`  
Attestation recorded: 2026-10-13.  
Scope: preparation steps 1–5 only. No canonical-entry switch, gate/hold/ledger/status modification, release, or launch was performed.

## Immutable artifact identity

- Original canonical audit: `experiments/stage_F0_outputs/runs/execution_control/prelaunch_audit_v14_fresh_search.json`, SHA256 `12c723a4fff4115e68bf03c4283adcca6b225cf739fdf1e5dc0846215ac59c39`, regular file, mode `0444`, device/inode `2049/54658476`.
- Original archive: `experiments/stage_F0_outputs/runs/execution_control/prelaunch_audit_v14_fresh_search_original_12c723a4.json`, the same SHA256, device/inode, and mode. It was created as an exclusive hard link with `os.link(..., follow_symlinks=False)`; the file and parent directory were fsynced. A later preparation pass rechecked its bytes and inode.
- Corrected audit: `experiments/stage_F0_outputs/runs/execution_control/prelaunch_audit_v14_fresh_search_corrected_v2.json`, SHA256 `8945203e534e033de8ab3bf5eea60bbc367e110cef24809710f5615c46f4da9e`, mode `0444`.
- Correction manifest: `experiments/stage_F0_outputs/runs/execution_control/prelaunch_audit_v14_fresh_search_correction_v2.json`, SHA256 `24cbe29da3b12e09805922223509e5f5b58dc7ccc79224d219649552f912fd0d`, mode `0444`.

The corrected audit contains the full original audit graph and fresh observations. Its top-level integer `disk_free_bytes` is `291559395328`, equal to the nested `disk.free_bytes`. Its authorization fields for launch, release, retry, resume, new attempt, checkpoint load, formal, and test are all false. Both audit and manifest say canonical switching requires independent review; neither authorizes switching, release, or launch.

## Observation methods and results

The successful preparation ran through the required project environment with:

```text
conda run --no-capture-output -n phase0-3-env python /tmp/correction_prepare.py
```

The script held `execution_control/execution_state.lock` while checking the frozen graph and collecting the observations below. The audit/manifest observation stamp is `2026-10-13T05:08:14.724918Z` (Unix `1791868094.724929`). The GPU query returned at Unix `1791868094.0224078`, within the same locked observation pass.

- **GPU:** Queried with `nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits`. GPU 0: UUID `GPU-4ac4b341-3045-1093-fec3-492516863e5a`, used `4427087872` bytes / total `25769803776` bytes (4222/24576 MiB; fraction `0.1717936198`). GPU 1: UUID `GPU-2a59fb49-6bef-5320-74ab-ffb24d056908`, used `15728640` bytes / total `25769803776` bytes (15/24576 MiB; fraction `0.0006103516`). Both were below 90%. CUDA visibility and PyTorch version were read with `torch.cuda.is_available()` and `torch.__version__` (`True`, `2.9.0+cu126`).
- **Processes:** Enumerated process arguments using `ps -eo pid=,ppid=,args=`. No F0 worker/child process lines were found. The audit records one task Python process and one audit Python process (the preflight itself), zero other project Python lines, zero F0 worker Python processes, and zero F0 worker process Python processes.
- **Claims and queue:** Under the execution-state lock, used `stage_F0_postabnormal_v14._queue_and_statuses(runs)` to inspect all 288 fixed queue entries and their status files. Counts were 97 `VALIDATION_COMPLETE`, 51 `RECOVERY_BLOCKED`, 140 `UNCLAIMED`, zero `CLAIMED`, and zero `RUNNING`; all 51 H51 entries were blocked.
- **Production locks:** Probed the 293 paths recorded in the original audit. For present lock files, opened existing files read-only and attempted `fcntl.flock(fd, LOCK_EX | LOCK_NB)`, then released successful probes. The execution-state lock was already held by this audit, so its availability was evidenced by successful acquisition. All 293 were available; 114 existed and 179 were absent. Absent locks were not created.
- **Scheduler slots:** Probed the five existing files under `/tmp/phase0-3-F0-scheduler-a095f9a9ed5f9da0/slot{i}.lock` with read-only opens and nonblocking exclusive `flock`. All five were available; no slot files were created.
- **Disk:** Used `shutil.disk_usage` on `/home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs`: free `291559395328` bytes, total `983344672768` bytes, used `641758580736` bytes.
- **Frozen graph:** Validated the source snapshot with `validate_source_snapshot(..., allow_live_fresh_progress=False)`. Rechecked the source snapshot, queue, control inventory, migration ledger, committed active-hold hash, H51 statuses, and immutable implementation-review bindings. The current active hold SHA was `29597e3ca48a14137c1214874c3281913436a8683174c4d776504c23a663fb86`. No v1.4 release artifacts existed.

## Creation and durability

The corrected audit and manifest were created only if all required checks passed. Each was written with exclusive creation (`O_CREAT | O_EXCL`) at mode `0444`; JSON was serialized with sorted keys and indentation, written and flushed, the file descriptor was fsynced, and the containing directory was fsynced. The immutable corrected-audit SHA is bound in the manifest. The canonical audit remains byte-for-byte unchanged at its original SHA.

Two earlier immutable NO-GO preparation records remain in the execution-control directory. They resulted from an initial runner check that compared the pre-migration pending-gate hold fields to the post-migration active hold. The successful pass instead verified the pending gate through the original audit’s bound SHA and verified the current active hold through the committed migration ledger. The NO-GO records were preserved and were not edited.

This attestation verifies preparation provenance only. Independent review of the exact corrected audit, manifest, and this attestation remains required before any canonical-entry switch.


Correction note: v2 corrects the transcription typo in the original canonical SHA above. All other attestation content is unchanged from v1.
