"""Claim the first frozen search key and run exactly one epoch through F0Runner API."""
import json
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from stage_F0_data import load_b1_bundles
from stage_F0_protocol import protocol_hash
from stage_F0_queue import (QueueFileLock, acquire_process_slot, claim_next_entry,
                            code_hash, queue_status_path)
from stage_F0_run import F0Runner, runtime_preflight, validate_run_gate
from stage_F0_worker_control import (PRESERVED_H49_RUN_IDS,
                                     assert_no_global_hold,
                                     validate_execution_gate)


def main():
    # This historical helper can create a child run without the audited
    # supervisor's process-group, receipt, and per-child gates. Keep it
    # permanently closed under the v1.4 execution model.
    raise PermissionError(
        "manual first-epoch execution is disabled; use the reviewed v1.4 worker supervisor")

    queue_path = os.path.join(ROOT, "experiments/stage_F0_outputs/queues/F0-B1-ML-search-queue.json")
    gate_path = os.path.join(ROOT, "experiments/stage_F0_outputs/run_gate_search.json")
    output_root = os.path.join(ROOT, "experiments/stage_F0_outputs/runs")
    # The manual first-epoch helper shares the production fail-closed gate and
    # cannot bypass v1.3 H49 evidence/hold checks before data or CUDA access.
    validate_execution_gate()
    assert_no_global_hold(output_root)
    dataset_path = "/home/xzf/data/ml-1m/ml-1m.csv"
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)
    entry = dict(queue["entries"][0])
    if entry["backbone"] != "m3oe" or entry["scope"] != "mdmt" or entry["seed"] != 2018:
        raise RuntimeError("frozen first search key differs from expected M3OE/MDMT/development-2018")
    slot = acquire_process_slot(role="F0-first-search-epoch")
    gpu_lock = canonical_lock = None
    try:
        bundles = load_b1_bundles(dataset_path)
        dataset_hash = bundles["mdmt"].preprocessing_metadata["raw_file_sha256"]
        digest = code_hash(require_tests=True)
        preflight = runtime_preflight(output_root)
        gpu_index = 1
        if gpu_index not in preflight["resource_concurrency_policy"]["usable_gpu_indices"]:
            raise RuntimeError("GPU1 did not pass current resource preflight")
        gpu = preflight["gpus"][gpu_index]
        gpu_lock = QueueFileLock(os.path.join(output_root, "gpu_locks", "gpu{}.lock".format(gpu_index)))
        gpu_lock.write_owner({"pid": os.getpid(), "host": socket.gethostname(), "gpu_index": gpu_index,
                              "gpu_uuid": gpu["uuid"], "claimed_unix": time.time()})
        entry, canonical_lock, resume_path = claim_next_entry(
            {**queue, "entries": [entry]}, output_root, gpu_uuid=str(gpu["uuid"]),
            gpu_index=gpu_index, held_run_ids=list(PRESERVED_H49_RUN_IDS))
        if entry is None or canonical_lock is None or resume_path is not None:
            raise RuntimeError("first search identity was unavailable or already had a checkpoint")
        with open(gate_path, encoding="utf-8") as f:
            gate = json.load(f)
        validate_run_gate(gate, entry=entry, queue_hash=queue["queue_hash"],
                          dataset_file_hash=dataset_hash)
        if digest != entry["code_hash"] or protocol_hash() != entry["protocol_hash"]:
            raise RuntimeError("implementation/protocol hash changed after queue freeze")
        attempt = os.path.join(output_root, entry["run_id"], "attempts", entry["attempt_id"])
        os.makedirs(attempt, exist_ok=True)
        status = queue_status_path(output_root, entry["run_id"])
        runner = F0Runner(entry, bundles[entry["scope"]], attempt,
                          device="cuda:{}".format(gpu_index), code_digest=digest,
                          protocol_digest=protocol_hash())
        result = runner.run(max_epochs=1, gate_verified=True,
                            canonical_lock=canonical_lock, gpu_lock=gpu_lock,
                            status_file=status)
        canonical_lock = None
        gpu_lock = None
        with open(os.path.join(attempt, "first_epoch_resource_receipt.json"), "w", encoding="utf-8") as f:
            json.dump({"run_id": entry["run_id"], "run_id_full_sha256": entry["run_id_full_sha256"],
                       "attempt_id": entry["attempt_id"], "pid": os.getpid(), "host": socket.gethostname(),
                       "gpu_index": gpu_index, "gpu_uuid": gpu["uuid"], "one_epoch_status": result["status"],
                       "completed_epochs": 1, "first_epoch_resource_metrics": os.path.join(attempt, "resource_epoch_metrics.jsonl"),
                       "resume_checkpoint": os.path.join(attempt, "resume_epoch_001.pt"),
                       "resume_checkpoint_bytes": os.path.getsize(os.path.join(attempt, "resume_epoch_001.pt")),
                       "resume_exact": True, "next_action": "claim same canonical key and resume to 50 epochs"},
                      f, ensure_ascii=False, sort_keys=True, indent=2)
        print(json.dumps({"run_id": entry["run_id"], "status": result["status"],
                          "completed_epochs": 1, "resume_checkpoint_bytes": os.path.getsize(
                              os.path.join(attempt, "resume_epoch_001.pt"))}))
    finally:
        if canonical_lock is not None:
            canonical_lock.close()
        if gpu_lock is not None:
            gpu_lock.close()
        slot.close()


if __name__ == "__main__":
    main()
