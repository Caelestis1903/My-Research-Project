"""Collect measured training and pulse costs from accepted, hash-bound scalar records."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/conditional_history_shared_update_20261122"


def load(path):
    return json.loads(path.read_bytes())


def main():
    rows = []
    for path in sorted(ART.glob("attempt*/runs/*/result.json")):
        if not path.with_name("COMPLETE_VERIFIED.json").is_file():
            continue
        result = load(path)
        job = result["job"]
        rows.append(dict(stage=job["stage"], dataset=job["dataset"], arm=job["arm"],
                         seed=job["seed"], trajectory_id=job["key"], attempt=result["attempt_id"],
                         cost_class=result["cost_class"], epochs=result["epochs"],
                         worker_seconds=result["physical_worker_seconds"],
                         peak_process_gpu_bytes=result["peak_process_gpu_bytes"],
                         max_rss_bytes=result["max_rss_bytes"], **result["counts"]))
    for receipt_path in sorted(ART.glob("B_PULSE/*/*.receipt.json")):
        receipt = load(receipt_path)
        path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", ".json"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        result = load(path)
        rows.append(dict(stage="B_PULSE", dataset=result["dataset"], arm=result["family"],
                         seed=result["seed"], trajectory_id=path.parent.name, anchor_epoch=result["epoch"],
                         cost_class="COUNTERFACTUAL_SUFFIX", worker_seconds=receipt["worker_seconds"],
                         attempt=receipt["owner_generation"], **result["counts"]))
    for receipt_path in sorted(ART.glob("A_TARGET/*/E*.receipt.json")):
        receipt = load(receipt_path)
        path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", ".json"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        result = load(path)
        rows.append(dict(stage="A_TARGET", dataset=result["dataset"], arm=result["arm"],
                         seed=result["seed"], trajectory_id=result["parent_job_key"],
                         anchor_epoch=result["anchor_epoch"], cost_class="DIAGNOSTIC_SWEEP/FORWARD",
                         worker_seconds=result["worker_seconds"],
                         process_residency_bytes=result["process_residency_bytes"], **result["counts"]))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ART / "physical_costs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
