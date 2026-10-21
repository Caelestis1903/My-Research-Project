#!/usr/bin/env python
"""Copy small raw research artifacts for Git delivery; checkpoints stay external."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=Path("/home_new/xzf_m3oe_reproduction_20261021"))
    p.add_argument("--destination", type=Path, default=Path(__file__).parent / "raw_results")
    args = p.parse_args()
    receipt, checkpoints = [], []
    for subtree in ["formal", "pipeline"]:
        base = args.input_root / subtree
        for source in sorted(base.rglob("*")):
            if not source.is_file() or source.suffix not in {".csv", ".json", ".log", ".txt"}:
                continue
            target = args.destination / subtree / source.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            receipt.append(dict(file=str(target.relative_to(args.destination)), sha256=sha(target), bytes=target.stat().st_size))
            if source.name == "completion.json":
                c = json.loads(source.read_text())
                if c.get("valid") and c.get("formal"):
                    checkpoints.append({k: c.get(k) for k in ["dataset", "arm", "seed", "selected_epoch", "checkpoint", "checkpoint_sha256"]})
    for source in sorted(args.input_root.glob("*")):
        if source.is_file() and source.suffix in {".json", ".jsonl", ".log", ".txt"}:
            target = args.destination / "runtime_metadata" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            receipt.append(dict(file=str(target.relative_to(args.destination)), sha256=sha(target), bytes=target.stat().st_size))
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "artifact_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    with (args.destination / "checkpoint_index.csv").open("w", newline="") as stream:
        fields = ["dataset", "arm", "seed", "selected_epoch", "checkpoint", "checkpoint_sha256"]
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(checkpoints)
    print(json.dumps(dict(copied_files=len(receipt), formal_checkpoints=len(checkpoints),
                          total_bytes=sum(r["bytes"] for r in receipt), destination=str(args.destination))))


if __name__ == "__main__":
    main()
