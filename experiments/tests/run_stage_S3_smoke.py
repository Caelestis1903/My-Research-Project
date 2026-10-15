"""Nonformal Stage S3 objective/access and two-process CUDA smoke."""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
from stage_S3_orchestrator import concurrent_resource_smoke
from stage_S3_run import run_s3_update_smoke


def main() -> int:
    cpu = [run_s3_update_smoke(torch.device("cpu"), objective=objective, representative=False)
           for objective in ("equal", "sample")]
    print(json.dumps({"cpu_updates": cpu}, sort_keys=True))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen two-process Stage S3 smoke")
    resource = concurrent_resource_smoke()
    print(json.dumps(resource, sort_keys=True))
    if not resource.get("passed"):
        raise RuntimeError("Stage S3 two-process CUDA smoke failed")
    print("STAGE_S3_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
