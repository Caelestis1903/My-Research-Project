"""D2 S0 engineering checks and supervised S1 runtime."""

import argparse
import copy
import gc
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch

from experiments.conditional_history_shared_update_20261122.protocol import digest, read, sha, write
from experiments.conditional_information_momentum_20261102_v3.guardian import request, serve

from . import scheduler, training

DESIGN_SHA = "7c1df7c873468de90b17ff53e32deab0721001bdd53dab2600de3b924467a74b"
S0_FILES = ("SOURCE_MANIFEST.json", "DATA_MANIFEST.json", "MODEL_FIDELITY.json", "PARAMETER_MANIFEST.json",
            "PARAMETER_SUPPORT.csv", "CPU_TESTS.json", "IMPLEMENTATION_MAP.md")
PYTHON = Path("/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python")


def guard_stop(art):
    if scheduler.stopping(art):
        raise RuntimeError("D2 stop or pause control is active")


def prepare(art):
    from . import diagnostics, s0

    guard_stop(art)
    datasets = s0.audit(art / "S0")
    for dataset, (bundle, _) in datasets.items():
        diagnostics.prepare(bundle, dataset, art)
    (art / "S0" / "IMPLEMENTATION_MAP.md").write_text(
        "# D2 implementation\n\n"
        "- `source.py` reproduces the canonical Train and Validation population.\n"
        "- `s0.py` checks source identity, Adam fidelity and parameter support.\n"
        "- `blocks.py` applies native Adam blocks and canonical EMA updates.\n"
        "- `training.py` executes J_ORIG and verifies epoch, anchor and fork RAM state.\n"
        "- `diagnostics.py` freezes panels and executes isolated S1 measurements.\n"
        "- `scheduler.py` supervises the registered S1 job matrix.\n"
        "- `statistics.py` constructs the registered scalar tables and Validation decision.\n"
    )
    return datasets


def review_input(art):
    root = scheduler.ROOT
    design = scheduler.REPORTS / "DESIGN_SPECIFICATION.md"
    erratum = scheduler.REPORTS / "DESIGN_ERRATUM.md"
    if sha(design) != DESIGN_SHA:
        raise RuntimeError("Approved design SHA-256 changed")
    paths = sorted(path for path in scheduler.PACKAGE.rglob("*") if path.suffix in (".py", ".cpp"))
    history = root / "experiments/conditional_history_shared_update_20261122"
    paths += [history / name for name in ("__init__.py", "protocol.py", "scheduler.py", "state.py", "results.py",
                                         "b_panel_identity.py", "b_build_panels.py")]
    files = {str(path.relative_to(root)): sha(path) for path in [*paths, design, erratum]}
    names = [f"S0/{name}" for name in S0_FILES] + [f"PANELS_{dataset}.json" for dataset in scheduler.DATASETS]
    evidence = {name: sha(art / name) for name in names}
    source = read(art / "S0/SOURCE_MANIFEST.json")
    for path, record in source["source_files"].items():
        if sha(root / path) != record["sha256"]:
            raise RuntimeError(f"Frozen source differs: {path}")
    for name in ("SOURCE_MANIFEST", "DATA_MANIFEST", "MODEL_FIDELITY", "PARAMETER_MANIFEST", "CPU_TESTS"):
        if read(art / "S0" / f"{name}.json")["status"] != "PASS":
            raise RuntimeError(f"Required S0 check did not pass: {name}")
    if read(art / "S0/DATA_MANIFEST.json")["test_status"] != "SEALED":
        raise RuntimeError("D2 Test seal is required")
    return dict(design_sha256=DESIGN_SHA, implementation_hash=digest(files), files=files, evidence=evidence)


def require_review(art, review_path):
    required = review_input(art)
    receipt = read(review_path)
    if (receipt.get("verdict") not in ("PASS", "PASS WITH FIXES") or receipt.get("blocking_issues") != []
            or any(receipt.get(key) != value for key, value in required.items())):
        raise RuntimeError("Independent review does not bind the reviewed, passing S0 implementation")
    return required


def start_owner(owner):
    marker = owner / "D2_OWNER"
    if (owner / "guardian.json").exists():
        if not marker.exists():
            raise RuntimeError("A D2-owned RAM guardian is required")
        return training.owner_identity(owner)
    owner.mkdir(parents=True, exist_ok=True)
    marker.touch()
    with (owner / "guardian.log").open("a") as log:
        process = subprocess.Popen([sys.executable, "-m", scheduler.MODULE, "_ram-owner",
                                    "--guardian", str(owner), "--parent", str(os.getpid())],
                                   cwd=scheduler.ROOT, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if (owner / "guardian.json").exists():
            return training.owner_identity(owner)
        if process.poll() is not None:
            break
        time.sleep(.1)
    raise RuntimeError("D2 RAM guardian did not start; see guardian.log")


def smoke_recovery(dataset, bundle, device, owner, generation):
    from . import blocks, source

    cfg = source.config(dataset)
    source.setup(2022, device)
    model = source.make_model(bundle, dataset).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = source.optimizer(model, cfg)
    loader = torch.Generator().manual_seed(13022)
    # The fixture treats the real first update as its complete checkpoint boundary.
    order = training.next_order(len(bundle.train_y), loader)
    first = order[:cfg["batch_size"]]
    blocks.step(model, opt, ema, (training.batch_inputs(bundle, first, device),), cfg, detailed=False)
    packet = training.capture(model, opt, ema, loader, first, device, epoch=1, owner=generation,
                              counts=dict(optimizer_steps=1, ema_updates=1))
    key = "D2:S0:" + uuid.uuid4().hex
    training.register(owner, key, dict(dataset=dataset, role="D2_GPU_RAM_SMOKE"))
    try:
        path = owner / f"{key.replace(':', '_')}.json"
        receipt = training.commit(owner, key, 1, path, packet, 0)
        next_order = training.next_order(len(bundle.train_y), loader)
        ix = next_order[:cfg["batch_size"]]
        args = (training.batch_inputs(bundle, ix, device),)
        expected = blocks.step(model, opt, ema, args, cfg, detailed=False)
        expected_states = training.capture(model, opt, ema, loader, ix, device)
        saved = training.verify(owner, receipt)
        training.restore(saved, model, opt, ema, loader, device)
        actual_order = training.next_order(len(bundle.train_y), loader)
        if training.v3.state_digest(actual_order) != training.v3.state_digest(next_order):
            raise RuntimeError("Restored CUDA fixture changed the next epoch permutation")
        actual = blocks.step(model, opt, ema, (training.batch_inputs(bundle, actual_order[:len(ix)], device),),
                             cfg, detailed=False)
        actual_states = training.capture(model, opt, ema, loader, ix, device)
        for field in ("model", "optimizer", "ema"):
            torch.testing.assert_close(actual_states[field], expected_states[field], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=2e-5, atol=2e-6)
        if not training.v3.equal(actual_states["rng"], expected_states["rng"]):
            raise RuntimeError("Restored CUDA fixture changed the subsequent RNG state")
        torch.cuda.synchronize(device)
        peak, telemetry = training.process_memory()
        return dict(status="PASS", receipt=receipt, next_permutation_hash=training.v3.state_digest(actual_order),
                    next_rows_hash=training.v3.state_digest(bundle.train_row_ids[ix]),
                    next_update_exact=training.v3.equal(actual_states, expected_states),
                    source_loss=expected["loss"], restored_loss=actual["loss"], process_gpu_bytes=peak,
                    telemetry=telemetry)
    finally:
        request(owner, "release", key=key)
        request(owner, "failed", key=key, failure="S0_FIXTURE_RELEASED")


def smoke(art, owner, review_path, gpu):
    from . import s0, source

    guard_stop(art)
    status_path = art / "S0/GPU_SMOKE.json"
    review = require_review(art, review_path)
    if Path(sys.executable).resolve() != PYTHON.resolve() or torch.__version__ != "2.9.0+cu126":
        raise RuntimeError("Run CUDA checks in the project Python and PyTorch environment")
    telemetry = scheduler.sample()
    card = next(row for row in telemetry["devices"] if row["index"] == gpu)
    if not scheduler.admit(card, telemetry["host_ram_used"]):
        result = dict(status="PAUSED_RESOURCE", telemetry=telemetry, reason="CUDA_SMOKE_HEADROOM")
        write(status_path, result)
        return result
    generation = start_owner(owner)
    props = {}
    for index in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(index)
        name = str(info.uuid)
        props[name if name.startswith("GPU-") else "GPU-" + name] = (index, info)
    usable = {key: value[1].total_memory for key, value in props.items()}
    device = torch.device(f"cuda:{props[card['uuid']][0]}")
    resources = dict(status="PASS", telemetry=telemetry, cuda_usable_total_bytes=usable,
                     python=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version(), cpu=platform.machine(),
                     cpu_count=os.cpu_count(), host=platform.node(),
                     gpu_names={key: value[1].name for key, value in props.items()},
                     git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=scheduler.ROOT, text=True).strip(),
                     git_status=subprocess.check_output(["git", "status", "--short"], cwd=scheduler.ROOT, text=True))
    write(art / "S0/RESOURCE_AUDIT.json", resources)
    data = read(art / "S0/DATA_MANIFEST.json")["datasets"]
    records, profiles = {}, {}
    try:
        for dataset in scheduler.DATASETS:
            guard_stop(art)
            telemetry = scheduler.sample()
            card = next(row for row in telemetry["devices"] if row["index"] == gpu)
            if not scheduler.admit(card, telemetry["host_ram_used"]):
                result = dict(status="PAUSED_RESOURCE", telemetry=telemetry, reason="CUDA_SMOKE_HEADROOM")
                write(status_path, result)
                return result
            bundle, identity = source.load_data(dataset)
            if identity != data[dataset]:
                raise RuntimeError("S0 GPU fixture data differs from reviewed identity")
            torch.cuda.reset_peak_memory_stats(device)
            fidelity = s0.joint_fidelity(dataset, bundle, device)
            torch.cuda.synchronize(device)
            before, _ = training.process_memory()
            gc.collect()
            torch.cuda.empty_cache()
            recovery = smoke_recovery(dataset, bundle, device, owner, generation)
            profiles[dataset] = max(before, recovery["process_gpu_bytes"])
            records[dataset] = dict(fidelity=fidelity, recovery=recovery,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            del bundle
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as exc:
        write(status_path, dict(status="PAUSED_RESOURCE" if isinstance(exc, torch.cuda.OutOfMemoryError)
                                else "TECHNICAL_BLOCKED", failure=repr(exc), records=records))
        raise
    result = dict(status="PASS", review_sha256=sha(review_path), implementation_hash=review["implementation_hash"],
                  evidence=review["evidence"], owner=generation, usable=usable, profiles=profiles,
                  records=records, new_scientific_runs=0, test_evaluations=0,
                  resource_sha256=sha(art / "S0/RESOURCE_AUDIT.json"))
    write(status_path, result)
    return result


def authority(art, owner, review_path):
    review = require_review(art, review_path)
    smoke_result = read(art / "S0/GPU_SMOKE.json")
    if (not (owner / "D2_OWNER").exists()
            or smoke_result["status"] != "PASS" or smoke_result["review_sha256"] != sha(review_path)
            or smoke_result["implementation_hash"] != review["implementation_hash"]
            or smoke_result["evidence"] != review["evidence"]
            or smoke_result["resource_sha256"] != sha(art / "S0/RESOURCE_AUDIT.json")
            or smoke_result["owner"] != training.owner_identity(owner)):
        raise RuntimeError("S1 requires passing real CUDA and RAM recovery evidence for this reviewed source and owner")
    for dataset in scheduler.DATASETS:
        item = smoke_result["records"][dataset]
        if (item["fidelity"]["status"] != "PASS" or item["recovery"]["status"] != "PASS"
                or item["recovery"]["receipt"]["status"] != "RAM_VERIFIED"):
            raise RuntimeError("Required first/tail CUDA or genuine RAM readback did not pass")
    source_hash = digest(dict(implementation=review["implementation_hash"],
                              donor=review["evidence"]["S0/SOURCE_MANIFEST.json"]))
    data = read(art / "S0/DATA_MANIFEST.json")["datasets"]
    return dict(source_hash=source_hash, datasets={d: data[d]["data_hash"] for d in scheduler.DATASETS},
                panels={d: read(art / f"PANELS_{d}.json")["panel_hash"] for d in scheduler.DATASETS},
                usable=smoke_result["usable"], profiles=smoke_result["profiles"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "review-input", "smoke", "run", "analyze", "_worker",
                                           "_ram-owner"))
    parser.add_argument("--artifacts", type=Path, default=scheduler.ARTIFACTS)
    parser.add_argument("--guardian", type=Path)
    parser.add_argument("--review", type=Path, default=scheduler.REPORTS / "IMPLEMENTATION_REVIEW.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--logical-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--parent", type=int)
    args = parser.parse_args(argv)
    art = args.artifacts.resolve()
    owner = (args.guardian or art / "ram").resolve()
    if not owner.is_relative_to(art):
        parser.error("D2 RAM owner must live within its artifact directory")
    if args.action in ("_worker", "_ram-owner") and (not args.parent or os.getppid() != args.parent):
        parser.error("Mechanical subprocess must be spawned by the current foreground D2 runner")
    if args.action in ("prepare", "check", "smoke", "run", "_worker"):
        guard_stop(art)
    result = None
    if args.action == "prepare":
        prepare(art)
    elif args.action == "check":
        from . import s0
        s0.cpu_checks(art / "S0", prepare(art))
    elif args.action == "review-input":
        result = review_input(art)
        write(art / "S0/REVIEW_INPUT.json", result)
    elif args.action == "smoke":
        result = smoke(art, owner, args.review, args.gpu)
    elif args.action == "run":
        result = scheduler.supervise(art, owner, authority(art, owner, args.review))
    elif args.action == "_worker":
        manifest = read(art / "S1_RUN_MANIFEST.json")
        if read(art / "S1_RUN.json")["status"] != "RUNNING":
            raise RuntimeError("D2 mechanical worker requires S1 foreground supervision")
        job = next(job for job in manifest["jobs"] if job["logical_id"] == args.logical_id)
        training.worker(art, owner, job, args.attempt_id, "cuda:0",
                        lambda: os.getppid() != args.parent or scheduler.stopping(art))
    elif args.action == "_ram-owner":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        serve(owner)
    else:
        from . import statistics
        result = statistics.analyze(art)
    if result is not None and "status" in result:
        print(result["status"])


if __name__ == "__main__":
    main()
