"""Bounded D4/D1 V-row conversion diagnostic and D1 formal launcher."""
from __future__ import annotations

import argparse
import contextlib
import faulthandler
import gc
import io
import json
import os
import random
import runpy
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "research_reports/mdmt_baselines_2526"
CONFIG_PATH = ROOT / "experiments/mdmt_sota_baselines_2526/default_config.json"
SOURCE_PAIR_PATH = REPORTS / "RUNTIME_IO_SOURCE_EQUIVALENCE.json"


def _set_profile_environment(profile: str) -> dict:
    inherited = {name: os.environ.get(name) for name in
                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    if profile == "D4":
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
    elif profile == "D1":
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
    else:
        raise ValueError(f"Unknown runtime profile: {profile}")
    os.environ["PYTHONFAULTHANDLER"] = "1"
    faulthandler.enable()
    return inherited


def _load_stack(profile: str):
    inherited = _set_profile_environment(profile)
    import numpy as np
    import torch

    if profile == "D1":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("D1 could not establish torch intra/inter-op thread limits before workload")
    if os.environ.get("OMP_NUM_THREADS") != "4" or os.environ.get("MKL_NUM_THREADS") != "4":
        if profile == "D4":
            raise RuntimeError("D4 could not establish OMP/MKL thread limits before workload")
    if profile == "D1" and any(os.environ.get(key) != "1" for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("D1 could not establish all CPU-library limits before workload")
    return np, torch, inherited


def _numpy_rng_equal(left, right, np):
    return (left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:])


def _rng_state(np, torch, device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state(device).clone()}


def _rng_unchanged(before, after, np, torch):
    return {"python": before["python"] == after["python"],
            "numpy": _numpy_rng_equal(before["numpy"], after["numpy"], np),
            "torch_cpu": torch.equal(before["torch_cpu"], after["torch_cpu"]),
            "torch_cuda_device": torch.equal(before["torch_cuda"], after["torch_cuda"])}


def _array_info(value):
    return {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(value.dtype),
            "strides": list(value.strides), "owns_data": bool(value.flags.owndata),
            "writeable": bool(value.flags.writeable),
            "base_type": type(value.base).__name__ if value.base is not None else None}


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cpu_lifetime_check(np, torch):
    expected = [0.25, -1.5, 7.0, 2.125]
    temporary = np.array(expected, dtype=np.float32)
    retained = torch.as_tensor(temporary, dtype=torch.float32, device="cpu")
    del temporary
    gc.collect()
    if retained.tolist() != expected:
        raise AssertionError("CPU temporary-array lifetime check changed values")
    return True


def _cpu_selfcheck():
    np, torch, inherited = _load_stack("D1")
    before = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.get_rng_state().clone()}
    source = np.array([[1.25, 0.0], [-3.5, 4.0]], dtype=np.float32)
    owned_copy = np.array(source)
    tensor = torch.as_tensor(owned_copy, dtype=torch.float32, device="cpu")
    if not np.array_equal(tensor.numpy(), source):
        raise AssertionError("CPU conversion self-check differs from input")
    lifetime = _cpu_lifetime_check(np, torch)
    after = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state().clone()}
    rng = {"python": before["python"] == after["python"],
           "numpy": _numpy_rng_equal(before["numpy"], after["numpy"], np),
           "torch_cpu": torch.equal(before["torch_cpu"], after["torch_cpu"])}
    if not all(rng.values()):
        raise AssertionError(f"CPU conversion self-check consumed RNG state: {rng}")
    return {"status": "PASS_CPU_SYNTHETIC_ONLY", "profile": "D1",
            "value_equality": True, "temporary_lifetime": lifetime,
            "rng_unchanged": rng, "torch_threads": {
                "intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()},
            "inherited_thread_environment": inherited}


def _diagnose(profile: str, output_dir: Path, compare_d4: Path | None):
    np, torch, inherited = _load_stack(profile)
    device = torch.device("cuda:1")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("The diagnostic requires visible cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.synchronize(device)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_pair = json.loads(SOURCE_PAIR_PATH.read_text(encoding="utf-8"))
    if config["models"]["HYPERGATE"]["eval_batch_size"] != 512:
        raise ValueError("Registered HyperGate evaluation batch size changed from 512")
    prepared = ROOT / config["prepared_root"] / "kuairand"
    x_path, rows_path = prepared / "x.npy", prepared / "rows_validation.npy"
    x = np.load(x_path, allow_pickle=False)
    x.setflags(write=False)
    v_rows = np.sort(np.load(rows_path, allow_pickle=False))
    if x.ndim != 2 or x.dtype != np.float32 or v_rows.ndim != 1 or not len(v_rows):
        raise ValueError("KuaiRand X or V row IDs have invalid shape/dtype")
    if int(v_rows.min()) < 0 or int(v_rows.max()) >= len(x):
        raise ValueError("V row IDs are outside X")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "stages.jsonl"
    ids_path = output_dir / "v_row_ids.npy"
    np.save(ids_path, v_rows, allow_pickle=False)
    batch_size = 512
    batch_count = (len(v_rows) + batch_size - 1) // batch_size
    traversal_match = None
    if profile == "D1":
        if compare_d4 is None or not compare_d4.is_file():
            raise ValueError("D1 requires the D4 v_row_ids.npy path for direct traversal comparison")
        old_ids = np.load(compare_d4, allow_pickle=False)
        traversal_match = bool(np.array_equal(v_rows, old_ids))
        if not traversal_match:
            raise AssertionError("D1 V row traversal differs from the D4 traversal")
    elif compare_d4 is not None:
        raise ValueError("Only D1 accepts --compare-d4")

    runtime_output = io.StringIO()
    with contextlib.redirect_stdout(runtime_output):
        np.show_runtime()
    runtime_info = runtime_output.getvalue().strip()
    before_rng = _rng_state(np, torch, device)
    thread_env = {key: os.environ.get(key) for key in
                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    torch_threads = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    if profile == "D1" and torch_threads != {"intra_op": 1, "inter_op": 1}:
        raise RuntimeError("D1 effective torch thread limits are not both one")

    evidence = {"status": "RUNNING", "profile": profile, "dataset": "kuairand", "method_context": "HYPERGATE evaluation input",
                "device": str(device), "python_executable": sys.executable,
                "python_faulthandler_enabled": faulthandler.is_enabled(),
                "files_read": [str(x_path.resolve()), str(rows_path.resolve())],
                "y_loaded": False, "labels_read": False,
                "configuration": str(CONFIG_PATH.resolve()),
                "source_identity": {"source_equivalence_record": str(SOURCE_PAIR_PATH.resolve()),
                                    "kind": source_pair.get("kind"), "status": source_pair.get("status"),
                                    "protocol": source_pair.get("protocol"),
                                    "source_commits": config.get("source_commits")},
                "inherited_thread_environment": inherited, "effective_thread_environment": thread_env,
                "torch_threads": torch_threads, "numpy_runtime": runtime_info,
                "x": _array_info(x), "v_row_ids": _array_info(v_rows),
                "v_rows_count": int(len(v_rows)), "eval_batch_size": batch_size,
                "sweeps": 2, "batches_per_sweep": batch_count,
                "tail_batch_rows": int(len(v_rows) - (batch_count - 1) * batch_size),
                "traversal_equal_to_D4": traversal_match,
                "traversal_ids_file": str(ids_path),
                "cpu_temporary_lifetime_exact": None,
                "cuda_temporary_lifetime_exact": None,
                "array_examples": {}, "sweep_checks": []}
    _write_json(output_dir / "diagnostic.json", evidence)

    def mark(stream, stage, boundary, sweep, batch, begin, row_ids=None, **extra):
        item = {"stage": stage, "boundary": boundary, "sweep": sweep, "batch": batch,
                "row_start_index": begin, **extra}
        if row_ids is not None:
            item.update(row_count=int(len(row_ids)), first_row_id=int(row_ids[0]),
                        last_row_id=int(row_ids[-1]))
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        stream.flush()

    cpu_batches = cuda_batches = 0
    with log_path.open("x", encoding="utf-8") as stream:
        mark(stream, "cpu_temporary_lifetime", "before", 0, 0, 0)
        cpu_lifetime = _cpu_lifetime_check(np, torch)
        mark(stream, "cpu_temporary_lifetime", "after", 0, 0, 0, exact=cpu_lifetime)
        cuda_expected = [0.25, -1.5, 7.0, 2.125]
        mark(stream, "cuda_temporary_as_tensor", "before", 0, 0, 0)
        cuda_temporary = np.array(cuda_expected, dtype=np.float32)
        cuda_retained = torch.as_tensor(cuda_temporary, dtype=torch.float32, device=device)
        del cuda_temporary
        gc.collect()
        mark(stream, "cuda_temporary_as_tensor", "after", 0, 0, 0)
        mark(stream, "cuda_temporary_synchronize", "before", 0, 0, 0)
        torch.cuda.synchronize(device)
        mark(stream, "cuda_temporary_synchronize", "after", 0, 0, 0)
        mark(stream, "cuda_temporary_value_validation", "before", 0, 0, 0)
        cuda_lifetime = cuda_retained.cpu().tolist() == cuda_expected
        mark(stream, "cuda_temporary_value_validation", "after", 0, 0, 0, exact=cuda_lifetime)
        if not cuda_lifetime:
            raise AssertionError("CUDA temporary-array lifetime check changed values")
        del cuda_retained
        evidence.update(cpu_temporary_lifetime_exact=cpu_lifetime,
                        cuda_temporary_lifetime_exact=cuda_lifetime)
        _write_json(output_dir / "diagnostic.json", evidence)
        for sweep in (1, 2):
            sweep_cpu = sweep_cuda = 0
            for batch_no, begin in enumerate(range(0, len(v_rows), batch_size), start=1):
                mark(stream, "slice_v_ids", "before", sweep, batch_no, begin,
                     expected_rows=min(batch_size, len(v_rows) - begin))
                row_ids = v_rows[begin:begin + batch_size]
                mark(stream, "slice_v_ids", "after", sweep, batch_no, begin, row_ids)
                mark(stream, "advanced_index_x", "before", sweep, batch_no, begin, row_ids)
                batch = x[row_ids]
                mark(stream, "advanced_index_x", "after", sweep, batch_no, begin, row_ids)
                mark(stream, "numpy_array_copy", "before", sweep, batch_no, begin, row_ids)
                owned_copy = np.array(batch)
                mark(stream, "numpy_array_copy", "after", sweep, batch_no, begin, row_ids)
                if sweep == 1 and batch_no == 1:
                    evidence["array_examples"] = {"index_slice": _array_info(row_ids),
                                                   "advanced_index_batch": _array_info(batch),
                                                   "owned_numpy_copy": _array_info(owned_copy)}
                mark(stream, "as_tensor_cpu", "before", sweep, batch_no, begin, row_ids)
                cpu_tensor = torch.as_tensor(owned_copy, dtype=torch.float32, device="cpu")
                mark(stream, "as_tensor_cpu", "after", sweep, batch_no, begin, row_ids)
                mark(stream, "as_tensor_cuda", "before", sweep, batch_no, begin, row_ids)
                cuda_tensor = torch.as_tensor(owned_copy, dtype=torch.float32, device=device)
                mark(stream, "as_tensor_cuda", "after", sweep, batch_no, begin, row_ids)
                mark(stream, "cuda_synchronize", "before", sweep, batch_no, begin, row_ids)
                torch.cuda.synchronize(device)
                mark(stream, "cuda_synchronize", "after", sweep, batch_no, begin, row_ids)
                mark(stream, "roundtrip_value_validation", "before", sweep, batch_no, begin, row_ids)
                cpu_equal = np.array_equal(cpu_tensor.numpy(), batch)
                cuda_equal = np.array_equal(cuda_tensor.cpu().numpy(), batch)
                mark(stream, "roundtrip_value_validation", "after", sweep, batch_no, begin, row_ids,
                     cpu_exact=bool(cpu_equal), cuda_exact=bool(cuda_equal))
                if not cpu_equal or not cuda_equal:
                    raise AssertionError(f"Conversion values differ at sweep {sweep}, batch {batch_no}")
                cpu_batches += 1
                cuda_batches += 1
                sweep_cpu += 1
                sweep_cuda += 1
                if sweep == 1 and batch_no == 1:
                    evidence["array_examples"]["cpu_tensor"] = {
                        "shape": list(cpu_tensor.shape), "dtype": str(cpu_tensor.dtype),
                        "device": str(cpu_tensor.device)}
                    evidence["array_examples"]["cuda_tensor"] = {
                        "shape": list(cuda_tensor.shape), "dtype": str(cuda_tensor.dtype),
                        "device": str(cuda_tensor.device)}
                del cuda_tensor, cpu_tensor, owned_copy, batch, row_ids
            evidence["sweep_checks"].append({"sweep": sweep, "cpu_batches_exact": sweep_cpu,
                                             "cuda_batches_exact": sweep_cuda})
            _write_json(output_dir / "diagnostic.json", evidence)
    after_rng = _rng_state(np, torch, device)
    rng = _rng_unchanged(before_rng, after_rng, np, torch)
    evidence.update({"status": "PASS_DIAGNOSTIC_VALUES_AND_RNG" if all(rng.values()) else "FAIL_RNG_DRIFT",
                     "cpu_batches_exact": cpu_batches, "cuda_batches_exact": cuda_batches,
                     "rng_unchanged": rng, "stage_log": str(log_path.resolve())})
    if not all(rng.values()):
        _write_json(output_dir / "diagnostic.json", evidence)
        raise AssertionError(f"Diagnostic conversion consumed RNG state: {rng}")
    _write_json(output_dir / "diagnostic.json", evidence)
    return evidence


def _launch_context(argv):
    cli_argv = list(argv)
    if cli_argv and cli_argv[0] == "--":
        cli_argv.pop(0)
    config_path = CONFIG_PATH
    command_index = 0
    if cli_argv[:1] == ["--config"]:
        if len(cli_argv) < 3:
            raise ValueError("--config must precede an existing CLI command")
        config_path = Path(cli_argv[1]).expanduser().resolve()
        command_index = 2
    elif cli_argv[:1] and cli_argv[0].startswith("--config="):
        config_path = Path(cli_argv[0].split("=", 1)[1]).expanduser().resolve()
        command_index = 1
    commands = {"audit", "prepare-data", "smoke", "train", "evaluate", "freeze", "report"}
    if len(cli_argv) <= command_index or cli_argv[command_index] not in commands:
        raise ValueError("Pass an existing mdmt-sota-baselines CLI command after --")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    command = cli_argv[command_index]
    command_args = cli_argv[command_index + 1:]

    def value(flag, default=None):
        for i, item in enumerate(command_args):
            if item == flag and i + 1 < len(command_args):
                return command_args[i + 1]
            if item.startswith(flag + "="):
                return item.split("=", 1)[1]
        return default

    run_dir = None
    if command == "train":
        dataset, method, seed = value("--dataset"), value("--method"), value("--seed")
        if dataset and method and seed:
            phase = value("--phase", "development")
            run_dir = Path(value("--output", ROOT / "artifacts/mdmt_baselines_2526/runs" /
                                 phase / dataset / method / f"seed{seed}")).expanduser()
    elif command == "evaluate":
        run_dir = Path(value("--run-dir")).expanduser() if value("--run-dir") else None
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    dataset = value("--dataset") if command in {"train", "evaluate"} else None
    manifest_path = ROOT / config["prepared_root"] / dataset / "DATA_MANIFEST.json" if dataset else None
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.is_file() else None
    run_identity = None
    if run_dir is not None and (run_dir / "run.json").is_file():
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        run_identity = {key: run.get(key) for key in ("protocol", "phase", "method", "dataset", "seed", "status")}
        data_manifest = run.get("data_manifest", data_manifest)
    return cli_argv, config_path, config, command, run_dir, data_manifest, run_identity


def _launch_d1(sidecar_path: Path, original_argv):
    _, torch, inherited = _load_stack("D1")
    normalized_argv, config_path, config, command, run_dir, data_manifest, run_identity = _launch_context(original_argv)
    sidecar_path = sidecar_path.resolve()
    if sidecar_path.exists():
        raise FileExistsError(f"Refusing to overwrite runtime sidecar: {sidecar_path}")
    source_pair = json.loads(SOURCE_PAIR_PATH.read_text(encoding="utf-8"))
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": "LAUNCH_STARTED", "profile": "D1", "pid": os.getpid(),
              "python_executable": sys.executable, "original_cli_argv": list(normalized_argv),
              "runpy_module": "experiments.mdmt_sota_baselines_2526", "cli_command": command,
              "run_dir": str(run_dir.resolve()) if run_dir is not None else None,
              "process_environment": {key: os.environ.get(key) for key in
                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                       "NUMEXPR_NUM_THREADS", "PYTHONFAULTHANDLER")},
              "inherited_thread_environment": inherited,
              "torch_threads": {"intra_op": torch.get_num_threads(),
                               "inter_op": torch.get_num_interop_threads()},
              "source_identity": {"source_equivalence_record": str(SOURCE_PAIR_PATH.resolve()),
                                  "kind": source_pair.get("kind"), "status": source_pair.get("status"),
                                  "protocol": source_pair.get("protocol"),
                                  "source_commits": config.get("source_commits")},
              "configuration_path": str(config_path.resolve()), "configuration": config,
              "prepared_data_manifest": data_manifest, "run_identity_before_cli": run_identity}
    if record["torch_threads"] != {"intra_op": 1, "inter_op": 1}:
        raise RuntimeError("D1 launcher did not establish torch thread limits")
    _write_json(sidecar_path, record)
    sys.path.insert(0, str(ROOT))
    sys.argv = ["experiments.mdmt_sota_baselines_2526", *normalized_argv]
    try:
        runpy.run_module("experiments.mdmt_sota_baselines_2526", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        record["entrypoint_exit_code"] = int(exc.code or 0)
        record["status"] = "CLI_EXITED" if exc.code in (None, 0) else "CLI_FAILED"
        _write_json(sidecar_path, record)
        if exc.code not in (None, 0):
            raise
        return 0
    except BaseException as exc:
        record.update(status="CLI_EXCEPTION", exception_type=type(exc).__name__, exception=str(exc))
        _write_json(sidecar_path, record)
        raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("selfcheck", help="CPU-only synthetic values/lifetime/RNG check")
    diagnose = commands.add_parser("diagnose", help="one fresh-process D4 or D1 two-sweep CUDA diagnostic")
    diagnose.add_argument("--profile", choices=("D4", "D1"), required=True)
    diagnose.add_argument("--output-dir", type=Path, required=True)
    diagnose.add_argument("--compare-d4", type=Path, default=None)
    launch = commands.add_parser("launch", help="D1 runpy launch of an existing baseline CLI command")
    launch.add_argument("--sidecar", type=Path, required=True)
    launch.add_argument("cli_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        print(json.dumps(_cpu_selfcheck(), indent=2))
        return 0
    if args.command == "diagnose":
        result = _diagnose(args.profile, args.output_dir, args.compare_d4)
        print(json.dumps({key: value for key, value in result.items() if key != "numpy_runtime"}, indent=2))
        return 0
    return _launch_d1(args.sidecar, args.cli_argv)


if __name__ == "__main__":
    raise SystemExit(main())
