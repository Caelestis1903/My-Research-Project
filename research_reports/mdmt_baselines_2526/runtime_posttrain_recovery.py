"""One-use completion of the preserved AUTOIFS/KuaiRand/2022 checkpoints.

`selfcheck` is CPU-synthetic only. `recover` is the separately reviewed,
single-use CUDA postprocessing path; it does not resume or train a model.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import json
import math
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "research_reports/mdmt_baselines_2526"
RUN_DIR = ROOT / "artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2022"
RUN_PATH = RUN_DIR / "run.json"
EPOCH_LOG = RUN_DIR / "epoch_metrics.jsonl"
CRASH3_DIR = REPORTS / "runtime_crash3"
CRASH4_DIR = REPORTS / "runtime_crash4"
CRASH4_DIAGNOSTIC = CRASH4_DIR / "diagnostic/diagnostic.json"
CRASH4_EXECUTION = CRASH4_DIR / "gdb_execution.json"
INCIDENTS_PATH = REPORTS / "RUN_INCIDENTS.json"
SOURCE_PAIR_PATH = REPORTS / "RUNTIME_IO_SOURCE_EQUIVALENCE.json"
RECOVERY_DIR = REPORTS / "runtime_posttrain_recovery/AUTOIFS_kuairand_seed2022"
ORIGINAL_RECEIPT = RECOVERY_DIR / "original_training_run.json"
RECOVERY_RECEIPT = RECOVERY_DIR / "posttraining_recovery.json"
RECOVERY_ENVIRONMENT = RECOVERY_DIR / "runtime_environment.json"
EXECUTION_RECEIPT = RECOVERY_DIR / "recovery_execution.json"

EXPECTED_PROTOCOL = "MDMT-SOTA-2526-v1"
EXPECTED_SOURCE_COMMITS = {
    "hypergate": "fdc82e3f3feee23694a8123f91092615f977900e",
    "recstudio": "f08cca607e998e8be4c5e915a393346baffa71f6",
    "autoifs": "65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e",
    "repository_main": "8eb461f5e0abc9c06385a4e8239e9621cae87ef8",
    "repository_reference": "97151c74544c3651feb805d1898bc9ce1b8d12d3",
}
EXPECTED_RUN = {"protocol": EXPECTED_PROTOCOL, "phase": "formal", "method": "AUTOIFS",
                "dataset": "kuairand", "seed": 2022}
EXPECTED_CHECKPOINTS = {
    "rewind": {"filename": "rewind_epoch9.pt", "stage": "search_rewind_start",
               "epoch_index": 9, "global_steps": 3798, "next_epoch": 9,
               "optimizer_count": 2, "optimizer_steps": 3798, "size_bytes": 28941869},
    "selected": {"filename": "selected.pt", "stage": "reuse", "epoch": 2,
                  "selected_epoch": 2, "global_steps": 5064, "next_epoch": 3,
                  "optimizer_count": 1, "optimizer_steps": 844, "size_bytes": 34525463},
    "fixed_final": {"filename": "fixed_final.pt", "stage": "reuse", "epoch": 3,
                     "selected_epoch": 2, "global_steps": 5486, "next_epoch": 4,
                     "optimizer_count": 1, "optimizer_steps": 1266, "size_bytes": 34525811},
}
EXPECTED_STATE_TENSORS = 26
EXPECTED_STATE_ELEMENTS = 2_436_811
SUMMARY_FIELDS = ("cells", "macro_auc", "macro_logloss", "natural_auc",
                  "natural_auc_defined_requests", "macro_auc_defined_cells",
                  "macro_logloss_defined_cells", "natural_logloss", "pooled_auc_by_task",
                  "pooled_auc_valid_requests_by_task", "valid_requests")
EMBEDDED_FIELDS = ("protocol", "phase", "method", "dataset", "seed", "config",
                   "source_commits", "data_manifest", "selection_metric", "environment",
                   "implementation_identity", "output_dir", "started_at_utc")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_new_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _replace_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _embedded_identity(run):
    return {field: run.get(field) for field in EMBEDDED_FIELDS}


def _validate_run_identity(run, expected, output_dir):
    _require(isinstance(run, dict), "Original run manifest is not an object")
    for key, value in expected.items():
        _require(run.get(key) == value, f"Original run {key} differs from the fixed recovery target")
    _require(run.get("status") == "TRAINING", "Original canonical run must remain TRAINING")
    _require(run.get("source_commits") == EXPECTED_SOURCE_COMMITS,
             "Original run pinned source commits differ")
    _require(run.get("selection_metric") == "V_macro_logloss", "Original selection metric differs")
    _require(Path(run.get("output_dir", "")).resolve() == Path(output_dir).resolve(),
             "Original run output directory differs")
    _require(run.get("e_policy") == "locked_during_training", "E policy differs")
    _require("posttraining_recovery" not in run and "completion_mode" not in run,
             "Run already contains recovery metadata")


def _validate_crash4_gate(diagnostic, execution, expected_run, *, expected_rows=None):
    _require(diagnostic.get("status") == "CHECKPOINT_V_REPRODUCTION_COMPLETE",
             "Crash4 diagnostic did not complete both exact V views")
    _require(execution.get("command_invocation_count") == 1
             and execution.get("gdb_exit_code") == 0
             and execution.get("inferior_outcome") == "EXITED_NORMALLY"
             and execution.get("inferior_received_sigsegv") is False
             and execution.get("classification") == "NOT_REPRODUCED"
             and execution.get("diagnostic_status") == "CHECKPOINT_V_REPRODUCTION_COMPLETE"
             and execution.get("profile") == "D1" and execution.get("device") == "cuda:1",
             "Crash4 GDB execution receipt is not a clean complete inferior exit")
    _require(diagnostic.get("configuration") == expected_run.get("config"),
             "Crash4 configuration differs from the preserved run")
    _require(diagnostic.get("partial_run_status_preserved") == "TRAINING"
             and diagnostic.get("prepared_data_manifest_verified") is True
             and diagnostic.get("training_or_optimizer_steps") == 0
             and diagnostic.get("labels_accessed") == ["V/select"]
             and diagnostic.get("e_labels_read") is False
             and diagnostic.get("run_manifest_written") is False
             and diagnostic.get("canonical_evaluation_index_written") is False
             and diagnostic.get("selected_model_retained_during_fixed_final") is True
             and diagnostic.get("gdb_managed") is True
             and diagnostic.get("profile") == "D1" and diagnostic.get("device") == "cuda:1"
             and diagnostic.get("torch_threads") == {"intra_op": 1, "inter_op": 1}
             and diagnostic.get("batch_size")
             == expected_run["config"]["models"]["AUTOIFS"]["batch_size"],
             "Crash4 scope, original status, or evaluation identity flags are invalid")
    _require(diagnostic.get("source_identity", {}).get("source_equivalence_record")
             == str(SOURCE_PAIR_PATH.resolve())
             and diagnostic.get("source_identity", {}).get("r2_source_snapshot_verified") is True
             and diagnostic.get("source_identity", {}).get("partial_run_snapshot_verified_as_r2") is True
             and diagnostic.get("source_identity", {}).get("current_package_equals_r2") is True
             and diagnostic.get("source_identity", {}).get("source_commits") == EXPECTED_SOURCE_COMMITS,
             "Crash4 source identity is not the exact reviewed R2 source")
    identity = diagnostic.get("run_identity")
    _require(isinstance(identity, dict), "Crash4 has no direct run identity record")
    for key, value in {**EXPECTED_RUN, "status": "TRAINING",
                       "output_dir": str(RUN_DIR.resolve()),
                       "selection_metric": "V_macro_logloss"}.items():
        _require(identity.get(key) == value, f"Crash4 run identity differs at {key}")
    _require(Path(diagnostic.get("execution_receipt_path", "")).resolve()
             == CRASH4_EXECUTION.resolve(), "Crash4 diagnostic names a different execution receipt")
    checkpoints = diagnostic.get("checkpoints")
    _require(isinstance(checkpoints, dict) and set(checkpoints) == set(EXPECTED_CHECKPOINTS),
             "Crash4 checkpoint inventory is incomplete")
    for name, spec in EXPECTED_CHECKPOINTS.items():
        item = checkpoints[name]
        expected_path = (RUN_DIR / spec["filename"]).resolve()
        _require(item.get("filename") == spec["filename"]
                 and Path(item.get("path", "")).resolve() == expected_path
                 and item.get("size_bytes") == spec["size_bytes"],
                 f"Crash4 {name} checkpoint identity differs")
        _require((item.get("present") is True and item.get("loaded_by_diagnostic") is False
                  if name == "rewind"
                  else item.get("strict_state_load") is True),
                 f"Crash4 {name} checkpoint presence/load evidence is missing")
        if name != "rewind":
            _require(item.get("strict_state_load") is True
                     and item.get("stage") == spec["stage"]
                     and item.get("epoch") == spec["epoch"]
                     and item.get("global_steps") == spec["global_steps"]
                     and item.get("state_tensor_count") == EXPECTED_STATE_TENSORS
                     and item.get("state_element_count") == EXPECTED_STATE_ELEMENTS
                     and item.get("ticket") is True and item.get("temp") == 100.0
                     and item.get("embedded_run_identity_matches") is True
                     and item.get("embedded_source_snapshot_matches_r2") is True,
                     f"Crash4 {name} checkpoint metadata/source identity differs")
    for view, checkpoint in (("selected", "selected"), ("fixed_final", "fixed_final")):
        item = diagnostic.get("views", {}).get(view)
        expected = EXPECTED_CHECKPOINTS[checkpoint]
        _require(isinstance(item, dict) and item.get("checkpoint_path") == str(
            (RUN_DIR / expected["filename"]).resolve()), f"Crash4 {view} checkpoint path differs")
        _require(item.get("status") == "EXACT_EPOCH_METRICS"
                 and item.get("reference_stage") == "reuse"
                 and item.get("reference_epoch") == expected["epoch"]
                 and item.get("reference_available") is True
                 and item.get("metrics_exact") is True
                 and item.get("metric_differences") == []
                 and item.get("reference_path") == str(EPOCH_LOG.resolve()),
                 f"Crash4 {view} is missing exact reuse-epoch V parity")
        _require(item.get("rows") == expected_rows if expected_rows is not None else True,
                 f"Crash4 {view} V row count differs")
        execution_view = execution.get("views", {}).get(view, {})
        expected = EXPECTED_CHECKPOINTS[checkpoint]
        relative_checkpoint = f"artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2022/{expected['filename']}"
        expected_batches = ((expected_rows + 4095) // 4096) if expected_rows is not None else None
        _require(execution_view.get("checkpoint") == relative_checkpoint
                 and execution_view.get("epoch") == expected["epoch"]
                 and execution_view.get("global_steps") == expected["global_steps"]
                 and execution_view.get("metrics_exact") is True
                 and execution_view.get("metric_differences") == []
                 and execution_view.get("epoch_metrics_status") == "EXACT_EPOCH_METRICS"
                 and (expected_rows is None or (execution_view.get("rows") == expected_rows
                                                and execution_view.get("batches") == expected_batches)),
                 f"Crash4 execution receipt {view} summary differs")
    rows = diagnostic.get("v_row_ids", {})
    _require(rows.get("exact_label_ids_match") is True and rows.get("ordered_as_data_roles_V") is True
             and rows.get("dtype") == "int64" and rows.get("count") == expected_rows,
             "Crash4 V row IDs are not exact direct V-role IDs")
    _require(diagnostic.get("v_row_count") == expected_rows
             and diagnostic.get("manifest_v_count") == expected_rows,
             "Crash4 direct and manifest V row counts differ")


def _read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank/truncated epoch log line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid epoch log JSON at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Epoch log record {line_number} is not an object")
            records.append(record)
    return records


def _validate_epoch_metrics(record, description, validation_cells):
    validation = record.get("validation")
    _require(isinstance(validation, dict) and len(validation.get("cells", [])) == 6,
             f"{description} lacks the six V domain/task cells")
    _require(_is_number(validation.get("macro_auc")) and _is_number(validation.get("macro_logloss")),
             f"{description} has invalid macro V metrics")
    expected = {(cell["domain"], cell["task"]): cell for cell in validation_cells}
    cells = {(cell.get("domain"), cell.get("task")) for cell in validation["cells"]}
    _require(cells == set(expected),
             f"{description} V support cells are duplicated or missing")
    for cell in validation["cells"]:
        _require(isinstance(cell.get("n"), int) and cell["n"] > 0
                 and _is_number(cell.get("auc")) and _is_number(cell.get("logloss")),
                 f"{description} V cell has invalid support or metrics")
        reference = expected[(cell["domain"], cell["task"])]
        _require(all(cell.get(key) == reference[key] for key in ("n", "positive", "negative")),
                 f"{description} V cell support differs from the prepared-data manifest")
    requests = sum(cell["n"] for cell in validation["cells"])
    pooled_counts = [sum(cell["n"] for cell in validation["cells"] if cell["task"] == task)
                     for task in range(2)]
    _require(validation.get("valid_requests") == requests
             and validation.get("natural_auc_defined_requests") == requests
             and validation.get("macro_auc_defined_cells") == 6
             and validation.get("macro_logloss_defined_cells") == 6
             and validation.get("pooled_auc_valid_requests_by_task") == pooled_counts
             and len(validation.get("pooled_auc_by_task", [])) == 2
             and all(_is_number(value) for value in validation["pooled_auc_by_task"])
             and _is_number(validation.get("natural_auc"))
             and _is_number(validation.get("natural_logloss")),
             f"{description} V summary support or metrics are incomplete")
    _require(isinstance(record.get("components"), dict)
             and all(_is_number(value) for value in record["components"].values()),
             f"{description} has missing or non-finite loss components")
    _require(_is_number(record.get("train_loss")) and _is_number(record.get("seconds"))
             and record["seconds"] >= 0, f"{description} has invalid loss or duration")


def _validate_training_log(records, config, train_rows, train_domain_counts, validation_cells):
    cfg = config["models"]["AUTOIFS"]
    search_count = int(cfg["search_epochs"])
    batch_size = int(cfg["batch_size"])
    updates_per_epoch = (int(train_rows) + batch_size - 1) // batch_size
    exposures = int(train_rows) * 2
    tolerance = float(config["metrics"]["tie_tolerance"])
    reuse_count = int(cfg["reuse_max_epochs"])
    stages = [record.get("stage") for record in records]
    _require(stages[0] == "search_setup" and stages[-1] == "reuse_pruning_summary",
             "AutoIFS setup/pruning records are not present at the boundaries")
    search_setup = records[0]
    owners = search_setup.get("optimizer_parameter_ownership", {})
    _require(set(owners) == {"base", "gate"} and owners["base"] and owners["gate"]
             and search_setup.get("base_parameter_count", 0) > 0
             and search_setup.get("gate_parameter_count", 0) > 0,
             "AutoIFS search optimizer ownership is incomplete")
    search = [record for record in records if record.get("stage") == "search"]
    reuse = [record for record in records if record.get("stage") == "reuse"]
    rewind = [record for record in records if record.get("stage") == "search_rewind"]
    reuse_setup = [record for record in records if record.get("stage") == "reuse_setup"]
    pruning = [record for record in records if record.get("stage") == "reuse_pruning_summary"]
    _require(len(search) == search_count and [item.get("epoch") for item in search] == list(range(search_count)),
             "Search epochs are missing, duplicated or out of order")
    _require(len(rewind) == 1 and len(reuse_setup) == 1 and len(pruning) == 1,
             "Rewind, reuse setup or pruning summary is duplicated/missing")
    _require(stages.index("search_rewind") == search_count,
             "Rewind was not captured between search epochs 8 and 9")
    _require(rewind[0].get("rewind_epoch_index") == int(cfg["rewind_epoch_index"])
             and Path(rewind[0].get("checkpoint", "")).resolve()
             == (RUN_DIR / "rewind_epoch9.pt").resolve(), "Rewind log identity/path differs")

    start = float(cfg["temperature_start"])
    end = float(cfg["temperature_end"])
    multiplier = (end / start) ** (1.0 / (search_count - 1)) if search_count > 1 else 1.0
    for index, record in enumerate(search):
        expected_temp = start * multiplier ** index
        _require(record.get("updates") == updates_per_epoch
                 and record.get("valid_label_exposures") == exposures,
                 f"Search epoch {index} update/exposure count differs")
        _require(_is_number(record.get("temperature")) and math.isclose(
            record["temperature"], expected_temp, rel_tol=1e-12, abs_tol=1e-12),
            f"Search epoch {index} temperature schedule differs")
        _validate_epoch_metrics(record, f"Search epoch {index}", validation_cells)
    _require(math.isclose(float(rewind[0].get("temperature_before_final_search_epoch", -1)),
                          float(search[-2]["temperature"]), rel_tol=1e-12, abs_tol=1e-12),
             "Rewind temperature differs from the state entering search epoch 9")

    setup = reuse_setup[0]
    _require(setup.get("rewind_epoch_index") == int(cfg["rewind_epoch_index"])
             and setup.get("fresh_model_from_rewind") is True
             and setup.get("fresh_base_optimizer") is True
             and setup.get("search_optimizer_state_reused") is False
             and setup.get("selector_transplant_equal") is True
             and setup.get("ticket") is True and setup.get("temperature") == end
             and setup.get("gate_frozen") is True,
             "Reuse setup/selector transplant/freeze record is invalid")
    reuse_owners = setup.get("optimizer_parameter_ownership", {})
    _require(set(reuse_owners) == {"base", "gate"} and reuse_owners["base"]
             and reuse_owners["gate"] == [], "Reuse optimizer groups are not base-only")
    _require([item.get("epoch") for item in reuse] == list(range(1, len(reuse) + 1))
             and len(reuse) <= int(cfg["reuse_max_epochs"]), "Reuse epochs are missing or duplicated")
    rewind_index = int(cfg["rewind_epoch_index"])
    expected_stages = (["search_setup"] + ["search"] * rewind_index + ["search_rewind"]
                       + ["search"] * (search_count - rewind_index) + ["reuse_setup"]
                       + ["reuse"] * len(reuse) + ["reuse_pruning_summary"])
    _require(stages == expected_stages, "Epoch log stages are missing, duplicated or out of order")
    best, selected_epoch, bad_epochs, stopped_by_patience = float("inf"), None, 0, False
    for record in reuse:
        epoch = int(record["epoch"])
        _require(record.get("updates") == updates_per_epoch
                 and record.get("valid_label_exposures") == exposures
                 and record.get("ticket") is True and record.get("temperature") == end,
                 f"Reuse epoch {epoch} update/exposure/ticket/temperature differs")
        _validate_epoch_metrics(record, f"Reuse epoch {epoch}", validation_cells)
        score = float(record["validation"]["macro_logloss"])
        if score < best - tolerance:
            best, selected_epoch, bad_epochs = score, epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["reuse_patience"]):
                _require(epoch == len(reuse), "Reuse continued after the first patience stop")
                stopped_by_patience = True
                break
    if not stopped_by_patience:
        _require(len(reuse) == int(cfg["reuse_max_epochs"]), "Reuse log stops before patience or epoch limit")
    _require(selected_epoch == 2 and len(reuse) == 3 and reuse[-1]["epoch"] == 3,
             "V macro LogLoss selection/patience does not select reuse epoch 2 and stop at 3")

    train_rows_by_cell = {(cell["domain"], cell["task"]): int(cell["rows"])
                          for cell in train_domain_counts}
    pruning = pruning[0]
    _require(pruning.get("source") == "T0_union_F" and pruning.get("selector_dimensions") == 4
             and len(pruning.get("cells", [])) == 6, "Actual reuse pruning summary is missing")
    observed_pruning = {(cell.get("domain"), cell.get("task")): cell
                        for cell in pruning["cells"]}
    _require(set(observed_pruning) == set(train_rows_by_cell), "Pruning summary cells are duplicated/missing")
    for key, cell in observed_pruning.items():
        row_count = train_rows_by_cell[key]
        decisions = row_count * 4
        _require(cell.get("rows") == row_count and cell.get("gate_decisions") == decisions
                 and cell.get("removed", -1) + cell.get("retained", -1) == decisions,
                 f"Pruning denominator/counts differ for {key}")
        fraction = cell.get("pruned_fraction")
        _require(_is_number(fraction) and math.isclose(
            fraction, cell["removed"] / decisions, rel_tol=1e-15, abs_tol=1e-15),
            f"Pruning fraction differs for {key}")
    lower_bound = math.fsum(float(record["seconds"]) for record in search + reuse)
    return {"records": records, "search_setup": search_setup,
            "reuse_setup": setup, "search": search, "reuse": reuse, "rewind": rewind[0],
            "pruning": pruning, "selected": reuse[1], "final": reuse[-1],
            "global_steps": updates_per_epoch * (search_count + len(reuse)),
            "rewind_global_steps": updates_per_epoch * int(cfg["rewind_epoch_index"]),
            "logged_epoch_seconds_lower_bound": lower_bound}


def _check_tree_finite(value, description):
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        if not bool(value.isfinite().all()):
            raise ValueError(f"{description} contains a non-finite tensor")
    elif isinstance(value, dict):
        for child in value.values():
            _check_tree_finite(child, description)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_tree_finite(child, description)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{description} contains a non-finite scalar")


def _tree_equal(left, right, torch):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b, torch) for a, b in zip(left, right))
    return left == right


def _validate_rng_state(rng, torch):
    python_state = rng.get("python")
    _require(isinstance(python_state, tuple) and len(python_state) == 3
             and python_state[0] == 3 and isinstance(python_state[1], tuple)
             and len(python_state[1]) == 625
             and all(type(value) is int for value in python_state[1])
             and (python_state[2] is None or _is_number(python_state[2])),
             "Checkpoint Python RNG state has the wrong structure")
    numpy_state = rng.get("numpy")
    _require(isinstance(numpy_state, tuple) and len(numpy_state) == 5
             and numpy_state[0] == "MT19937"
             and getattr(numpy_state[1], "shape", None) == (624,)
             and str(getattr(numpy_state[1], "dtype", "")) == "uint32"
             and type(numpy_state[2]) is int and 0 <= numpy_state[2] <= 624
             and type(numpy_state[3]) is int and numpy_state[3] in (0, 1)
             and _is_number(numpy_state[4]),
             "Checkpoint NumPy RNG state has the wrong structure")
    cpu_state = rng.get("torch_cpu")
    _require(isinstance(cpu_state, torch.Tensor) and cpu_state.dtype == torch.uint8
             and tuple(cpu_state.shape) == (5056,),
             "Checkpoint Torch CPU RNG state has the wrong size or dtype")
    cuda_states = rng.get("torch_cuda")
    _require(isinstance(cuda_states, (list, tuple)) and len(cuda_states) == 2
             and all(isinstance(state, torch.Tensor) and state.dtype == torch.uint8
                     and tuple(state.shape) == (16,) for state in cuda_states),
             "Checkpoint Torch CUDA RNG state has the wrong count, size or dtype")


def _validate_optimizer_state(optimizer, param_names, model_state, *, lr, weight_decay,
                              betas, eps, expected_steps, torch, description):
    groups = optimizer.get("param_groups")
    _require(isinstance(groups, list) and len(groups) == 1,
             f"{description} must contain exactly one Adam parameter group")
    group = groups[0]
    parameter_ids = group.get("params")
    _require(isinstance(parameter_ids, list) and len(parameter_ids) == len(param_names)
             and all(type(value) is int for value in parameter_ids)
             and parameter_ids == list(range(len(param_names))),
             f"{description} parameter IDs/order differ from the recorded ownership")
    _require(group.get("lr") == lr and group.get("weight_decay") == weight_decay
             and tuple(group.get("betas", ())) == tuple(betas) and group.get("eps") == eps
             and group.get("amsgrad") is False and group.get("maximize") is False,
             f"{description} Adam hyperparameters differ from the registered stage")
    parameters = {}
    for parameter_id, name in zip(parameter_ids, param_names):
        parameter = model_state.get(f"network.{name}")
        _require(isinstance(parameter, torch.Tensor),
                 f"{description} owned parameter is absent from model state: {name}")
        parameters[parameter_id] = parameter
    expected_state_ids = {parameter_id for parameter_id, parameter in parameters.items()
                          if parameter.numel() > 0}
    state = optimizer.get("state")
    _require(isinstance(state, dict) and set(state) == expected_state_ids and bool(state),
             f"{description} Adam state entries do not match nonempty owned parameters")
    for parameter_id, name in zip(parameter_ids, param_names):
        if parameter_id not in expected_state_ids:
            continue
        parameter = model_state.get(f"network.{name}")
        item = state[parameter_id]
        _require(isinstance(item, dict) and set(item) == {"step", "exp_avg", "exp_avg_sq"},
                 f"{description} Adam state fields differ for {name}")
        step = item["step"]
        _require(isinstance(step, torch.Tensor) and step.ndim == 0
                 and step.dtype == torch.float32 and float(step.item()) == expected_steps,
                 f"{description} Adam local step count differs for {name}")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = item[moment_name]
            _require(isinstance(moment, torch.Tensor) and moment.dtype == torch.float32
                     and tuple(moment.shape) == tuple(parameter.shape)
                     and bool(torch.isfinite(moment).all()),
                     f"{description} {moment_name} is invalid for {name}")


def _validate_checkpoint_metadata(checkpoint, name, epochs):
    spec = EXPECTED_CHECKPOINTS[name]
    if name == "rewind":
        _require(checkpoint.get("stage") == spec["stage"]
                 and checkpoint.get("epoch_index") == spec["epoch_index"]
                 and checkpoint.get("epoch") is None and checkpoint.get("selected_epoch") is None
                 and checkpoint.get("global_steps") == spec["global_steps"]
                 and checkpoint.get("next_epoch") == spec["next_epoch"]
                 and checkpoint.get("sampler_position") == 0
                 and _is_number(checkpoint.get("temperature")),
                 "Rewind stage/cursor/temperature metadata differs")
        _require(math.isclose(float(checkpoint["temperature"]),
                              float(epochs["rewind"]["temperature_before_final_search_epoch"]),
                              rel_tol=1e-12, abs_tol=1e-12), "Rewind checkpoint temperature differs")
    else:
        _require(checkpoint.get("stage") == spec["stage"]
                 and checkpoint.get("epoch") == spec["epoch"]
                 and checkpoint.get("selected_epoch") == spec["selected_epoch"]
                 and checkpoint.get("global_steps") == spec["global_steps"]
                 and checkpoint.get("next_epoch") == spec["next_epoch"]
                 and checkpoint.get("sampler_position") == 0
                 and checkpoint.get("best_metric") == epochs["selected"]["validation"]["macro_logloss"]
                 and checkpoint.get("model_meta", {}).get("ticket") is True
                 and checkpoint.get("model_meta", {}).get("temp") == 100.0,
                 f"{name} checkpoint stage/cursor/selection/ticket/temp differs")
    return True


def _validate_checkpoint_optimizers(checkpoint, name, run, epochs, torch):
    cfg = run["config"]["models"]["AUTOIFS"]
    fixed = run["config"]["models"]["M3OE_FIXED1"]
    if name == "rewind":
        owners = epochs["search_setup"]["optimizer_parameter_ownership"]
        _require(len(owners["base"]) == 16 and len(owners["gate"]) == 8,
                 "Search optimizer ownership differs from the fixed checkpoint schema")
        expected = ((owners["base"], float(cfg["search_lr"]), float(cfg["search_weight_decay"])),
                    (owners["gate"], float(cfg["search_lr"] * cfg["gate_lr_ratio"]), 0.0))
    else:
        owners = epochs["reuse_setup"]["optimizer_parameter_ownership"]
        _require(len(owners["base"]) == 16 and owners["gate"] == [],
                 "Reuse optimizer ownership differs from the fixed checkpoint schema")
        expected = ((owners["base"], float(cfg["reuse_lr"]), float(cfg["reuse_weight_decay"])),)
    optimizers = checkpoint.get("optimizer_states")
    _require(isinstance(optimizers, list) and len(optimizers) == len(expected),
             f"{name} optimizer count differs from the fixed stage")
    for index, (optimizer, (names, lr, weight_decay)) in enumerate(zip(optimizers, expected)):
        zero_expected = ["embedding.dense_vector"] if index == 0 else []
        actual_zero = [param_name for param_name in names
                       if checkpoint["model_state"].get(f"network.{param_name}") is not None
                       and checkpoint["model_state"][f"network.{param_name}"].numel() == 0]
        _require(actual_zero == zero_expected,
                 f"{name} optimizer {index} zero-element parameter ownership differs")
        _validate_optimizer_state(
            optimizer, names, checkpoint["model_state"], lr=lr, weight_decay=weight_decay,
            betas=fixed["betas"], eps=float(fixed["eps"]),
            expected_steps=EXPECTED_CHECKPOINTS[name]["optimizer_steps"], torch=torch,
            description=f"{name} optimizer {index}")


def _validate_checkpoint(checkpoint, name, run, epochs, path, torch):
    spec = EXPECTED_CHECKPOINTS[name]
    _require(isinstance(checkpoint, dict), f"{name} checkpoint is not a mapping")
    state = checkpoint.get("model_state")
    _require(isinstance(state, dict) and len(state) == EXPECTED_STATE_TENSORS
             and all(isinstance(tensor, torch.Tensor) for tensor in state.values())
             and sum(int(tensor.numel()) for tensor in state.values()) == EXPECTED_STATE_ELEMENTS,
             f"{name} model state schema differs from the audited checkpoint")
    _check_tree_finite(state, f"{name} model state")
    _validate_checkpoint_metadata(checkpoint, name, epochs)
    if name != "rewind":
        _require(isinstance(checkpoint.get("rewind_state"), dict)
                 and isinstance(checkpoint.get("selector_state"), dict),
                 f"{name} checkpoint is missing AutoIFS reuse provenance")
    _validate_checkpoint_optimizers(checkpoint, name, run, epochs, torch)
    rng = checkpoint.get("rng_state")
    _require(isinstance(rng, dict) and {"python", "numpy", "torch_cpu", "torch_cuda"} <= set(rng),
             f"{name} RNG state is incomplete")
    _validate_rng_state(rng, torch)
    embedded = checkpoint.get("run_manifest") or {}
    _require(embedded.get("status") == "TRAINING"
             and _embedded_identity(embedded) == _embedded_identity(run),
             f"{name} embedded run identity differs from the original TRAINING receipt")
    _require(Path(path).resolve() == (RUN_DIR / spec["filename"]).resolve(),
             f"{name} checkpoint path differs")
    return state


def _validate_checkpoint_set(checkpoints, run, epochs, torch):
    states = {}
    for name in ("rewind", "selected", "fixed_final"):
        spec = EXPECTED_CHECKPOINTS[name]
        path = RUN_DIR / spec["filename"]
        _require(path.is_file() and path.stat().st_size == spec["size_bytes"],
                 f"{name} checkpoint file is absent or changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        states[name] = _validate_checkpoint(checkpoint, name, run, epochs, path, torch)
        checkpoints[name] = checkpoint
    reference = states["selected"]
    _require(states["rewind"].keys() == reference.keys()
             and states["fixed_final"].keys() == reference.keys(),
             "Rewind/selected/final model state parameter names differ")
    for key, expected in reference.items():
        _require(states["rewind"][key].shape == expected.shape
                 and states["fixed_final"][key].shape == expected.shape,
                 f"Checkpoint state tensor shape differs for {key}")
        _require(torch.equal(checkpoints["selected"]["rewind_state"][key], states["rewind"][key])
                 and torch.equal(checkpoints["fixed_final"]["rewind_state"][key], states["rewind"][key]),
                 "Reuse checkpoint rewind state differs from the saved search rewind")
    _require(_tree_equal(checkpoints["selected"]["selector_state"],
                         checkpoints["fixed_final"]["selector_state"], torch),
             "Selected/final selector transplant state differs")
    return states


def _validate_stage_log(path, row_ids, batch_size):
    records = _read_jsonl(path)
    expected = []

    def add(view, stage, boundary, batch, begin, ids=None, **extras):
        item = {"view": view, "stage": stage, "boundary": boundary,
                "batch": batch, "row_start_index": begin, **extras}
        if ids is not None:
            item.update(row_count=int(len(ids)), first_row_id=int(ids[0]), last_row_id=int(ids[-1]))
        expected.append(item)

    stages = ("slice_v_ids", "advanced_index_x", "numpy_copy", "as_tensor",
              "domain_model_forward", "detach_cpu_numpy")
    for view in ("selected", "fixed_final"):
        for batch_no, begin in enumerate(range(0, len(row_ids), int(batch_size)), start=1):
            ids = row_ids[begin:begin + int(batch_size)]
            for stage in stages:
                if stage == "slice_v_ids":
                    add(view, stage, "before", batch_no, begin,
                        expected_rows=min(int(batch_size), len(row_ids) - begin))
                else:
                    add(view, stage, "before", batch_no, begin, ids)
                extras = {}
                if stage == "domain_model_forward":
                    extras = {"logits_shape": [len(ids), 2], "logits_dtype": "torch.float32"}
                if stage == "detach_cpu_numpy":
                    extras = {"scores_shape": [len(ids), 2], "scores_dtype": "float32"}
                add(view, stage, "after", batch_no, begin, ids, **extras)
        add(view, "logit_aggregation", "before", None, 0, row_ids)
        add(view, "logit_aggregation", "after", None, 0, row_ids,
            logits_shape=[len(row_ids), 2], logits_dtype="float32")
        add(view, "metric_evaluation", "before", None, 0, row_ids)
        add(view, "metric_evaluation", "after", None, 0, row_ids)
    _require(len(records) == len(expected), "Crash4 stage log has missing/extra operations")
    for index, (actual, wanted) in enumerate(zip(records, expected)):
        for key, value in wanted.items():
            _require(actual.get(key) == value, f"Crash4 stage log differs at event {index}: {key}")
    return len(records)


def _metric_summary(metrics):
    return {key: metrics[key] for key in SUMMARY_FIELDS}


def _validate_crash4_outputs(diagnostic, run, data, records, np, metric_module):
    ids_info = diagnostic["v_row_ids"]
    ids_path = Path(ids_info["file"]).resolve()
    _require(ids_path == (CRASH4_DIR / "diagnostic/v_row_ids.npy").resolve(),
             "Crash4 V row ID artifact path differs")
    row_ids = np.load(ids_path, allow_pickle=False)
    expected_ids = data.roles["V"]
    _require(row_ids.dtype == np.int64 and row_ids.shape == expected_ids.shape
             and np.array_equal(row_ids, expected_ids), "Crash4 row IDs differ from the direct V role")
    _require(ids_info.get("shape") == list(row_ids.shape) and ids_info.get("count") == len(row_ids)
             and ids_info.get("first_row_id") == int(row_ids[0])
             and ids_info.get("last_row_id") == int(row_ids[-1]),
             "Crash4 V row ID summary differs from the stored exact IDs")
    labels, mask, label_ids = data.labels("V", "select")
    _require(np.array_equal(label_ids, row_ids), "V/select labels are not aligned with Crash4 rows")
    domain = data.domain[row_ids]
    metrics = {}
    logits_receipts = {}
    for view, epoch_key in (("selected", "selected"), ("fixed_final", "final")):
        item = diagnostic["views"][view]
        metric_path = Path(item["metrics_path"]).resolve()
        logits_path = Path(item["logits_path"]).resolve()
        _require(metric_path == (CRASH4_DIR / "diagnostic" / f"metrics_{view}.json").resolve()
                 and logits_path == (CRASH4_DIR / "diagnostic" / f"logits_{view}.npy").resolve(),
                 f"Crash4 {view} V artifact path differs")
        stored = _read_json(metric_path)
        logits = np.load(logits_path, allow_pickle=False)
        _require(logits.shape == (len(row_ids), 2) and logits.dtype == np.float32
                 and np.isfinite(logits).all(), f"Crash4 {view} logits are invalid")
        recomputed = metric_module.evaluate(labels, logits, domain, mask)
        defined_auc = [cell for cell in recomputed["cells"] if cell["auc"] is not None]
        auc_requests = sum(cell["n"] for cell in defined_auc)
        recomputed.update({
            "natural_auc": (sum(cell["n"] * cell["auc"] for cell in defined_auc) / auc_requests
                            if auc_requests else None),
            "natural_auc_defined_requests": int(auc_requests),
            "pooled_auc_valid_requests_by_task": [int(mask[:, task].sum()) for task in range(2)],
            "macro_auc_defined_cells": sum(cell["auc"] is not None for cell in recomputed["cells"]),
            "macro_logloss_defined_cells": sum(cell["logloss"] is not None for cell in recomputed["cells"]),
            "method": "AUTOIFS", "dataset": "kuairand", "seed": 2022,
            "role": "V", "checkpoint_view": view,
            "checkpoint_id": f"AUTOIFS/kuairand/seed2022/{view}",
            "row_ids": row_ids.tolist(),
        })
        summary = _metric_summary(recomputed)
        reference_epoch = 2 if view == "selected" else 3
        reference = next(record for record in records
                         if record.get("stage") == "reuse" and record.get("epoch") == reference_epoch)
        _require(summary == reference["validation"] == _metric_summary(stored),
                 f"Crash4 {view} metrics/logits do not exactly match logged reuse epoch {reference_epoch}")
        _require(item.get("metrics") == summary and item.get("metrics_exact") is True,
                 f"Crash4 {view} summary differs from direct V recomputation")
        metrics[view] = summary
        logits_receipts[view] = {"path": str(logits_path), "shape": list(logits.shape),
                                 "dtype": str(logits.dtype)}
    return row_ids, metrics, logits_receipts


def _validate_original_incident(run, lower_bound):
    launch_path = CRASH3_DIR / "AUTOIFS_kuairand_seed2022.json"
    process_path = CRASH3_DIR / "AUTOIFS_kuairand_seed2022_execution.json"
    launch = _read_json(launch_path)
    process = _read_json(process_path)
    incidents = _read_json(INCIDENTS_PATH).get("incidents", [])
    matches = [item for item in incidents if item.get("protocol") == EXPECTED_PROTOCOL
               and item.get("phase") == "formal" and item.get("method") == "AUTOIFS"
               and item.get("dataset") == "kuairand" and item.get("seed") == 2022]
    _require(len(matches) == 1, "Original native crash incident is absent or duplicated")
    incident = matches[0]
    _require(launch.get("status") == "LAUNCH_STARTED"
             and launch.get("cli_command") == "train"
             and launch.get("profile") == "D1"
             and launch.get("run_dir") == str(RUN_DIR.resolve())
             and launch.get("original_cli_argv") == [
                 "--config", "experiments/mdmt_sota_baselines_2526/default_config.json", "train",
                 "--method", "AUTOIFS", "--dataset", "kuairand", "--seed", "2022",
                 "--phase", "formal", "--device", "cuda:1", "--output",
                 "artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2022"],
             "Original launch sidecar is not the preserved TRAINING launch record")
    _require(process.get("exit_code") == 139 and process.get("elapsed_seconds") == 129.106386328
             and process.get("profile") == "D1"
             and process.get("pid") == launch.get("pid")
             and process.get("sidecar") == str((CRASH3_DIR / launch_path.name).relative_to(ROOT)),
             "Original process execution receipt differs from the SIGSEGV run")
    _require(incident.get("status") == "NATIVE_PROCESS_CRASH" and incident.get("exit_code") == 139
             and incident.get("signal") == 11 and incident.get("run_status_before_crash") == "TRAINING"
             and incident.get("canonical_partial_preserved") is True
             and incident.get("pid") == process.get("pid")
             and incident.get("logged_stage_records") == 17
             and incident.get("completed_search_epochs") == 10
             and incident.get("completed_reuse_epochs") == 3
             and incident.get("selected_checkpoint_saved") is True
             and incident.get("fixed_final_checkpoint_saved") is True
             and incident.get("rewind_checkpoint_saved") is True
             and incident.get("counted_as_independent_complete_result") is False
             and incident.get("E_was_read") is False,
             "Original incident record no longer matches the preserved SIGSEGV")
    _require(incident.get("process_elapsed_seconds") == process["elapsed_seconds"]
             and incident.get("logged_epoch_seconds_lower_bound") == lower_bound,
             "Original wall/epoch lower-bound costs differ from their existing receipts")
    _require(run.get("status") == "TRAINING", "Original run status changed before recovery")
    return {"launch_sidecar": str(launch_path.resolve()), "launch_status": launch["status"],
            "original_execution_receipt": str(process_path.resolve()), "exit_code": 139,
            "signal": 11, "original_process_elapsed_seconds": process["elapsed_seconds"],
            "incident_record": str(INCIDENTS_PATH.resolve()),
            "logged_epoch_seconds_lower_bound": lower_bound,
            "native_root_cause": "UNRESOLVED"}


def _build_completion_manifest(run, epochs, metrics, checkpoint_metadata,
                               parameter_count, inference_timing, recovery_seconds,
                               inference_seconds, receipt_path, lower_bound, original_wall):
    _require(_is_number(lower_bound) and _is_number(original_wall)
             and _is_number(recovery_seconds) and _is_number(inference_seconds)
             and int(parameter_count) > 0,
             "Recovery costs/parameter count must be finite and positive where applicable")
    completed = copy.deepcopy(run)
    completed.update({
        "status": "TRAIN_COMPLETE_NOT_EVALUATED",
        "completion_mode": "CHECKPOINT_POSTTRAIN_RECOVERY",
        "selected_checkpoint": str((RUN_DIR / "selected.pt").resolve()),
        "fixed_final_checkpoint": str((RUN_DIR / "fixed_final.pt").resolve()),
        "selected_epoch": 2,
        "final_epoch": 3,
        "checkpoint_metadata": checkpoint_metadata,
        "selected_validation": metrics["selected"],
        "final_validation": metrics["fixed_final"],
        "training_seconds": None,
        "total_elapsed_seconds": None,
        "peak_cuda_bytes": None,
        "logged_epoch_seconds_lower_bound": lower_bound,
        "original_process_elapsed_seconds": original_wall,
        "global_steps": 5486,
        "parameter_count": int(parameter_count),
        "inference_timing": {"checkpoint_view": "selected", **inference_timing,
                              "measurement_kind": "posttraining_recovery"},
        "posttraining_recovery_seconds": float(recovery_seconds),
        "posttraining_inference_seconds": float(inference_seconds),
        "posttraining_recovery": str(Path(receipt_path).resolve()),
        "epochs": copy.deepcopy(epochs["records"]),
    })
    _require(_embedded_identity(completed) == _embedded_identity(run),
             "Completion metadata changed checkpoint-embedded identity fields")
    _require(all(completed[field] is None for field in
                 ("training_seconds", "total_elapsed_seconds", "peak_cuda_bytes")),
             "Unknown original costs must remain null")
    return completed


def _reserve_output_dir(path, original_run_bytes):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    receipt = path / "original_training_run.json"
    with receipt.open("xb") as stream:
        stream.write(original_run_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    _require(receipt.read_bytes() == original_run_bytes,
             "Preserved original TRAINING receipt is not byte-identical")
    return receipt


def _capture_environment(torch, np, device):
    return {"captured_at_utc": _utc_now(), "python": sys.version,
            "python_executable": sys.executable, "platform": sys.platform,
            "torch": torch.__version__, "numpy": np.__version__,
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "torch_threads": {"intra_op": torch.get_num_threads(),
                              "inter_op": torch.get_num_interop_threads()},
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS")},
            "training_or_optimizer_steps": 0, "backward_calls": 0}


def _commit_manifest(path, original_bytes, completed):
    path = Path(path)
    _require(path.read_bytes() == original_bytes,
             "Canonical TRAINING receipt changed during recovery; refusing commit")
    temp = path.with_name("run.posttrain_recovery.tmp")
    payload = json.dumps(completed, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _require(_embedded_identity(_read_json(temp)) == _embedded_identity(
        json.loads(original_bytes.decode("utf-8"))),
        "Atomic candidate manifest changed embedded identity")
    os.replace(temp, path)


def _load_stack():
    helper = runpy.run_path(str(REPORTS / "runtime_crash3_tool.py"),
                            run_name="runtime_posttrain_recovery_helpers")
    np, torch, inherited = helper["_load_stack"]("D1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _require(all(os.environ.get(key) == "1" for key in
                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
             and torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
             "D1 CPU thread limits were not established before model loading")
    return np, torch, inherited


def _direct_script_import_smoke():
    np, torch, _ = _load_stack()
    from experiments.mdmt_relation_basis_v2 import metrics as metric_module
    from experiments.mdmt_sota_baselines_2526 import config, data, evaluation, models, report

    return {"status": "PASS_CPU_DIRECT_SCRIPT_IMPORT_ONLY",
            "python_executable": sys.executable, "repository_root_on_path": str(ROOT) in sys.path,
            "modules_imported": [metric_module.__name__, config.__name__, data.__name__,
                                 evaluation.__name__, models.__name__, report.__name__],
            "torch_threads": {"intra_op": torch.get_num_threads(),
                              "inter_op": torch.get_num_interop_threads()},
            "arrays_or_checkpoints_loaded": False}


def _recover_worker():
    original_bytes = ORIGINAL_RECEIPT.read_bytes()
    _require(RUN_PATH.read_bytes() == original_bytes,
             "Canonical run changed after preservation; recovery is one-use")
    run = json.loads(original_bytes.decode("utf-8"))
    started = time.perf_counter()
    np, torch, inherited_threads = _load_stack()
    from experiments.mdmt_relation_basis_v2 import metrics as metric_module
    from experiments.mdmt_sota_baselines_2526.config import load_config
    from experiments.mdmt_sota_baselines_2526.data import Data
    from experiments.mdmt_sota_baselines_2526.evaluation import load_checkpoint, measure_inference
    from experiments.mdmt_sota_baselines_2526.models import build_model
    from experiments.mdmt_sota_baselines_2526.report import (
        _validate_runtime_io_amendment, verify_current_package_source_snapshot,
        verify_frozen_source_snapshot,
    )

    config = load_config()
    _validate_run_identity(run, {**EXPECTED_RUN, "config": config}, RUN_DIR)
    _require(run.get("data_manifest", {}).get("dataset") == "kuairand",
             "Original run data identity is not KuaiRand")
    amendment, approved_sources, _ = _validate_runtime_io_amendment(SOURCE_PAIR_PATH)
    r2_snapshot = amendment["r2_source_snapshot"]
    verify_current_package_source_snapshot(r2_snapshot)
    verify_frozen_source_snapshot(run["implementation_identity"], r2_snapshot)
    _require(run.get("source_commits") == amendment["source_commits"]
             and run.get("config") == amendment["configuration"],
             "Original run differs from reviewed R2 configuration/source")

    data = Data("kuairand", ROOT / config["prepared_root"])
    _require(run.get("data_manifest") == data.manifest
             and amendment.get("prepared_data_manifests", {}).get("kuairand") == data.manifest,
             "Original and reviewed prepared KuaiRand identities differ")
    train_ids = data.train_ids
    domain_counts = []
    for domain in range(3):
        rows = int((data.domain[train_ids] == domain).sum())
        for task in range(2):
            domain_counts.append({"domain": domain, "task": task, "rows": rows, "n": rows})
    records = _read_jsonl(EPOCH_LOG)
    epochs = _validate_training_log(records, config, len(train_ids), domain_counts,
                                    data.manifest["roles"]["validation"]["cells"])
    _require(epochs["global_steps"] == 5486 and epochs["rewind_global_steps"] == 3798
             and epochs["selected"]["epoch"] == 2 and epochs["final"]["epoch"] == 3,
             "Logged AutoIFS trajectory counters/selection differ from expected state")
    original_incident = _validate_original_incident(run, epochs["logged_epoch_seconds_lower_bound"])

    diagnostic = _read_json(CRASH4_DIAGNOSTIC)
    execution = _read_json(CRASH4_EXECUTION)
    _validate_crash4_gate(diagnostic, execution, run, expected_rows=len(data.roles["V"]))
    _require(diagnostic.get("prepared_data_manifest") == data.manifest,
             "Crash4 prepared-data manifest differs from the current exact data")
    _require(diagnostic.get("run_manifest") == str(RUN_PATH.resolve()),
             "Crash4 diagnostic is not bound to the preserved canonical run")
    stage_count = _validate_stage_log(CRASH4_DIR / "diagnostic/stages.jsonl",
                                      data.roles["V"], config["models"]["AUTOIFS"]["batch_size"])
    row_ids, metrics, logits_receipts = _validate_crash4_outputs(
        diagnostic, run, data, records, np, metric_module)

    model_config = dict(config["models"]["AUTOIFS"], dataset="kuairand", seed=2022)
    source_roots = {name: Path(path) for name, path in config["source_roots"].items()}
    device = torch.device("cuda:1")
    _require(torch.cuda.is_available() and torch.cuda.device_count() > 1,
             "Posttraining inference requires visible cuda:1")
    torch.cuda.set_device(device)
    checkpoints = {}
    _validate_checkpoint_set(checkpoints, run, epochs, torch)
    selected_model = build_model("AUTOIFS", data.schema, model_config, source_roots).to(device)
    selected_checkpoint = load_checkpoint(
        selected_model, RUN_DIR / "selected.pt", expected_method="AUTOIFS",
        expected_dataset="kuairand", expected_seed=2022, expected_config=config)
    _require(selected_checkpoint["stage"] == "reuse" and selected_model.network.ticket is True
             and selected_model.network.temp == 100.0, "Selected model ticket/temp restoration failed")
    final_model = build_model("AUTOIFS", data.schema, model_config, source_roots).to(device)
    final_checkpoint = load_checkpoint(
        final_model, RUN_DIR / "fixed_final.pt", expected_method="AUTOIFS",
        expected_dataset="kuairand", expected_seed=2022, expected_config=config)
    _require(final_checkpoint["stage"] == "reuse" and final_model.network.ticket is True
             and final_model.network.temp == 100.0, "Fixed-final model ticket/temp restoration failed")
    selected_parameter_count = sum(parameter.numel() for parameter in selected_model.parameters())
    final_parameter_count = sum(parameter.numel() for parameter in final_model.parameters())
    _require(selected_parameter_count == final_parameter_count,
             "Selected and fixed-final parameter counts differ")

    timing_rows = row_ids[:min(256, len(row_ids))]
    timing_x = data.x[timing_rows]
    inference_started = time.perf_counter()
    inference_timing = measure_inference(selected_model, timing_x, device=device,
                                         warmup=5, repetitions=25)
    inference_seconds = time.perf_counter() - inference_started
    recovery_seconds = time.perf_counter() - started
    environment = _capture_environment(torch, np, device)
    environment["inherited_thread_environment"] = inherited_threads
    _write_new_json(RECOVERY_ENVIRONMENT, environment)
    checkpoint_metadata = {}
    checkpoint_receipts = {}
    for view, name in (("selected", "selected"), ("fixed_final", "fixed_final")):
        ckpt = checkpoints[name]
        checkpoint_metadata[view] = {key: ckpt[key] for key in
                                     ("stage", "epoch", "global_steps")}
        checkpoint_metadata[view]["model_meta"] = ckpt["model_meta"]
    for name, checkpoint in checkpoints.items():
        state = checkpoint["model_state"]
        checkpoint_receipts[name] = {
            "path": str((RUN_DIR / EXPECTED_CHECKPOINTS[name]["filename"]).resolve()),
            "size_bytes": (RUN_DIR / EXPECTED_CHECKPOINTS[name]["filename"]).stat().st_size,
            "stage": checkpoint.get("stage"), "epoch": checkpoint.get("epoch"),
            "epoch_index": checkpoint.get("epoch_index"),
            "global_steps": checkpoint.get("global_steps"),
            "next_epoch": checkpoint.get("next_epoch"),
            "selected_epoch": checkpoint.get("selected_epoch"),
            "state_tensor_count": len(state),
            "state_element_count": sum(int(value.numel()) for value in state.values()),
            "optimizer_state_count": len(checkpoint["optimizer_states"]),
            "model_meta": checkpoint.get("model_meta"),
            "temperature": checkpoint.get("temperature"),
            "embedded_run_identity_matches_original_training": True,
        }
    recovery = {
        "status": "POSTTRAIN_RECOVERY_COMPLETE",
        "completion_mode": "CHECKPOINT_POSTTRAIN_RECOVERY",
        "completed_at_utc": _utc_now(),
        "protocol": EXPECTED_PROTOCOL, "phase": "formal", "method": "AUTOIFS",
        "dataset": "kuairand", "seed": 2022,
        "original_run_manifest": str(RUN_PATH.resolve()),
        "original_training_receipt": str(ORIGINAL_RECEIPT.resolve()),
        "runtime_environment_sidecar": str(RECOVERY_ENVIRONMENT.resolve()),
        "original_incident": original_incident,
        "crash4_diagnostic": str(CRASH4_DIAGNOSTIC.resolve()),
        "crash4_execution_receipt": str(CRASH4_EXECUTION.resolve()),
        "crash4_status": diagnostic["status"],
        "crash4_v_rows": int(len(row_ids)), "crash4_stage_records_verified": stage_count,
        "crash4_logits": logits_receipts,
        "validation_metrics": metrics,
        "training_epoch_log": str(EPOCH_LOG.resolve()),
        "search_epoch_count": len(epochs["search"]), "reuse_epoch_count": len(epochs["reuse"]),
        "selected_epoch": 2, "final_epoch": 3,
        "rewind_checkpoint": str((RUN_DIR / "rewind_epoch9.pt").resolve()),
        "selected_checkpoint": str((RUN_DIR / "selected.pt").resolve()),
        "fixed_final_checkpoint": str((RUN_DIR / "fixed_final.pt").resolve()),
        "validated_checkpoints": checkpoint_receipts,
        "pruning_summary": epochs["pruning"],
        "global_steps": 5486, "optimizer_steps_during_recovery": 0,
        "backward_calls_during_recovery": 0,
        "logged_epoch_seconds_lower_bound": epochs["logged_epoch_seconds_lower_bound"],
        "original_process_elapsed_seconds": original_incident["original_process_elapsed_seconds"],
        "recovery_seconds": recovery_seconds,
        "inference_seconds": inference_seconds,
        "inference_timing": inference_timing,
        "e_labels_read": False, "s_or_legacy_rows_read": False,
        "native_root_cause": "UNRESOLVED",
        "execution_receipt": str(EXECUTION_RECEIPT.resolve()),
    }
    _write_new_json(RECOVERY_RECEIPT, recovery)
    completed = _build_completion_manifest(
        run, epochs, metrics, checkpoint_metadata, selected_parameter_count,
        inference_timing, recovery_seconds, inference_seconds, RECOVERY_RECEIPT,
        epochs["logged_epoch_seconds_lower_bound"],
        original_incident["original_process_elapsed_seconds"])
    _commit_manifest(RUN_PATH, original_bytes, completed)
    return {"status": recovery["status"], "run_manifest": str(RUN_PATH.resolve()),
            "recovery_receipt": str(RECOVERY_RECEIPT.resolve()),
            "runtime_environment": str(RECOVERY_ENVIRONMENT.resolve()),
            "execution_receipt": str(EXECUTION_RECEIPT.resolve()),
            "training_or_optimizer_steps": 0, "backward_calls": 0,
            "e_labels_read": False}


def _launch_recovery():
    _require(not RECOVERY_DIR.exists(), "A recovery attempt already exists; refusing a second execution")
    original_bytes = RUN_PATH.read_bytes()
    original = json.loads(original_bytes.decode("utf-8"))
    _validate_run_identity(original, EXPECTED_RUN, RUN_DIR)
    original_receipt = _reserve_output_dir(RECOVERY_DIR, original_bytes)
    start_receipt = {"status": "RECOVERY_STARTED", "started_at_utc": _utc_now(),
                     "python_executable": sys.executable, "wrapper_pid": os.getpid(),
                     "worker_exit_code": None,
                     "original_run_manifest": str(RUN_PATH.resolve()),
                     "original_training_receipt": str(original_receipt.resolve()),
                     "scope": "AUTOIFS/KuaiRand/2022 posttraining completion only"}
    _write_new_json(EXECUTION_RECEIPT, start_receipt)
    command = [sys.executable, str(Path(__file__).resolve()), "_recover_worker"]
    worker = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start_receipt["worker_pid"] = worker.pid
    _replace_json(EXECUTION_RECEIPT, start_receipt)
    stdout, stderr = worker.communicate()
    final = {**start_receipt, "status": "POSTTRAIN_RECOVERY_PROCESS_EXITED",
             "completed_at_utc": _utc_now(),
             "worker_exit_code": int(worker.returncode),
             "actual_exit_code": int(worker.returncode),
             "canonical_run_status": _read_json(RUN_PATH).get("status")}
    _replace_json(EXECUTION_RECEIPT, final)
    if stdout:
        print(stdout, end="")
    if stderr:
        sys.stderr.write(stderr)
    if worker.returncode != 0:
        raise SystemExit(worker.returncode)
    _require(final["canonical_run_status"] == "TRAIN_COMPLETE_NOT_EVALUATED",
             "Recovery worker exited 0 without atomically completing run.json")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0


def _assert_no_training_calls():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"train_one", "_train_epoch", "_fit_autoifs", "_make_optimizer", "backward", "step"}
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                called.add(function.id)
            elif isinstance(function, ast.Attribute):
                called.add(function.attr)
    _require(not (called & forbidden), "Recovery tool contains a training/optimizer call")


def _fixture_record(stage, epoch=None, *, score=0.8, seconds=1.0, updates=2,
                    exposures=6, temperature=1.0, ticket=False):
    record = {"stage": stage}
    if stage in ("search", "reuse"):
        record.update({"epoch": epoch, "train_loss": 0.5, "valid_label_exposures": exposures,
                       "updates": updates, "components": {"predictive": 0.5},
                       "temperature": temperature, "seconds": seconds,
                       "validation": {"cells": [{"domain": d, "task": t, "n": 2,
                                                   "auc": 0.6, "logloss": score, "positive": 1,
                                                   "negative": 1, "cell": d * 2 + t}
                                                  for d in range(3) for t in range(2)],
                                      "macro_auc": 0.6, "macro_logloss": score,
                                      "valid_requests": 12, "natural_auc": 0.6,
                                      "natural_auc_defined_requests": 12,
                                      "macro_auc_defined_cells": 6,
                                      "macro_logloss_defined_cells": 6,
                                      "natural_logloss": score,
                                      "pooled_auc_by_task": [0.6, 0.6],
                                      "pooled_auc_valid_requests_by_task": [6, 6]}})
        if stage == "reuse":
            record["ticket"] = ticket
    return record


def _synthetic_selfcheck():
    config = {"metrics": {"tie_tolerance": 1e-12},
              "models": {"AUTOIFS": {"search_epochs": 2, "batch_size": 2,
                                      "reuse_max_epochs": 4, "reuse_patience": 1,
                                      "temperature_start": 1.0, "temperature_end": 4.0,
                                      "rewind_epoch_index": 1}}}
    records = [{"stage": "search_setup", "optimizer_parameter_ownership": {
        "base": ["base"], "gate": ["gate"]}, "base_parameter_count": 2, "gate_parameter_count": 2},
        _fixture_record("search", 0, temperature=1.0),
        {"stage": "search_rewind", "rewind_epoch_index": 1,
         "temperature_before_final_search_epoch": 1.0,
         "checkpoint": str((RUN_DIR / "rewind_epoch9.pt").resolve())},
        _fixture_record("search", 1, temperature=4.0),
        {"stage": "reuse_setup", "rewind_epoch_index": 1,
         "fresh_model_from_rewind": True, "fresh_base_optimizer": True,
         "search_optimizer_state_reused": False, "selector_transplant_equal": True,
         "ticket": True, "temperature": 4.0, "gate_frozen": True,
         "optimizer_parameter_ownership": {"base": ["base"], "gate": []}},
        _fixture_record("reuse", 1, score=0.5, temperature=4.0, ticket=True),
        _fixture_record("reuse", 2, score=0.4, temperature=4.0, ticket=True),
        _fixture_record("reuse", 3, score=0.401, temperature=4.0, ticket=True),
        {"stage": "reuse_pruning_summary", "source": "T0_union_F", "selector_dimensions": 4,
         "cells": [{"domain": d, "task": t, "rows": 1, "gate_decisions": 4,
                    "removed": 1, "retained": 3, "pruned_fraction": 0.25}
                   for d in range(3) for t in range(2)]}]
    counts = [{"domain": d, "task": t, "rows": 1, "n": 1}
              for d in range(3) for t in range(2)]
    validation_cells = [{"domain": d, "task": t, "n": 2, "positive": 1, "negative": 1}
                        for d in range(3) for t in range(2)]
    checked = _validate_training_log(records, config, 3, counts, validation_cells)
    _require(checked["selected"]["epoch"] == 2 and checked["final"]["epoch"] == 3,
             "Synthetic selection/patience check failed")

    def must_reject(action, label):
        try:
            action()
        except (ValueError, FileExistsError):
            return label
        raise AssertionError(f"Synthetic negative check accepted {label}")

    changed = {**EXPECTED_RUN, "status": "TRAINING", "source_commits": EXPECTED_SOURCE_COMMITS,
               "selection_metric": "V_macro_logloss", "output_dir": str(RUN_DIR)}
    must_reject(lambda: _validate_run_identity({**changed, "seed": 2023}, EXPECTED_RUN, RUN_DIR),
                "changed run identity")
    bad_sources = {**changed, "source_commits": {**EXPECTED_SOURCE_COMMITS, "autoifs": "other"}}
    must_reject(lambda: _validate_run_identity(bad_sources, EXPECTED_RUN, RUN_DIR), "source drift")
    bad_log = copy.deepcopy(records)
    bad_log[6]["validation"]["macro_logloss"] = 0.6
    must_reject(lambda: _validate_training_log(bad_log, config, 3, counts, validation_cells),
                "invalid selected checkpoint/patience")
    bad_patience = copy.deepcopy(config)
    bad_patience["models"]["AUTOIFS"]["reuse_patience"] = 2
    must_reject(lambda: _validate_training_log(records, bad_patience, 3, counts, validation_cells),
                "changed patience")
    must_reject(lambda: _validate_training_log(records[:-1], config, 3, counts, validation_cells),
                "missing pruning/checkpoint-selection record")
    checkpoint_epochs = {"rewind": {"temperature_before_final_search_epoch": 1.0},
                         "selected": records[6], "final": records[7]}
    valid_selected_checkpoint = {"stage": "reuse", "epoch": 2, "selected_epoch": 2,
        "global_steps": 5064, "next_epoch": 3, "sampler_position": 0,
        "best_metric": 0.4, "model_meta": {"ticket": True, "temp": 100.0}}
    _validate_checkpoint_metadata(valid_selected_checkpoint, "selected", checkpoint_epochs)
    _validate_checkpoint_metadata({"stage": "search_rewind_start", "epoch": None,
        "epoch_index": 9, "selected_epoch": None, "global_steps": 3798,
        "next_epoch": 9, "sampler_position": 0, "temperature": 1.0},
        "rewind", checkpoint_epochs)
    must_reject(lambda: _validate_checkpoint_metadata(
        {**valid_selected_checkpoint, "global_steps": 5063}, "selected", checkpoint_epochs),
        "invalid checkpoint stage/step metadata")

    fake_config = {"models": {"AUTOIFS": {"batch_size": 4096}}}
    valid_diagnostic = {"status": "CHECKPOINT_V_REPRODUCTION_COMPLETE",
        "configuration": fake_config, "profile": "D1", "device": "cuda:1",
        "torch_threads": {"intra_op": 1, "inter_op": 1}, "batch_size": 4096,
        "v_row_count": 4, "manifest_v_count": 4,
        "partial_run_status_preserved": "TRAINING", "prepared_data_manifest_verified": True,
        "training_or_optimizer_steps": 0, "labels_accessed": ["V/select"], "e_labels_read": False,
        "run_manifest_written": False, "canonical_evaluation_index_written": False,
        "selected_model_retained_during_fixed_final": True, "gdb_managed": True,
        "source_identity": {"source_equivalence_record": str(SOURCE_PAIR_PATH.resolve()),
                            "r2_source_snapshot_verified": True,
                            "current_package_equals_r2": True,
                            "source_commits": EXPECTED_SOURCE_COMMITS,
                            "partial_run_snapshot_verified_as_r2": True},
        "run_identity": {**EXPECTED_RUN, "config": None, "source_commits": EXPECTED_SOURCE_COMMITS,
                         "status": "TRAINING", "output_dir": str(RUN_DIR.resolve()),
                         "selection_metric": "V_macro_logloss"},
        "execution_receipt_path": str(CRASH4_EXECUTION.resolve()),
        "checkpoints": {name: {"filename": spec["filename"],
                                "path": str((RUN_DIR / spec["filename"]).resolve()),
                                "present": True, "size_bytes": spec["size_bytes"],
                                "loaded_by_diagnostic": name != "rewind",
                                **({"strict_state_load": True, "stage": spec["stage"],
                                   "epoch": spec["epoch"], "global_steps": spec["global_steps"],
                                   "state_tensor_count": EXPECTED_STATE_TENSORS,
                                   "state_element_count": EXPECTED_STATE_ELEMENTS,
                                   "ticket": True, "temp": 100.0,
                                   "embedded_run_identity_matches": True,
                                   "embedded_source_snapshot_matches_r2": True}
                                   if name != "rewind" else {})}
                        for name, spec in EXPECTED_CHECKPOINTS.items()},
        "views": {view: {"checkpoint_path": str((RUN_DIR / EXPECTED_CHECKPOINTS[view]["filename"]).resolve()),
                         "status": "EXACT_EPOCH_METRICS",
                         "reference_stage": "reuse", "reference_epoch": EXPECTED_CHECKPOINTS[view]["epoch"],
                         "reference_available": True, "reference_path": str(EPOCH_LOG.resolve()),
                         "metrics_exact": True, "metric_differences": [], "rows": 4}
                  for view in ("selected", "fixed_final")},
        "v_row_ids": {"label_ids_direct_match": True, "order": "ascending",
                      "dtype": "int64", "count": 4, "exact_label_ids_match": True,
                      "ordered_as_data_roles_V": True}}
    good_execution = {"command_invocation_count": 1, "gdb_exit_code": 0,
                      "inferior_outcome": "EXITED_NORMALLY", "inferior_received_sigsegv": False,
                      "classification": "NOT_REPRODUCED",
                      "diagnostic_status": "CHECKPOINT_V_REPRODUCTION_COMPLETE",
                      "profile": "D1", "device": "cuda:1"}
    good_execution["views"] = {view: {
        "checkpoint": f"artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2022/{EXPECTED_CHECKPOINTS[view]['filename']}",
        "epoch": EXPECTED_CHECKPOINTS[view]["epoch"],
        "global_steps": EXPECTED_CHECKPOINTS[view]["global_steps"], "rows": 4, "batches": 1,
        "metrics_exact": True, "metric_differences": [],
        "epoch_metrics_status": "EXACT_EPOCH_METRICS"}
        for view in ("selected", "fixed_final")}
    fake_run = {**EXPECTED_RUN, "config": fake_config, "data_manifest": None}
    _validate_crash4_gate(valid_diagnostic, good_execution, fake_run, expected_rows=4)
    must_reject(lambda: _validate_crash4_gate(valid_diagnostic, {**good_execution, "gdb_exit_code": 139},
                                               fake_run, expected_rows=4),
                "failed Crash4 execution")
    bad_diag = copy.deepcopy(valid_diagnostic)
    bad_diag["views"].pop("fixed_final")
    must_reject(lambda: _validate_crash4_gate(bad_diag, good_execution, fake_run, expected_rows=4),
                "missing Crash4 V view")
    bad_diag = copy.deepcopy(valid_diagnostic)
    bad_diag["checkpoints"]["selected"]["size_bytes"] += 1
    must_reject(lambda: _validate_crash4_gate(bad_diag, good_execution, fake_run, expected_rows=4),
                "nonmatching Crash4 checkpoint evidence")

    synthetic_identity = {**EXPECTED_RUN, "config": {"synthetic": True}, "data_manifest": {"synthetic": True},
                          "source_commits": EXPECTED_SOURCE_COMMITS, "selection_metric": "V_macro_logloss",
                          "environment": {"recorded": True}, "implementation_identity": {"kind": "synthetic"},
                          "output_dir": str(RUN_DIR), "started_at_utc": "2026-10-29T00:00:00+00:00",
                          "status": "TRAINING"}
    selected_metric = {key: None for key in SUMMARY_FIELDS}
    embedded_training_receipt = copy.deepcopy(synthetic_identity)
    completed = _build_completion_manifest(
        synthetic_identity, {"records": records}, {"selected": selected_metric,
        "fixed_final": selected_metric}, {"selected": {}, "fixed_final": {}},
        10, {"batch_size": 4, "device": "cuda:1", "warmup": 5, "repetitions": 25,
             "p50_ms": 1.0, "p95_ms": 1.2}, 1.0, 0.1, RECOVERY_RECEIPT, 2.0, 3.0)
    _require(_embedded_identity(completed) == _embedded_identity(synthetic_identity)
             and embedded_training_receipt.get("status") == "TRAINING"
             and _embedded_identity(completed) == _embedded_identity(embedded_training_receipt)
             and completed["status"] == "TRAIN_COMPLETE_NOT_EVALUATED"
             and completed["training_seconds"] is None
             and completed["total_elapsed_seconds"] is None and completed["peak_cuda_bytes"] is None,
             "Synthetic completion no longer passes checkpoint embedded-identity/null-cost rules")
    must_reject(lambda: _build_completion_manifest(
        synthetic_identity, {"records": records}, {"selected": selected_metric,
        "fixed_final": selected_metric}, {"selected": {}, "fixed_final": {}},
        10, {}, 1, 1, RECOVERY_RECEIPT, float("nan"), 3), "non-finite lower-bound cost")

    with tempfile.TemporaryDirectory() as temporary:
        reserved = Path(temporary) / "one-use"
        reserved.mkdir()
        must_reject(lambda: _reserve_output_dir(reserved, b"{}"), "second recovery execution")
        row_ids = [5, 8, 10]
        events = []

        def event(view, stage, boundary, batch, begin, ids=None, **extras):
            value = {"view": view, "stage": stage, "boundary": boundary,
                     "batch": batch, "row_start_index": begin, **extras}
            if ids is not None:
                value.update(row_count=len(ids), first_row_id=ids[0], last_row_id=ids[-1])
            events.append(value)

        for view in ("selected", "fixed_final"):
            for batch_no, begin in enumerate((0, 2), start=1):
                ids = row_ids[begin:begin + 2]
                for stage in ("slice_v_ids", "advanced_index_x", "numpy_copy", "as_tensor",
                              "domain_model_forward", "detach_cpu_numpy"):
                    if stage == "slice_v_ids":
                        event(view, stage, "before", batch_no, begin, expected_rows=len(ids))
                    else:
                        event(view, stage, "before", batch_no, begin, ids)
                    extras = {}
                    if stage == "domain_model_forward":
                        extras = {"logits_shape": [len(ids), 2], "logits_dtype": "torch.float32"}
                    elif stage == "detach_cpu_numpy":
                        extras = {"scores_shape": [len(ids), 2], "scores_dtype": "float32"}
                    event(view, stage, "after", batch_no, begin, ids, **extras)
            event(view, "logit_aggregation", "before", None, 0, row_ids)
            event(view, "logit_aggregation", "after", None, 0, row_ids,
                  logits_shape=[3, 2], logits_dtype="float32")
            event(view, "metric_evaluation", "before", None, 0, row_ids)
            event(view, "metric_evaluation", "after", None, 0, row_ids)
        stage_path = Path(temporary) / "stages.jsonl"
        stage_path.write_text("".join(json.dumps(record) + "\n" for record in events), encoding="utf-8")
        _require(_validate_stage_log(stage_path, row_ids, 2) == len(events),
                 "Synthetic full stage-log check failed")
        missing_stage_path = Path(temporary) / "stages_missing.jsonl"
        missing_stage_path.write_text("".join(json.dumps(record) + "\n" for record in events[:-1]),
                                      encoding="utf-8")
        must_reject(lambda: _validate_stage_log(missing_stage_path, row_ids, 2),
                    "missing Crash4 stage boundary")
    np, torch, _ = _load_stack()
    model_state = {"network.base0": torch.zeros(2), "network.empty": torch.zeros(0),
                   "network.base2": torch.zeros(1)}

    def adam_fixture(step, lr=1e-4, weight_decay=3e-4):
        state = {}
        for parameter_id in (0, 2):
            parameter = model_state[f"network.{('base0', 'empty', 'base2')[parameter_id]}"]
            state[parameter_id] = {"step": torch.tensor(float(step), dtype=torch.float32),
                                   "exp_avg": torch.zeros_like(parameter),
                                   "exp_avg_sq": torch.zeros_like(parameter)}
        return {"state": state, "param_groups": [{"params": [0, 1, 2], "lr": lr,
                "weight_decay": weight_decay, "betas": (0.9, 0.999), "eps": 1e-8,
                "amsgrad": False, "maximize": False}]}

    adam = adam_fixture(844)
    _validate_optimizer_state(adam, ["base0", "empty", "base2"], model_state,
                              lr=1e-4, weight_decay=3e-4, betas=(0.9, 0.999),
                              eps=1e-8, expected_steps=844, torch=torch,
                              description="synthetic selected")
    bad_adam = copy.deepcopy(adam)
    bad_adam["state"][0]["step"].zero_()
    must_reject(lambda: _validate_optimizer_state(
        bad_adam, ["base0", "empty", "base2"], model_state, lr=1e-4,
        weight_decay=3e-4, betas=(0.9, 0.999), eps=1e-8, expected_steps=844,
        torch=torch, description="synthetic selected"), "optimizer Adam step reset to zero")
    bad_adam = copy.deepcopy(adam)
    del bad_adam["state"][2]
    must_reject(lambda: _validate_optimizer_state(
        bad_adam, ["base0", "empty", "base2"], model_state, lr=1e-4,
        weight_decay=3e-4, betas=(0.9, 0.999), eps=1e-8, expected_steps=844,
        torch=torch, description="synthetic selected"), "missing Adam parameter state")
    bad_adam = copy.deepcopy(adam)
    bad_adam["param_groups"][0]["lr"] = 0.2
    must_reject(lambda: _validate_optimizer_state(
        bad_adam, ["base0", "empty", "base2"], model_state, lr=1e-4,
        weight_decay=3e-4, betas=(0.9, 0.999), eps=1e-8, expected_steps=844,
        torch=torch, description="synthetic selected"), "changed Adam learning rate")
    rng = {"python": __import__("random").Random(2022).getstate(),
           "numpy": np.random.RandomState(2022).get_state(),
           "torch_cpu": torch.zeros(5056, dtype=torch.uint8),
           "torch_cuda": [torch.zeros(16, dtype=torch.uint8) for _ in range(2)]}
    _validate_rng_state(rng, torch)
    for key, invalid in (("python", None), ("numpy", None),
                         ("torch_cpu", torch.empty(0, dtype=torch.uint8)),
                         ("torch_cuda", [])):
        must_reject(lambda key=key, invalid=invalid: _validate_rng_state(
            {**rng, key: invalid}, torch), f"invalid {key} RNG state")
    _assert_no_training_calls()
    return {"status": "PASS_CPU_SYNTHETIC_ONLY", "checks": [
        "exact selection and patience audit", "invalid run/checkpoint/patience rejection",
        "source drift rejection",
        "missing/nonmatching Crash4 V evidence rejection", "one-use output reservation",
        "complete Crash4 stage ordering and missing-boundary rejection",
        "embedded identity preserved with null original cost fields",
        "Adam group/config/state/step validation and empty-parameter handling",
        "Python/NumPy/Torch CPU/CUDA RNG structure and empty-state rejection",
        "no optimizer/backward/training calls", "no real checkpoint/data/GPU access"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selfcheck", help="CPU-only synthetic audit/identity/gate checks")
    commands.add_parser("importcheck", help="direct-script CPU import smoke; opens no run/data artifacts")
    commands.add_parser("recover", help="one reviewed CUDA posttraining recovery for the fixed run")
    commands.add_parser("_recover_worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        result = _synthetic_selfcheck()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "importcheck":
        result = _direct_script_import_smoke()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "recover":
        return _launch_recovery()
    result = _recover_worker()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
