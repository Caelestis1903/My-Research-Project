"""One D1/GDB checkpoint reproduction of the partial AutoIFS V path."""
from __future__ import annotations

import argparse
import io
import json
import os
import runpy
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "research_reports/mdmt_baselines_2526"
CRASH3_TOOL = REPORTS / "runtime_crash3_tool.py"
SOURCE_PAIR = REPORTS / "RUNTIME_IO_SOURCE_EQUIVALENCE.json"
RUN_DIR = ROOT / "artifacts/mdmt_baselines_2526/runs/formal/kuairand/AUTOIFS/seed2022"
EPOCH_LOG = RUN_DIR / "epoch_metrics.jsonl"
REWIND_CHECKPOINT = RUN_DIR / "rewind_epoch9.pt"
CHECKPOINTS = {"selected": (RUN_DIR / "selected.pt", 2, 5064),
               "fixed_final": (RUN_DIR / "fixed_final.pt", 3, 5486)}
EXPECTED_STATE_TENSORS = 26
EXPECTED_STATE_ELEMENTS = 2_436_811


def _load_d1_stack():
    helper = runpy.run_path(str(CRASH3_TOOL), run_name="runtime_crash3_helpers")
    np, torch, inherited = helper["_load_stack"]("D1")
    if inherited is None:
        raise RuntimeError("Crash 3 D1 profile helper did not return its environment receipt")
    if (any(os.environ.get(key) != "1" for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1):
        raise RuntimeError("D1 thread profile was not established before imports/workload")
    sys.path.insert(0, str(ROOT))
    return np, torch, inherited


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _manifest_role_count(manifest, role):
    return sum(int(manifest["roles"][part]["n"]) for part in manifest["role_map"][role])


def _actual_manifest_v_preflight():
    config = json.loads((ROOT / "experiments/mdmt_sota_baselines_2526/default_config.json").read_text(
        encoding="utf-8"))
    manifest_path = ROOT / config["prepared_root"] / "kuairand" / "DATA_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "kuairand":
        raise ValueError("Prepared-data metadata is not KuaiRand")
    count = _manifest_role_count(manifest, "V")
    if count != 250_767:
        raise AssertionError(f"KuaiRand V manifest count changed: expected250767, got{count}")
    return {"path": str(manifest_path.resolve()), "v_role_parts": manifest["role_map"]["V"],
            "v_count": count, "arrays_labels_or_checkpoints_read": False}


def _mark(stream, view, stage, boundary, batch, begin, row_ids=None, **extra):
    item = {"view": view, "stage": stage, "boundary": boundary,
            "batch": batch, "row_start_index": begin, **extra}
    if row_ids is not None:
        item.update(row_count=int(len(row_ids)), first_row_id=int(row_ids[0]),
                    last_row_id=int(row_ids[-1]))
    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    stream.flush()


def _evaluate_marked(np, torch, model, data, ids, labels, mask, *, method, seed, device,
                     batch_size, view, marker_stream):
    from experiments.mdmt_relation_basis_v2.metrics import evaluate

    domain = data.domain[ids]
    logits_parts = []
    model.eval()
    with torch.inference_mode():
        for batch_no, begin in enumerate(range(0, len(ids), int(batch_size)), start=1):
            _mark(marker_stream, view, "slice_v_ids", "before", batch_no, begin,
                  expected_rows=min(int(batch_size), len(ids) - begin))
            batch_ids = ids[begin:begin + int(batch_size)]
            _mark(marker_stream, view, "slice_v_ids", "after", batch_no, begin, batch_ids)
            _mark(marker_stream, view, "advanced_index_x", "before", batch_no, begin, batch_ids)
            x_view = data.x[batch_ids]
            _mark(marker_stream, view, "advanced_index_x", "after", batch_no, begin, batch_ids)
            _mark(marker_stream, view, "numpy_copy", "before", batch_no, begin, batch_ids)
            owned = np.array(x_view)
            _mark(marker_stream, view, "numpy_copy", "after", batch_no, begin, batch_ids)
            _mark(marker_stream, view, "as_tensor", "before", batch_no, begin, batch_ids)
            x = torch.as_tensor(owned, dtype=torch.float32, device=device)
            _mark(marker_stream, view, "as_tensor", "after", batch_no, begin, batch_ids)
            _mark(marker_stream, view, "domain_model_forward", "before", batch_no, begin, batch_ids)
            output = model(x, domain=x[:, -1].long())
            _mark(marker_stream, view, "domain_model_forward", "after", batch_no, begin, batch_ids,
                  logits_shape=list(output.shape), logits_dtype=str(output.dtype))
            _mark(marker_stream, view, "detach_cpu_numpy", "before", batch_no, begin, batch_ids)
            scores = output.detach().cpu().numpy()
            _mark(marker_stream, view, "detach_cpu_numpy", "after", batch_no, begin, batch_ids,
                  scores_shape=list(scores.shape), scores_dtype=str(scores.dtype))
            logits_parts.append(scores)
    _mark(marker_stream, view, "logit_aggregation", "before", None, 0, ids)
    logits = np.concatenate(logits_parts, axis=0) if logits_parts else np.empty((0, 2), np.float32)
    _mark(marker_stream, view, "logit_aggregation", "after", None, 0, ids,
          logits_shape=list(logits.shape), logits_dtype=str(logits.dtype))
    _mark(marker_stream, view, "metric_evaluation", "before", None, 0, ids)
    result = evaluate(labels, logits, domain, mask)
    defined_auc = [cell for cell in result["cells"] if cell["auc"] is not None]
    auc_requests = sum(cell["n"] for cell in defined_auc)
    result.update({
        "natural_auc": (sum(cell["n"] * cell["auc"] for cell in defined_auc) / auc_requests
                        if auc_requests else None),
        "natural_auc_defined_requests": int(auc_requests),
        "pooled_auc_valid_requests_by_task": [int(mask[:, task].sum()) for task in range(2)],
        "macro_auc_defined_cells": sum(cell["auc"] is not None for cell in result["cells"]),
        "macro_logloss_defined_cells": sum(cell["logloss"] is not None for cell in result["cells"]),
    })
    result.update({"method": method, "dataset": data.dataset, "seed": int(seed),
                   "role": "V", "checkpoint_view": view,
                   "checkpoint_id": f"{method}/{data.dataset}/seed{int(seed)}/{view}",
                   "row_ids": ids.tolist()})
    _mark(marker_stream, view, "metric_evaluation", "after", None, 0, ids,
          macro_auc=result["macro_auc"], macro_logloss=result["macro_logloss"])
    return logits, result


def _metric_summary(metrics):
    keys = ("cells", "macro_auc", "macro_logloss", "natural_auc",
            "natural_auc_defined_requests", "macro_auc_defined_cells",
            "macro_logloss_defined_cells", "natural_logloss",
            "pooled_auc_by_task", "pooled_auc_valid_requests_by_task", "valid_requests")
    return {key: metrics[key] for key in keys}


def _synthetic_selfcheck():
    np, torch, _ = _load_d1_stack()
    from experiments.mdmt_sota_baselines_2526.evaluation import evaluate_role

    class SyntheticData:
        dataset = "kuairand"

        def __init__(self):
            rows = np.arange(36, dtype=np.int64)
            domain = rows % 3
            self.x = np.column_stack(((rows % 7) / 6.0, ((rows // 3) % 5) / 4.0,
                                      domain)).astype(np.float32)
            self.domain = domain
            self.roles = {"V": rows}
            self.y = np.column_stack((rows % 2, (rows // 2) % 2)).astype(np.float32)

        def labels(self, role, purpose, row_ids=None):
            if role != "V" or purpose != "select":
                raise PermissionError("Synthetic fixture permits only V/select labels")
            ids = self.roles["V"] if row_ids is None else np.asarray(row_ids, dtype=np.int64)
            return self.y[ids], np.ones((len(ids), 2), dtype=bool), ids.copy()

    class SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.outputs = []

        def forward(self, x, domain=None):
            if domain is None:
                raise ValueError("Synthetic evaluator must pass domain IDs")
            result = torch.stack((x[:, 0] * 1.2 + domain.float() * 0.05 - 0.2,
                                  x[:, 1] * -0.7 + domain.float() * 0.1 + 0.15), dim=1)
            self.outputs.append(result.detach().cpu().numpy())
            return result

    data = SyntheticData()
    model = SyntheticModel()
    ref = evaluate_role(model, data, "V", method="AUTOIFS", seed=2022, device="cpu", batch_size=7)
    ref_logits = np.concatenate(model.outputs, axis=0)
    model.outputs.clear()
    markers = io.StringIO()
    ids = data.roles["V"]
    labels, mask, returned_ids = data.labels("V", "select")
    logits, got = _evaluate_marked(np, torch, model, data, ids, labels, mask, method="AUTOIFS", seed=2022,
                                   device="cpu", batch_size=7, view="selected", marker_stream=markers)
    events = [json.loads(line) for line in markers.getvalue().splitlines()]
    expected_stages = {"slice_v_ids", "advanced_index_x", "numpy_copy", "as_tensor",
                       "domain_model_forward", "detach_cpu_numpy", "logit_aggregation", "metric_evaluation"}
    per_batch_order = [(stage, boundary) for stage in
                       ("slice_v_ids", "advanced_index_x", "numpy_copy", "as_tensor",
                        "domain_model_forward", "detach_cpu_numpy")
                       for boundary in ("before", "after")]
    batch_events = [event for event in events if event["batch"] is not None]
    marker_order = all([(event["stage"], event["boundary"]) for event in batch_events
                        if event["batch"] == batch] == per_batch_order for batch in range(1, 7))
    if (not np.array_equal(returned_ids, ids) or not np.array_equal(logits, ref_logits)
            or got != ref or not expected_stages.issubset({event["stage"] for event in events})
            or not marker_order
            or sum(event["stage"] == "as_tensor" and event["boundary"] == "before" for event in events) != 6):
        raise AssertionError("Synthetic marked evaluator differs from the existing evaluator/markers")
    manifest_preflight = _actual_manifest_v_preflight()
    return {"status": "PASS_CPU_SYNTHETIC_ONLY", "rows": len(ids), "batch_size": 7,
            "batches": 6, "logits_equal": True, "metrics_equal": True,
            "marker_stage_coverage": sorted(expected_stages), "marker_events": len(events),
            "marker_order_per_batch": marker_order,
            "manifest_v_preflight": manifest_preflight}


def _read_epoch_reference(path, epoch):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("stage") == "reuse" and record.get("epoch") == epoch:
            return record
    return None


def _diagnose(output_dir: Path):
    np, torch, inherited = _load_d1_stack()
    from experiments.mdmt_sota_baselines_2526.config import load_config
    from experiments.mdmt_sota_baselines_2526.data import Data
    from experiments.mdmt_sota_baselines_2526.evaluation import load_checkpoint
    from experiments.mdmt_sota_baselines_2526.models import build_model
    from experiments.mdmt_sota_baselines_2526.report import (
        _verify_checkpoint_run, verify_current_package_source_snapshot, verify_frozen_source_snapshot,
    )
    from experiments.mdmt_sota_baselines_2526.training import _metric_summary as training_metric_summary
    from experiments.mdmt_sota_baselines_2526.training import _model_config, _source_roots

    output_dir = output_dir.expanduser().resolve()
    if not output_dir.is_relative_to(REPORTS.resolve()):
        raise ValueError("Diagnostic output must stay under research_reports/mdmt_baselines_2526")
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = output_dir / "diagnostic.json"
    stage_path = output_dir / "stages.jsonl"
    evidence = {"status": "RUNNING", "diagnostic": "Crash 4 AutoIFS selected/fixed-final V reproduction",
                "run_manifest": str((RUN_DIR / "run.json").resolve()), "profile": "D1",
                "device": "cuda:1", "labels_accessed": [], "checkpoints": {}, "views": {},
                "gdb_transcript_path": str((output_dir.parent / "gdb_transcript.txt").resolve()),
                "execution_receipt_path": str((output_dir.parent / "gdb_execution.json").resolve())}
    _write_json(evidence_path, evidence)
    started = time.perf_counter()
    try:
        config = load_config()
        source_pair = json.loads(SOURCE_PAIR.read_text(encoding="utf-8"))
        if source_pair.get("status") != "REVIEW_REPLAY_COMPLETE":
            raise ValueError("Approved R2 source-equivalence receipt is not complete")
        r2_source = source_pair["r2_source_snapshot"]
        if (source_pair.get("configuration") != config
                or source_pair.get("source_commits") != config["source_commits"]):
            raise ValueError("Approved R2 config/source commits differ from the active config")
        verify_current_package_source_snapshot(r2_source)
        run_path = RUN_DIR / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        expected_run = {"protocol": config["protocol"], "phase": "formal", "method": "AUTOIFS",
                        "dataset": "kuairand", "seed": 2022, "config": config,
                        "source_commits": config["source_commits"]}
        for key, expected in expected_run.items():
            if run.get(key) != expected:
                raise ValueError(f"Partial run {key} differs from the registered formal identity")
        if (run.get("source_commits") != source_pair["source_commits"]
                or run.get("selection_metric") != config["metrics"]["checkpoint_selection"]):
            raise ValueError("Partial run source/selection identity differs from the approved protocol")
        if run.get("status") != "TRAINING" or Path(run.get("output_dir", "")).resolve() != RUN_DIR.resolve():
            raise ValueError("The original partial run identity/status/path changed")
        verify_frozen_source_snapshot(run.get("implementation_identity"), r2_source)
        data = Data("kuairand", ROOT / config["prepared_root"])
        if (run.get("data_manifest") != data.manifest
                or source_pair.get("prepared_data_manifests", {}).get("kuairand") != data.manifest):
            raise ValueError("Prepared KuaiRand manifest differs from the partial run")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("The checkpoint reproduction requires visible cuda:1")
        device = torch.device("cuda:1")
        torch.cuda.set_device(device)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("D1 PyTorch thread counts must remain1/1")

        model_config = config["models"]["AUTOIFS"]
        batch_size = int(model_config.get("eval_batch_size", model_config["batch_size"]))
        if batch_size != 4096:
            raise ValueError("The registered AutoIFS V evaluation batch size must remain4096")
        epoch_records = {}
        for epoch in (2, 3):
            record = _read_epoch_reference(EPOCH_LOG, epoch)
            if record is None:
                epoch_records[epoch] = None
            elif record.get("ticket") is not True or record.get("stage") != "reuse":
                raise ValueError(f"Reuse epoch{epoch} log identity is invalid")
            else:
                epoch_records[epoch] = record

        ids = data.roles["V"]
        manifest_v_count = _manifest_role_count(data.manifest, "V")
        if len(ids) != manifest_v_count:
            raise ValueError("V labels/row coverage differs from the prepared V role")
        labels, mask, label_ids = data.labels("V", "select")
        if not np.array_equal(label_ids, ids):
            raise ValueError("V labels returned a row order different from the prepared V role")
        np.save(output_dir / "v_row_ids.npy", ids, allow_pickle=False)
        evidence.update({"status": "RUNNING", "configuration": config,
                         "source_identity": {"source_equivalence_record": str(SOURCE_PAIR.resolve()),
                                             "r2_source_snapshot_verified": True,
                                             "partial_run_snapshot_verified_as_r2": True,
                                             "current_package_equals_r2": True,
                                             "source_commits": run["source_commits"]},
                         "run_identity": {"run_manifest": str(run_path.resolve()),
                                          "protocol": run["protocol"], "phase": run["phase"],
                                          "method": run["method"], "dataset": run["dataset"],
                                          "seed": int(run["seed"]), "status": run["status"],
                                          "output_dir": run["output_dir"],
                                          "selection_metric": run["selection_metric"]},
                         "prepared_data_manifest_verified": True,
                         "prepared_data_manifest": data.manifest,
                         "partial_run_status_preserved": run["status"],
                         "labels_accessed": ["V/select"], "v_row_count": int(len(ids)),
                         "manifest_v_count": manifest_v_count,
                         "v_row_ids": {"file": str((output_dir / "v_row_ids.npy").resolve()),
                                       "shape": list(ids.shape), "dtype": str(ids.dtype),
                                       "count": int(len(ids)), "first_row_id": int(ids[0]),
                                       "last_row_id": int(ids[-1]),
                                       "ordered_as_data_roles_V": True,
                                       "exact_label_ids_match": True},
                         "batch_size": batch_size, "tail_batch_rows": int(len(ids) % batch_size or batch_size),
                         "inherited_thread_environment": inherited,
                "effective_thread_environment": {key: os.environ.get(key) for key in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                         "torch_threads": {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()},
                         "gdb_managed": True, "training_or_optimizer_steps": 0,
                         "checkpoints": {"rewind": {"path": str(REWIND_CHECKPOINT.resolve()),
                                                     "filename": REWIND_CHECKPOINT.name,
                                                     "present": REWIND_CHECKPOINT.is_file(),
                                                     "size_bytes": REWIND_CHECKPOINT.stat().st_size
                                                     if REWIND_CHECKPOINT.is_file() else None,
                                                     "loaded_by_diagnostic": False}},
                         "views": {}})
        if not evidence["checkpoints"]["rewind"]["present"]:
            raise FileNotFoundError(f"Preserved rewind checkpoint is missing: {REWIND_CHECKPOINT}")
        _write_json(evidence_path, evidence)

        with stage_path.open("x", encoding="utf-8") as marker_stream:
            selected_model = None
            for view, (checkpoint_path, expected_epoch, expected_steps) in CHECKPOINTS.items():
                model = build_model("AUTOIFS", data.schema,
                                    _model_config(config, "AUTOIFS", "kuairand", 2022),
                                    _source_roots(config)).to(device)
                if view == "selected":
                    selected_model = model
                elif selected_model is None or selected_model is model:
                    raise AssertionError("Selected checkpoint model must remain resident as a distinct model")
                checkpoint = load_checkpoint(model, checkpoint_path, expected_method="AUTOIFS",
                                             expected_dataset="kuairand", expected_seed=2022,
                                             expected_config=config)
                _verify_checkpoint_run(run, checkpoint, run_path)
                verify_frozen_source_snapshot(checkpoint["run_manifest"].get("implementation_identity"), r2_source)
                state = checkpoint["model_state"]
                state_tensors = len(state)
                state_elements = sum(int(value.numel()) for value in state.values())
                meta = checkpoint.get("model_meta", {})
                if (checkpoint.get("stage") != "reuse" or checkpoint.get("epoch") != expected_epoch
                        or checkpoint.get("global_steps") != expected_steps
                        or meta.get("ticket") is not True or float(meta.get("temp", -1)) != 100.0
                        or model.network.ticket is not True or float(model.network.temp) != 100.0
                        or state_tensors != EXPECTED_STATE_TENSORS or state_elements != EXPECTED_STATE_ELEMENTS):
                    raise ValueError(f"{view} checkpoint stage/state/ticket/temp identity is unexpected")
                reference = epoch_records[expected_epoch]
                if reference is not None and float(meta["temp"]) != float(reference["temperature"]):
                    raise ValueError(f"{view} checkpoint temperature differs from reuse epoch log")
                checkpoint_receipt = {"path": str(checkpoint_path.resolve()),
                                      "filename": checkpoint_path.name,
                                      "size_bytes": checkpoint_path.stat().st_size,
                                      "strict_state_load": True, "stage": checkpoint["stage"],
                                      "epoch": int(checkpoint["epoch"]),
                                      "global_steps": int(checkpoint["global_steps"]),
                                      "state_tensor_count": state_tensors,
                                      "state_element_count": state_elements,
                                      "ticket": bool(model.network.ticket), "temp": float(model.network.temp),
                                      "embedded_run_identity_matches": True,
                                      "embedded_source_snapshot_matches_r2": True}
                evidence["checkpoints"][view] = checkpoint_receipt
                _write_json(evidence_path, evidence)
                started_view = time.perf_counter()
                logits, result = _evaluate_marked(np, torch, model, data, ids, labels, mask,
                                                  method="AUTOIFS", seed=2022, device=device,
                                                  batch_size=batch_size, view=view,
                                                  marker_stream=marker_stream)
                elapsed = time.perf_counter() - started_view
                logits_path = output_dir / f"logits_{view}.npy"
                np.save(logits_path, logits, allow_pickle=False)
                persisted = {key: value for key, value in result.items() if key != "row_ids"}
                metrics_path = output_dir / f"metrics_{view}.json"
                _write_json(metrics_path, persisted)
                summary = training_metric_summary(result)
                ref_summary = reference.get("validation") if reference is not None else None
                exact = summary == ref_summary if ref_summary is not None else None
                evidence["views"][view] = {
                    "status": "EXACT_EPOCH_METRICS" if exact is True else
                             "METRICS_DIFFER" if exact is False else "REFERENCE_UNAVAILABLE",
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_filename": checkpoint_path.name,
                    "reference_stage": "reuse", "reference_epoch": expected_epoch,
                    "reference_available": reference is not None,
                    "reference_path": str(EPOCH_LOG.resolve()), "metrics_exact": exact,
                    "metric_differences": ([key for key in summary if summary[key] != ref_summary.get(key)]
                                            if ref_summary is not None else None),
                    "rows": int(len(ids)), "batches": (len(ids) + batch_size - 1) // batch_size,
                    "elapsed_seconds": elapsed, "metrics": summary,
                    "metrics_path": str(metrics_path.resolve()), "logits_path": str(logits_path.resolve())}
                _write_json(evidence_path, evidence)
                if view == "selected":
                    if selected_model is None or selected_model is not model:
                        raise AssertionError("Selected model was not retained during fixed-final evaluation")

        failed = [view for view, result in evidence["views"].items()
                  if result["status"] == "METRICS_DIFFER"]
        evidence.update(status="METRIC_PARITY_MISMATCH" if failed else
                        "CHECKPOINT_V_REPRODUCTION_COMPLETE",
                        elapsed_seconds=time.perf_counter() - started,
                        selected_model_retained_during_fixed_final=True,
                        run_manifest_written=False, canonical_evaluation_index_written=False,
                        e_labels_read=False)
        _write_json(evidence_path, evidence)
        if failed:
            raise ValueError(f"V metric parity differs for checkpoint views: {failed}")
        return evidence
    except Exception as exc:
        status = "METRIC_PARITY_MISMATCH" if evidence.get("status") == "METRIC_PARITY_MISMATCH" else "FAILED"
        evidence.update(status=status, exception_type=type(exc).__name__, exception=str(exc),
                        elapsed_seconds=time.perf_counter() - started,
                        run_manifest_written=False, canonical_evaluation_index_written=False)
        _write_json(evidence_path, evidence)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selfcheck", help="CPU-only synthetic evaluator/logit/marker parity")
    diagnose = commands.add_parser("diagnose", help="one D1/GDB partial-checkpoint V reproduction")
    diagnose.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        result = _synthetic_selfcheck()
        print(json.dumps(result, indent=2))
        return 0
    result = _diagnose(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
