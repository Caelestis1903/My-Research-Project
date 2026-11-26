"""Five-seed paired statistics and numeric stage decisions."""

import csv
import io
import json
import math
from collections import defaultdict
from statistics import mean, stdev

from .selection import (
    CELLS, CELL_LIMIT, DATASETS, EXTENSIONS, FAMILIES, PARENTS, READOUTS,
    RESOLUTION, SEEDS, STAGES, assert_open, choose_cstar, combination_gate, control_gates,
    extension_gates, file_hash, gaussian_paused_jobs, grid_selection, load, read_plan, read_queue,
    short_kernel, stage_decision, write_bytes, write_json,
)

T95 = 2.7764451051977987
PROGRESS = ("selected", "E10", "E25", "E50")
METRICS = (
    "macro_auc", "macro_logloss", "traffic_auc", "traffic_logloss", "pooled_auc", "pooled_logloss",
    "worst_cell_auc",
    *(f"{cell}_{kind}" for cell in CELLS for kind in ("auc", "logloss")),
    *(f"domain{domain}_macro_{kind}" for domain in range(3) for kind in ("auc", "logloss")),
    *(f"task{task}_macro_{kind}" for task in range(2) for kind in ("auc", "logloss")),
    *(f"task{task}_pooled_{kind}" for task in range(2) for kind in ("auc", "logloss")),
)
ALIASES = {"traffic_auc": "sample_weighted_cell_auc",
           "traffic_logloss": "sample_weighted_cell_logloss", "pooled_auc": "pooled_task_mean_auc"}


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=(), *, immutable=False):
    fields = list(dict.fromkeys((*fields, *(name for row in rows for name in row))))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return write_bytes(path, buffer.getvalue().encode(), immutable=immutable)


def number(value):
    if value in ("", None):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite metric: {value}")
    return result


def field(row, *names):
    return next((row[name] for name in names if row.get(name) not in (None, "")), None)


def _numeric_metrics(row):
    values = {metric: number(field(row, metric, ALIASES.get(metric, metric))) for metric in METRICS}
    for cell in CELLS:
        for name, alias in (("count", "n"), ("positive", "n_positive"), ("negative", "n_negative")):
            values[f"{cell}_{name}"] = number(row.get(f"{cell}_{name}"))
            if cell == row.get("cell"):
                values[f"{cell}_{name}"] = number(field(row, name, alias, f"{cell}_{name}"))
    cell = row.get("cell")
    if cell in CELLS:
        values[f"{cell}_auc"] = number(field(row, "cell_auc", "auc", f"{cell}_auc"))
        values[f"{cell}_logloss"] = number(field(row, "cell_logloss", "logloss", f"{cell}_logloss"))
    return values


def normalize_endpoints(rows, split):
    """Reduce cell-long records to one row per seed/model/readout/endpoint."""
    grouped = {}
    seen_cells = defaultdict(set)
    for raw in rows:
        if raw.get("split") != split:
            raise ValueError(f"unexpected split in {split} input: {raw.get('split')}")
        fields = {
            "dataset": raw["dataset"], "seed": int(raw["seed"]),
            "arm": field(raw, "config_id", "arm"), "readout": field(raw, "readout", "estimator", "model_kind"),
            "endpoint": field(raw, "endpoint", "checkpoint"),
            "epoch": int(field(raw, "selected_epoch", "epoch")),
            "model_hash": raw["model_hash"], "data_hash": raw["data_hash"],
            "source_hash": raw["source_hash"], "config_hash": raw["config_hash"],
            "group_hash": field(raw, "group_hash", "scope_hash"),
            "scope_hash": field(raw, "scope_hash", "group_hash"),
            "buffers_hash": field(raw, "buffers_hash"),
            "logical_run_id": field(raw, "logical_run_id", "job_key"),
            "attempt_id": field(raw, "attempt_id", "physical_attempt"),
            "split": split, "source_type": field(raw, "source_type", "source"),
        }
        key = tuple(fields[name] for name in ("dataset", "seed", "arm", "readout", "endpoint"))
        cell = raw.get("cell") or None
        if cell in seen_cells[key] or (cell is None and seen_cells[key]):
            raise ValueError(f"duplicate endpoint: {key}/{cell}")
        seen_cells[key].add(cell)
        values = {k: v for k, v in _numeric_metrics(raw).items() if v is not None}
        target = grouped.setdefault(key, {})
        for name, value in (fields | values).items():
            if name in target and target[name] != value:
                raise ValueError(f"conflicting endpoint values: {key}/{name}")
            target[name] = value
    for key, cells in seen_cells.items():
        if cells != {None} and cells != set(CELLS):
            raise ValueError(f"incomplete cell rows: {key}")
    return grouped


def _verify_curves(directory, group, specs, paused):
    path = directory / "curves.csv"
    curves = defaultdict(dict)
    for row in csv_rows(path):
        job = field(row, "logical_run_id", "job_key")
        if job in paused:
            continue
        if row.get("split") != "validation":
            raise ValueError("Validation curves contain an unexpected split")
        if row.get("metric_name") not in (None, "", "macro_auc") or row.get("cell") in CELLS:
            continue
        key = (job, row["readout"])
        epoch = int(row["epoch"])
        score = number(field(row, "macro_auc", "value"))
        identity = field(row, "model_hash", "state_hash")
        if epoch in curves[key] or score is None or identity is None:
            raise ValueError(f"duplicate or unsupported Validation epoch: {key}/E{epoch}")
        curves[key][epoch] = (score, identity)
    for job, spec in specs.items():
        for readout in spec.readouts:
            curve = curves[job, readout]
            if set(curve) != set(range(1, 51)):
                raise ValueError(f"incomplete E1–E50 Validation curve: {job}/{readout}")
            winner = min(curve, key=lambda epoch: (-curve[epoch][0], epoch))
            for endpoint in PROGRESS:
                row = group[spec.dataset, spec.seed, spec.arm, readout, endpoint]
                epoch = winner if endpoint == "selected" else int(endpoint[1:])
                if row["epoch"] != epoch or (row.get("macro_auc"), row["model_hash"]) != curve[epoch]:
                    raise ValueError(f"endpoint and strict earliest curve disagree: {job}/{readout}/{endpoint}")
    return path


def accepted_stage(root, stage, *, conditional_pause=None):
    from .protocol import RunSpec

    queue = read_queue(root, stage)
    directory = root / stage
    files = [directory / "QUEUE.json"]
    if any(reason["status"] == "TECHNICAL_PAUSED" for reason in queue["trigger_reasons"].values()):
        raise ValueError(f"{stage} has a triggered arm paused for source identity")
    if not queue["jobs"] and not queue["reuse_slots"]:
        return {}, files
    receipt_path = directory / "RUN_RECEIPT.json"
    receipt = load(receipt_path)
    if conditional_pause is None:
        expected_status = "COMPLETE"
        paused = set()
    else:
        expected_status = "TECHNICAL_PAUSED"
        paused = gaussian_paused_jobs(queue, conditional_pause)
    if (receipt["status"] != expected_status or receipt["queue_hash"] != queue["queue_hash"]
            or receipt.get("conditional_pause") != conditional_pause):
        raise ValueError(f"{stage} has no accepted receipt for the current queue")
    specs = {spec.job_key: spec for spec in map(RunSpec.from_dict, queue["jobs"])}
    specs = {key: spec for key, spec in specs.items() if key not in paused}
    if (len(receipt["jobs"]) != len(specs)
            or {row["job_key"] for row in receipt["jobs"] if row["status"] == "COMPLETE"} != set(specs)):
        raise ValueError(f"run receipt is missing an accepted queue slot: {stage}")
    runs_path = directory / "runs.csv"
    attempts = defaultdict(list)
    physical = defaultdict(set)
    for row in csv_rows(runs_path):
        key = field(row, "logical_run_id", "job_key")
        if key in paused:
            continue
        physical[key].add(field(row, "attempt_id", "physical_attempt"))
        if field(row, "state", "status") == "COMPLETE":
            attempts[key].append(row)
    if set(attempts) != set(specs) or any(len(rows) != 1 for rows in attempts.values()):
        raise ValueError(f"accepted runs do not match the queue: {stage}")
    endpoints_path = directory / "endpoints.csv"
    group = normalize_endpoints((row for row in csv_rows(endpoints_path)
                                 if field(row, "logical_run_id", "job_key") not in paused), "validation")
    expected = { (spec.dataset, spec.seed, spec.arm, readout, endpoint): spec
                for spec in specs.values() for readout in spec.readouts for endpoint in PROGRESS }
    if set(group) != set(expected):
        raise ValueError(f"Validation endpoint set does not match the planned slots: {stage}")
    for key, spec in expected.items():
        row = group[key]
        matched = {
            "logical_run_id": spec.job_key, "data_hash": spec.data_hash, "config_hash": spec.config_hash,
            "source_hash": spec.source_hash, "group_hash": spec.group_hash,
        }
        if (any(row[name] != value or not value for name, value in matched.items())
                or row["attempt_id"] not in physical[spec.job_key]):
            raise ValueError(f"endpoint identity differs from its accepted attempt: {key}")
        if not row["model_hash"] or any(row.get(metric) is None for metric in (
            "macro_auc", "macro_logloss", *(f"{cell}_auc" for cell in CELLS),
            *(f"{cell}_logloss" for cell in CELLS),
        )):
            raise ValueError(f"missing full-population Validation metrics: {key}")
    return group, files + [receipt_path, runs_path, endpoints_path, _verify_curves(directory, group, specs, paused)]


class EndpointTable:
    def __init__(self, rows):
        self.rows = rows

    def get(self, arm, dataset, seed, readout, endpoint="selected"):
        return self.rows.get((dataset, seed, arm, readout, endpoint))

    def average(self, arm, dataset, readout, metric="macro_auc"):
        values = [(self.get(arm, dataset, seed, readout) or {}).get(metric) for seed in SEEDS]
        return mean(values) if all(value is not None for value in values) else None

    def diff(self, arm, parent, dataset, readout, metric="macro_auc", endpoint="selected"):
        result = contrast(self, dataset, endpoint, metric, [(arm, readout, 1), (parent, readout, -1)])
        return result["mean_delta"]


def paired_stats(values):
    present = [value for value in values if value is not None]
    complete = len(present) == len(SEEDS)
    center = mean(present) if complete else None
    spread = stdev(present) if complete else None
    margin = T95 * spread / math.sqrt(len(SEEDS)) if complete else None
    return {
        "n": len(present), "mean_delta": center, "sd_delta": spread,
        "ci95_low": center - margin if complete else None,
        "ci95_high": center + margin if complete else None,
        "positive": sum(value > 0 for value in present), "zero": sum(value == 0 for value in present),
        "negative": sum(value < 0 for value in present),
    }


def contrast(table, dataset, endpoint, metric, terms, values=None):
    deltas = []
    for seed in SEEDS:
        rows = [table.get(arm, dataset, seed, readout, endpoint) for arm, readout, _ in terms]
        same = all(rows) and len({row["data_hash"] for row in rows}) == 1
        operands = [(row or {}).get(metric) for row in rows]
        delta = (math.fsum(value * term[2] for value, term in zip(operands, terms))
                 if same and all(value is not None for value in operands) else None)
        deltas.append(delta)
        if values is not None:
            record = {"seed": seed, "delta": delta, "identity_match": bool(same),
                      "candidate_value": operands[0], "reference_value": operands[1],
                      "operand_values": json.dumps(operands)}
            cell = metric[:4]
            if cell in CELLS and rows[1] is not None:
                count, positive, negative = (rows[1].get(f"{cell}_{name}")
                                             for name in ("count", "positive", "negative"))
                record.update(n_cell=count, n_positive=positive, n_negative=negative,
                              positive_rate=positive / count if positive is not None and count else None)
            values.append(record)
    return paired_stats(deltas)


def comparison_contract(stage, decisions, table):
    """Return the named, preregistered contrasts available by the end of a stage."""
    comparisons = []
    for dataset in DATASETS:
        families = decisions.get("family_representatives", {})
        cstar = decisions.get("Cstar")
        expected_arms = sorted({key[2] for key in table.rows if key[0] == dataset})
        for readout in READOUTS:
            pair_list = [("B1", "B0"), ("H0", "B0"), ("H0", "B1")]
            pair_list += [(arm, parent) for arm in expected_arms
                          if arm in (*FAMILIES["C1"], *FAMILIES["C2"], "C12", "S_C1", "S_C2")
                          for parent in PARENTS]
            for family, choice in families.items():
                original = choice["original_by_dataset"][dataset]
                rep = choice.get("representative_by_dataset", {}).get(dataset, original)
                pair_list += [(original, f"S_{family}"), (rep, f"M_{family}"), (rep, f"P_{family}")]
                pair_list += [("C12", original), *(("C12", f"{control}_{family}") for control in ("M", "P", "S"))]
            if cstar:
                clean = cstar["arms"][dataset]
                pair_list += [
                    ("Cstar_N", "H0_N"), ("HG_B1", "HG_B0"), ("HG_H0", "HG_B1"),
                    *(("HG_Cstar", parent) for parent in EXTENSIONS["HyperGate"][:3]),
                    ("G_B1", "G_B0"), ("G_H0", "G_B1"),
                    *(("G_Cstar", parent) for parent in EXTENSIONS["Gaussian"][:3]),
                ]
                for name, candidate, reference in (
                    ("scope_interaction", "Cstar_N", "H0_N"), ("gaussian_interaction", "G_Cstar", "G_H0"),
                ):
                    comparisons.append((dataset, readout, name,
                                        [(candidate, readout, 1), (reference, readout, -1),
                                         (clean, readout, -1), ("H0", readout, 1)], PROGRESS))
            if all(family in families for family in FAMILIES):
                comparisons.append((dataset, readout, "C12_interaction", [
                    ("C12", readout, 1), (families["C1"]["original_by_dataset"][dataset], readout, -1),
                    (families["C2"]["original_by_dataset"][dataset], readout, -1), ("H0", readout, 1),
                ], PROGRESS))
            for arm, parent in dict.fromkeys(pair_list):
                comparisons.append((dataset, readout, f"{arm}_vs_{parent}",
                                    [(arm, readout, 1), (parent, readout, -1)], PROGRESS))
                if readout == "fixed_ema":
                    comparisons.append((dataset, "interaction", f"{arm}_vs_{parent}_ema_interaction",
                                        [(arm, "fixed_ema", 1), (parent, "fixed_ema", -1),
                                         (arm, "online", -1), (parent, "online", 1)], PROGRESS))
        for arm in expected_arms:
            comparisons.append((dataset, "interaction", f"{arm}_ema", [(arm, "fixed_ema", 1),
                                                                          (arm, "online", -1)], PROGRESS))
    return comparisons


def write_comparisons(directory, table, comparisons):
    paired, summaries = [], []
    for dataset, readout, name, terms, endpoints in comparisons:
        if any(not any(table.get(arm, dataset, seed, kind) for seed in SEEDS) for arm, kind, _ in terms):
            continue
        for endpoint in endpoints:
            for metric in METRICS:
                values = []
                result = contrast(table, dataset, endpoint, metric, terms, values)
                cell = metric[:4] if metric[:4] in CELLS else ""
                common = dict(comparison_id=name, dataset=dataset, readout=readout, endpoint=endpoint,
                              cell=cell, metric=metric)
                paired.extend(dict(common, candidate_id=terms[0][0], reference_id=terms[1][0], **row)
                              for row in values)
                threshold = CELL_LIMIT if cell and metric.endswith("_auc") else (
                    RESOLUTION if metric == "macro_auc" else None)
                status = None
                if threshold is not None and result["mean_delta"] is not None:
                    status = (result["mean_delta"] >= threshold if cell else result["mean_delta"] > threshold)
                summaries.append(dict(common, **result, threshold=threshold, gate_result=status))
    write_csv(directory / "paired.csv", paired)
    write_csv(directory / "summary.csv", summaries)
    absolute = []
    for dataset, seed, arm, readout, endpoint in sorted(table.rows):
        if seed != SEEDS[0]:
            continue
        for metric in METRICS:
            values = [(table.get(arm, dataset, seed, readout, endpoint) or {}).get(metric) for seed in SEEDS]
            result = paired_stats(values)
            absolute.append(dict(dataset=dataset, arm=arm, readout=readout, endpoint=endpoint, metric=metric,
                                 n=result["n"], mean=result["mean_delta"], sd=result["sd_delta"]))
    write_csv(directory / "absolute.csv", absolute)


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_d1_summary(root, stage):
    path = root / stage / "steps.jsonl"
    aggregates = defaultdict(lambda: defaultdict(list))
    for row in json_lines(path):
        for record in row.get("groups", []):
            identity = (row["dataset"], row["seed"], row["arm"], row["phase"], record["parameter_group"])
            selected = row.get("source_cell")
            for source in ("all_steps", selected or "no_pair"):
                bucket = aggregates[(*identity, source)]
                for name, value in record.items():
                    if isinstance(value, (int, float)):
                        bucket[name].append(value)
    output = []
    for (data, seed, arm, phase, group, source), fields in sorted(aggregates.items()):
        for scalar, values in sorted(fields.items()):
            output.append(dict(dataset=data, seed=seed, arm=arm, phase=phase, parameter_group=group,
                               source_cell=source, metric=scalar, n_steps=len(values), mean=mean(values)))
    if not output:
        raise ValueError(f"{stage} has no D1 step diagnostics")
    write_csv(root / stage / "D1_SUMMARY.csv", output)
    return path


def verify_d2(root):
    path = root / "S1" / "diagnostics.jsonl"
    replays = {}
    panels = defaultdict(set)
    for row in json_lines(path):
        if row["diagnostic"] not in ("D2_REPLAY", "D2_PANEL"):
            continue
        key = (row["dataset"], row["seed"], row["epoch"])
        if row["diagnostic"] == "D2_REPLAY":
            if key in replays or set(row["actual_post"]) != {"model", "optimizer", "ema"}:
                raise ValueError(f"invalid exact H0 replay receipt: {key}")
            replays[key] = row["anchor_hash"]
        else:
            key = (*key, row["anchor_hash"])
            evaluation = (row["branch_id"], row["panel"], row["readout"], row["cell"])
            if evaluation in panels[key]:
                raise ValueError(f"duplicate D2 panel evaluation: {key}/{evaluation}")
            panels[key].add(evaluation)
    expected = {(data, seed, epoch) for data in DATASETS for seed in SEEDS for epoch in (5, 20, 35, 50)}
    branches = ("H0", "M9_V0", "M9_V1", *(arm for arms in FAMILIES.values() for arm in arms),
                *(f"M_{arm}" for arms in FAMILIES.values() for arm in arms))
    expected_rows = {(branch, panel, readout, cell) for branch in branches
                     for panel in ("P_TRAIN", "V_A", "V_B") for readout in READOUTS for cell in CELLS}
    if (set(replays) != expected or set(panels) != {(*key, h) for key, h in replays.items()}
            or any(rows != expected_rows for rows in panels.values())):
        raise ValueError("D2 anchor and panel completion does not match the frozen design")
    return path


def write_d4(root, stage, table):
    import numpy as np

    from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import state_digest
    from .diagnostics import fixed_pairs, pair_scores, pair_transition
    from .sources import load_train_validation

    arms = ["B1", "H0"]
    if stage == "S2":
        arms.extend(arm for grid in FAMILIES.values() for arm in grid)
    stages = {arm: "S1" if arm in ("B1", "H0") else "S2" for arm in arms}
    records = []
    inputs = [root / item / "predictions.jsonl" for item in sorted(set(stages.values()))]
    predictions_index = {}
    for manifest in inputs:
        for row in json_lines(manifest):
            key = (row["job_key"], row["readout"], row["endpoint"])
            if row["split"] != "validation" or key in predictions_index:
                raise ValueError(f"invalid or duplicate D4 prediction receipt: {key}")
            predictions_index[key] = row
    for dataset in DATASETS:
        bundle = load_train_validation(dataset, root / "train_validation_inputs")
        labels, domains = bundle.val_y, bundle.val_x[:, -1]
        for seed in SEEDS:
            pairs = fixed_pairs(dataset, seed, "VALIDATION", labels, domains, "D4", 32768)
            scores = {}
            endpoints = {}
            for arm in arms:
                for readout in READOUTS:
                    for endpoint in PROGRESS:
                        row = table.get(arm, dataset, seed, readout, endpoint)
                        path = root / stages[arm] / "predictions" / f"{row['logical_run_id']}_{endpoint}_{readout}.npy"
                        receipt = predictions_index[row["logical_run_id"], readout, endpoint]
                        if (receipt["relative_path"] != str(path.relative_to(root))
                                or receipt["epoch"] != row["epoch"] or receipt["model_hash"] != row["model_hash"]
                                or receipt["data_hash"] != row["data_hash"]
                                or receipt["file_sha256"] != file_hash(path)):
                            raise ValueError(f"D4 probabilities differ from the accepted endpoint: {path}")
                        predictions = np.load(path, mmap_mode="r", allow_pickle=False)
                        if predictions.shape != labels.shape or predictions.dtype != np.float32:
                            raise ValueError(f"invalid D4 probability array: {path}")
                        endpoints[arm, readout, endpoint] = row
                        for cell in CELLS:
                            scores[arm, readout, endpoint, cell] = pair_scores(predictions, cell, pairs[cell])
            for cell in CELLS:
                support = dict(count=int((domains == int(cell[1])).sum()),
                               positive=int(labels[domains == int(cell[1]), int(cell[3])].sum()))
                support["negative"] = support["count"] - support["positive"]
                indices = pairs[cell]
                pair_hash = state_digest(bundle.val_ids[np.r_[indices[0], indices[1]]]) if indices else None
                common = dict(dataset=dataset, seed=seed, cell=cell, panel="VALIDATION",
                              pair_hash=pair_hash, **support)
                for arm in arms:
                    for readout in READOUTS:
                        for before, after in (("E10", "E25"), ("E25", "E50"), ("E10", "E50")):
                            records.append(dict(common, kind="TRAJECTORY_RETENTION", arm=arm, readout=readout,
                                                endpoint=after, reference=before, **pair_transition(
                                                    scores[arm, readout, before, cell], scores[arm, readout, after, cell],
                                                )))
                    for endpoint in PROGRESS:
                        online = scores[arm, "online", endpoint, cell]
                        ema = scores[arm, "fixed_ema", endpoint, cell]
                        records.append(dict(common, kind="EMA_TRANSITION", arm=arm, endpoint=endpoint,
                                            **pair_transition(online, ema)))
                        references = () if arm == "B1" else ("B1",) if arm == "H0" else ("B1", "H0")
                        for ref in references:
                            ref_online = scores[ref, "online", endpoint, cell]
                            ref_ema = scores[ref, "fixed_ema", endpoint, cell]
                            stats = {}
                            for readout, base, candidate in (("online", ref_online, online), ("fixed_ema", ref_ema, ema)):
                                stats[readout] = pair_transition(base, candidate)
                                parent_auc = endpoints[ref, readout, endpoint].get(f"{cell}_auc")
                                records.append(dict(common, kind="PARENT_TRANSITION", arm=arm, reference=ref,
                                                    readout=readout, endpoint=endpoint, parent_auc=parent_auc,
                                                    traffic_mass=support["count"] / (2 * len(labels)), **stats[readout]))
                            overlap = {}
                            if online is not None:
                                for relation, left, right in (
                                    ("correction", online > ref_online, ref_ema > ref_online),
                                    ("destruction", online < ref_online, ref_ema < ref_online),
                                ):
                                    union, intersection = int((left | right).sum()), int((left & right).sum())
                                    overlap.update({f"{relation}_union": union, f"{relation}_intersection": intersection,
                                                    f"{relation}_jaccard": intersection / union if union else None,
                                                    f"{relation}_null_reason": None if union else "EMPTY_UNION",
                                                    f"{relation}_candidate_only": int((left & ~right).sum()),
                                                    f"{relation}_ema_only": int((right & ~left).sum())})
                            records.append(dict(
                                common, kind="DEPLOYMENT_INTERACTION" if endpoint == "selected" else "PROGRESS_INTERACTION",
                                arm=arm, reference=ref, endpoint=endpoint,
                                interaction=(stats["fixed_ema"]["net"] - stats["online"]["net"]
                                             if online is not None else None),
                                null_reason="NO_LABEL_SUPPORT" if online is None else None, **overlap,
                            ))
    path = write_csv(root / stage / "D4_COMPARISONS.csv", records)
    return [path, *inputs]


def analyze(root, stage):
    assert_open(root)
    read_plan()
    if (root / stage / "DECISIONS.json").exists():
        accepted = stage_decision(root, stage)
        latest = root / "decisions.json"
        if not latest.exists() or STAGES.index(load(latest)["stage"]) < STAGES.index(stage):
            write_json(latest, accepted)
        return accepted
    position = STAGES.index(stage)
    stage_queue = read_queue(root, stage)
    run_receipt = root / stage / "RUN_RECEIPT.json"
    pause = load(run_receipt).get("conditional_pause") if stage == "S3d" and run_receipt.exists() else None
    previous = stage_decision(root, STAGES[position - 1]) if position else {}
    decisions = {key: value for key, value in previous.items() if key not in
                 ("stage", "status", "queue_hash", "input_receipts")}
    if not previous:
        decisions = {
            "rules_version": 1, "family_representatives": {}, "selection_readouts": {},
            "Cstar_id": None, "Cstar_params_by_dataset": {}, "Cstar": None,
            "stage_gates": {}, "triggered_arms": {}, "not_triggered_arms": {}, "reason_refs": {},
        }
    rows = {}
    receipts = {}
    for dependency in STAGES[:position + 1]:
        data, paths = accepted_stage(root, dependency, conditional_pause=pause if dependency == stage else None)
        if rows.keys() & data.keys():
            raise ValueError("duplicate accepted trajectory endpoint across stages")
        rows.update(data)
        receipts.update({str(path.relative_to(root)): file_hash(path) for path in paths})
    table = EndpointTable(rows)
    if stage in ("S1", "S2"):
        diagnostic_paths = [write_d1_summary(root, stage)]
        if stage == "S1":
            diagnostic_paths.append(verify_d2(root))
        diagnostic_paths.extend(write_d4(root, stage, table))
        receipts.update({str(path.relative_to(root)): file_hash(path) for path in diagnostic_paths})
    families = decisions["family_representatives"]
    if stage == "S2":
        families, grid = grid_selection(table)
        decisions["family_representatives"] = families
        decisions["selection_readouts"] = {key: value["selection_readout"] for key, value in families.items()}
        write_csv(root / "config_selection.csv", [
            row | {"parent_deltas": json.dumps(row["parent_deltas"]), "cell_deltas": json.dumps(row["cell_deltas"]),
                   "selection_timestamp": load(root / "S2" / "RUN_RECEIPT.json").get("ended_at", "")}
            for row in grid
        ])
    if stage == "S3a":
        short_kernel(table, families)
    if stage == "S3b":
        control_gates(table, families)
        decisions["combination"] = combination_gate(families)
    if stage == "S3c":
        scopes = load(root / "PARAMETER_SCOPE_MANIFEST.json")
        subset = sum(scopes[data]["m3oe"]["s_dense_numel"] for data in DATASETS)
        elements = {"S_C1": 3 * subset, "C1": 3 * subset, "S_C2": 4 * subset,
                    "C2": 4 * subset, "C12": 4 * subset}
        winner, eligible = choose_cstar(table, families, decisions["combination"], elements)
        decisions["Cstar"], decisions["Cstar_id"] = winner, winner["id"] if winner else None
        decisions["cstar_eligible"] = eligible
        if winner:
            specs = { (job["dataset"], job["arm"]): job
                     for st in STAGES[:position + 1] for job in read_queue(root, st)["jobs"] }
            params = ("algorithm_id", "zeta", "alpha", "beta_pair")
            decisions["Cstar_params_by_dataset"] = {
                data: {key: specs[data, winner["arms"][data]][key] for key in params} for data in DATASETS
            }
        conditions = load(root / "AUDIT_RECEIPT.json")["conditional_sources"]
        decisions["stage_gates"]["S3d_extensions"] = extension_gates(table, winner, conditions)
    status = ("TECHNICAL_PAUSED" if pause else
              "COMPLETE" if stage_queue["jobs"] or stage_queue["reuse_slots"] else "NOT_TRIGGERED")
    arms = {}
    for arm, reason in stage_queue["trigger_reasons"].items():
        if reason["status"] == "READY":
            arms[arm] = "TECHNICAL_PAUSED" if pause and arm in EXTENSIONS["Gaussian"] else "COMPLETE"
            decisions["triggered_arms"][arm] = arms[arm]
        else:
            decisions["not_triggered_arms"][arm] = reason["status"]
            arms[arm] = reason["status"]
        decisions["reason_refs"][arm] = reason["reason"]
    decisions["stage_gates"][stage] = {"status": status, "arms": arms}
    decisions.update(stage=stage, status=status, input_receipts=receipts, queue_hash=stage_queue["queue_hash"])
    directory = root / stage
    if pause:
        directory /= "clean"
        decisions["stage_gates"][stage]["conditional_pause"] = {
            key: value for key, value in pause.items() if key != "modules"
        }
    write_comparisons(directory, table, comparison_contract(stage, decisions, table))
    write_json(directory / "DECISIONS.json", decisions, immutable=not pause)
    if not pause:
        write_json(root / "decisions.json", decisions)
    return decisions


def strong_method_gates(table, candidate_by_dataset, readout, published_models):
    output = {}
    for dataset in DATASETS:
        candidate = candidate_by_dataset[dataset]
        scores = {model: table.average(model, dataset, "online") for model in published_models}
        if any(value is None for value in scores.values()):
            raise ValueError(f"incomplete frozen published roster: {dataset}")
        best = max(scores, key=scores.get)
        records = {}
        for model in published_models:
            terms = [(candidate, readout, 1), (model, "online", -1)]
            records[model] = {
                metric: contrast(table, dataset, "selected", metric, terms) for metric in METRICS
            }
        mean_macro = records[best]["macro_auc"]["mean_delta"]
        cells = [records[best][f"{cell}_auc"]["mean_delta"] for cell in CELLS]
        output[dataset] = {
            "strongest_reference": best, "published_means": scores,
            "strong_method_gate": (mean_macro is not None and mean_macro >= .003 and
                                   all(value is not None and value >= CELL_LIMIT for value in cells)),
            "comparisons": records,
        }
    return output
