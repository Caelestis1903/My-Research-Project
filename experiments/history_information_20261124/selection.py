"""Immutable stage queues and pre-registered Validation decisions."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from statistics import mean

STUDY = "history_information_20261124"
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / STUDY
PLAN = ROOT / "research_plans/26-11-24_history_information/experiment_manifest.json"
DATASETS = ("movielens", "kuairand")
SEEDS = tuple(range(2022, 2027))
READOUTS = ("online", "fixed_ema")
CELLS = tuple(f"d{d}t{t}" for d in range(3) for t in range(2))
FAMILIES = {"C1": ("C1_Z050", "C1_Z100"), "C2": ("C2_A050", "C2_A100")}
PARENTS = ("B0", "B1", "H0")
RESOLUTION = 1e-4
CELL_LIMIT = -.001
STAGES = ("S1", "S2", "S3a", "S3b", "S3c", "S3d", "BASELINE_REPAIR")
EXTENSIONS = {
    "N": ("H0_N", "Cstar_N"),
    "HyperGate": ("HG_B0", "HG_B1", "HG_H0", "HG_Cstar"),
    "Gaussian": ("G_B0", "G_B1", "G_H0", "G_Cstar"),
}
C12_CONTROLS = tuple(f"{control}_{family}" for family in FAMILIES for control in ("M", "S", "P"))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_bytes(path, content, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError(f"immutable receipt differs: {path}")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_json(path, value, *, immutable=False):
    return write_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        immutable=immutable,
    )


def read_plan():
    plan = load(PLAN)
    if (plan["schema_version"] != 1 or plan["study_id"] != STUDY
            or tuple(plan["seeds"]) != SEEDS or tuple(plan["datasets"]) != DATASETS):
        raise ValueError("unexpected plan schema or population")
    return plan


def assert_open(root):
    for name in ("FINAL_FREEZE.json", "TEST_ACCESS.json"):
        if (root / name).exists():
            raise ValueError(f"Validation decisions and training are frozen by {name}")


def stage_decision(root, stage):
    path = root / stage / "DECISIONS.json"
    obj = load(path)
    if obj["stage"] != stage or obj["status"] not in ("COMPLETE", "NOT_TRIGGERED"):
        raise ValueError(f"{stage} has no accepted stage decision")
    queue = read_queue(root, stage)
    if obj["queue_hash"] != queue["queue_hash"]:
        raise ValueError(f"{stage} decision refers to a different queue")
    for relative, fingerprint in obj["input_receipts"].items():
        if file_hash(root / relative) != fingerprint:
            raise ValueError(f"decision input changed: {relative}")
    return obj


def parent_gate(table, arm, data, readout):
    macro = {}
    cells = {}
    for parent in PARENTS:
        macro[parent] = table.diff(arm, parent, data, readout)
        cells[parent] = {
            cell: table.diff(arm, parent, data, readout, f"{cell}_auc") for cell in CELLS
        }
    increment = (all(value is not None for value in macro.values())
                 and macro["B0"] >= 0 and macro["B1"] > RESOLUTION and macro["H0"] > RESOLUTION)
    safe = all(value is not None and value >= CELL_LIMIT
               for row in cells.values() for value in row.values())
    return {"parent_deltas": macro, "cell_deltas": cells, "parent_increment": increment, "safe": safe}


def grid_selection(table):
    choices = {}
    grid = []
    for family, arms in FAMILIES.items():
        selected = {readout: {} for readout in READOUTS}
        for data in DATASETS:
            for readout in READOUTS:
                rows = [
                    {"family": family, "dataset": data, "grid_config": arm,
                     "selection_readout": readout, "mean_val_auc": table.average(arm, data, readout),
                     **parent_gate(table, arm, data, readout)}
                    for arm in (*arms, "H0")
                ]
                valid = [row for row in rows[:2] if row["mean_val_auc"] is not None]
                if len(valid) != 2 or rows[2]["mean_val_auc"] is None:
                    raise ValueError(f"incomplete S2 grid: {family}/{data}/{readout}")
                candidates = [row for row in valid if row["safe"]] or valid
                best = max(row["mean_val_auc"] for row in candidates)
                chosen = next(row for row in candidates if best - row["mean_val_auc"] <= RESOLUTION)
                selected[readout][data] = chosen
                for row in rows:
                    row["selected"] = row is chosen
                    row["tie_rule"] = "PREFER_SMALLER_NONZERO_WITHIN_1E-4"
                    row["six_cell_safe"] = row["safe"]
                grid.extend(rows)
        safe_signal = [r for r in READOUTS if any(
            row["safe"] and row["parent_increment"] for row in selected[r].values())]
        increment_signal = [r for r in READOUTS if any(
            row["parent_increment"] for row in selected[r].values())]
        readout = (safe_signal or increment_signal or list(READOUTS))[0]
        choices[family] = {
            "selection_readout": readout,
            "status": ("CONTROLS" if safe_signal else
                       "TRADEOFF_CANDIDATE" if increment_signal else "SCIENTIFIC_CLOSED"),
            "original_by_dataset": {data: row["grid_config"] for data, row in selected[readout].items()},
            "selected_rows": selected,
        }
    return choices, grid


def short_kernel(table, families):
    for family, choice in families.items():
        if choice["status"] == "SCIENTIFIC_CLOSED":
            continue
        readout = choice["selection_readout"]
        short = f"S_{family}"
        gates = {data: parent_gate(table, short, data, readout) for data in DATASETS}
        differences = {data: table.diff(short, original, data, readout)
                       for data, original in choice["original_by_dataset"].items()}
        values = list(differences.values())
        average = mean(values) if all(value is not None for value in values) else None
        use_short = (
            average is not None and average >= -RESOLUTION
            and all(gate["safe"] for gate in gates.values())
            and any(gate["parent_increment"] for gate in gates.values())
        )
        choice["representative_by_dataset"] = (
            dict.fromkeys(DATASETS, short) if use_short else choice["original_by_dataset"].copy()
        )
        choice["representative_id"] = short if use_short else family
        choice["short_kernel"] = {
            "adopted": use_short, "mean_delta": average, "deltas": differences, "parent_gates": gates,
        }
    return families


def control_gates(table, families):
    for family, choice in families.items():
        if choice["status"] == "SCIENTIFIC_CLOSED":
            continue
        readout = choice["selection_readout"]
        short_adopted = choice["short_kernel"]["adopted"]
        references = (f"M_{family}", f"P_{family}") + (() if short_adopted else (f"S_{family}",))
        by_data = {}
        for data, arm in choice["representative_by_dataset"].items():
            gate = parent_gate(table, arm, data, readout)
            gaps = {ref: table.diff(arm, ref, data, readout) for ref in references}
            by_data[data] = {
                **gate, "control_deltas": gaps,
                "control_coverage": {ref: value is not None and value <= RESOLUTION
                                     for ref, value in gaps.items()},
                "qualifies_locally": gate["safe"] and gate["parent_increment"]
                and all(value is not None and value > RESOLUTION for value in gaps.values()),
            }
        choice["control_gates"] = by_data
        choice["eligible"] = (choice["status"] != "TRADEOFF_CANDIDATE"
                              and all(row["safe"] for row in by_data.values())
                              and any(row["qualifies_locally"] for row in by_data.values()))
    return families


def combination_gate(families):
    first, second = (families[family] for family in FAMILIES)
    shared_readout = first["selection_readout"] == second["selection_readout"]
    original = all(item.get("representative_id") == key for key, item in families.items())
    shared_datasets = [data for data in DATASETS if all(
        item.get("control_gates", {}).get(data, {}).get("qualifies_locally") for item in families.values()
    )]
    return {
        "triggered": original and shared_readout and all(x.get("eligible") for x in families.values())
        and bool(shared_datasets),
        "shared_datasets": shared_datasets, "shared_readout": shared_readout, "original": original,
        "readout": first["selection_readout"] if shared_readout else None,
    }


def choose_cstar(table, families, combo, active_elements):
    order = ("S_C1", "C1", "S_C2", "C2", "C12")
    candidates = []

    def entry(identifier, arms, readout):
        scores = [table.diff(arms[data], "B1", data, readout) for data in DATASETS]
        return {"id": identifier, "arms": arms, "readout": readout, "score": mean(scores),
                "active_state_elements": active_elements[identifier]}

    for choice in families.values():
        if choice.get("eligible"):
            candidates.append(entry(choice["representative_id"], choice["representative_by_dataset"],
                                    choice["selection_readout"]))
    if combo["triggered"]:
        readout = combo["readout"]
        gates = {data: parent_gate(table, "C12", data, readout) for data in DATASETS}
        controls = {data: {ref: table.diff("C12", ref, data, readout) for ref in C12_CONTROLS}
                    for data in DATASETS}
        joint = entry("C12", dict.fromkeys(DATASETS, "C12"), readout)
        best = max(x["score"] for x in candidates if x["readout"] == readout)
        eligible = (
            joint["score"] - best > RESOLUTION and all(gate["safe"] for gate in gates.values())
            and any(gates[data]["parent_increment"]
                    and all(value is not None and value > RESOLUTION for value in controls[data].values())
                    for data in DATASETS)
        )
        combo.update({"eligible": eligible, "gain_over_best_single": joint["score"] - best,
                      "parent_gates": gates, "control_deltas": controls})
        if eligible:
            candidates.append(joint)
    for readout in READOUTS:
        pool = [item for item in candidates if item["readout"] == readout]
        if pool:
            best = max(item["score"] for item in pool)
            return min((item for item in pool if best - item["score"] <= RESOLUTION),
                       key=lambda item: (item["active_state_elements"], order.index(item["id"]))), candidates
    return None, candidates


def extension_gates(table, cstar, conditions):
    ema = {} if not cstar else {
        data: parent_gate(table, cstar["arms"][data], data, "fixed_ema") for data in DATASETS
    }
    gaussian = bool(ema and all(gate["safe"] for gate in ema.values())
                    and any(gate["parent_increment"] for gate in ema.values()))
    outcomes = {}
    for module, triggered in (("N", bool(cstar)), ("HyperGate", bool(cstar)), ("Gaussian", gaussian)):
        source = conditions.get(module, "VERIFIED" if module == "N" else "UNRESOLVED_IDENTITY")
        outcomes[module] = {
            "triggered": triggered, "source_status": source,
            "status": ("NOT_TRIGGERED" if not triggered else
                       "READY" if source == "VERIFIED" else "TECHNICAL_PAUSED"),
            "ema_parent_gates": ema if module == "Gaussian" else {},
            "reason": "CSTAR_AND_FIXED_EMA_GATE" if module == "Gaussian" else "CSTAR_INDEPENDENT_INCREMENT",
        }
    return outcomes


def read_queue(root, stage):
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    path = root / stage / "QUEUE.json"
    queue = load(path)
    computed = digest({key: value for key, value in queue.items() if key != "queue_hash"})
    if (queue["schema_version"] != 1 or queue["stage"] != stage or queue["study_id"] != STUDY
            or queue["queue_hash"] != computed):
        raise ValueError(f"invalid queue identity: {path}")
    for relative, fingerprint in queue["prerequisite_receipts"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != fingerprint:
            raise ValueError(f"queue prerequisite changed: {relative}")
    return queue


def conditional_modules(queue):
    arms = {job["arm"] for job in queue["jobs"]}
    conditional_arms = EXTENSIONS | {
        "HyperGate": (*EXTENSIONS["HyperGate"], "REPAIR_HG"),
        "AutoIFS": ("REPAIR_AUTOIFS",),
    }
    return tuple(module for module, module_arms in conditional_arms.items() if arms.intersection(module_arms))


def gaussian_paused_jobs(queue, pause):
    jobs = [job for job in queue["jobs"] if job["arm"] in EXTENSIONS["Gaussian"]]
    identity = {key: queue[key] for key in ("study_id", "stage", "queue_hash")}
    if (queue["stage"] != "S3d" or not jobs or pause.get("module") != "Gaussian"
            or pause.get("status") != "TECHNICAL_PAUSED"
            or any(pause.get(key) != value for key, value in identity.items())
            or any(job["source_hash"] != pause.get("source_hash") for job in queue["jobs"])):
        raise ValueError("Gaussian pause does not match the immutable queue and source")
    return {digest(job) for job in jobs}


def pending_baseline_repairs(root, roster=None):
    if roster is None:
        roster = load(root / "PUBLISHED_ROSTER.json")
    transfer = read_queue(root, "S3d")
    hg = {(job["dataset"], job["seed"]) for job in transfer["jobs"] if job["arm"] == "HG_B0"}
    missing = {model: set() for model in ("HyperGate", "AutoIFS")}
    for slot in roster["slots"]:
        model = slot["model"]
        key = (slot["dataset"], slot["seed"])
        if model in missing and slot["status"] != "NUMERIC_REUSE" and (model != "HyperGate" or key not in hg):
            missing[model].add(key)
    return missing


def materialize(root, stage, decisions_path=None):
    from .protocol import RunSpec

    assert_open(root)
    plan = read_plan()
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    source = load(root / "SOURCE_MANIFEST.json")
    data_manifest = load(root / "DATA_MANIFEST.json")
    scopes = load(root / "PARAMETER_SCOPE_MANIFEST.json")
    audit = load(root / "AUDIT_RECEIPT.json")
    if audit["status"] != "COMPLETE":
        raise ValueError("S0 must pass before materializing a queue")
    receipt_paths = {
        "SOURCE_MANIFEST.json", "DATA_MANIFEST.json", "PARAMETER_SCOPE_MANIFEST.json",
        "AUDIT_RECEIPT.json", "REUSE_REGISTER.csv", "PUBLISHED_ROSTER.json",
        "STOP_CONTROLS.json", "DESIGN_SPECIFICATION.md", "TEST_POLICY.json",
    }
    index = STAGES.index(stage)
    dependencies = list(STAGES[:index])
    previous = stage_decision(root, dependencies[-1]) if dependencies else {}
    dependency_jobs = []
    for dependency in dependencies:
        stage_decision(root, dependency)
        dependency_jobs.extend(RunSpec.from_dict(row) for row in read_queue(root, dependency)["jobs"])
        receipt_paths.add(f"{dependency}/DECISIONS.json")
        receipt_paths.add(f"{dependency}/QUEUE.json")
    if decisions_path and load(decisions_path) != previous:
        raise ValueError("supplied decisions differ from the preceding validated stage decision")
    families = previous.get("family_representatives", {})
    templates = {row["arm_id"]: row for row in plan["arm_templates"]}
    jobs = []
    reasons = {}
    reused = []

    def add_spec(arm, dataset, seed):
        template = templates[arm]
        family = arm[-2:]
        kwargs = {}
        if template["stage"] in ("S3a", "S3b"):
            choice = families[family]
            original = templates[choice["original_by_dataset"][dataset]]
            kwargs["selection_readout"] = choice["selection_readout"]
            if arm.startswith(("S_", "M_")):
                kwargs.update(zeta=original["zeta"], alpha=original["alpha"])
            if arm.startswith("M_"):
                target = templates[choice["representative_by_dataset"][dataset]]
                kwargs.update(beta_pair=target["beta_pair"], target_algorithm_id=target["algorithm_id"])
        if arm == "C12":
            kwargs.update(
                zeta=templates[families["C1"]["original_by_dataset"][dataset]]["zeta"],
                alpha=templates[families["C2"]["original_by_dataset"][dataset]]["alpha"],
                selection_readout=previous["combination"]["readout"],
            )
        if template["stage"] == "S3d":
            winner = previous["Cstar"]
            kwargs["selection_readout"] = winner["readout"]
            if template["algorithm_id"] == "FROZEN_CSTAR":
                kwargs.update(previous["Cstar_params_by_dataset"][dataset])
        group = scopes[dataset][template["backbone"]]["scope_hash"][template.get("scope", "NATIVE_ALL")]
        prerequisites = tuple(spec.job_key for spec in dependency_jobs
                              if spec.dataset == dataset and spec.seed == seed)
        jobs.append(RunSpec(dataset=dataset, seed=seed, arm=arm, stage=stage,
                            source_hash=source["source_hash"], data_hash=data_manifest[dataset]["data_hash"],
                            group_hash=group, dependencies=prerequisites, **kwargs).to_dict())

    roster = load(root / "PUBLISHED_ROSTER.json")
    repair_slots = pending_baseline_repairs(root, roster) if stage == "BASELINE_REPAIR" else {}
    if stage == "BASELINE_REPAIR" and not any(repair_slots.values()):
        return None
    for arm, template in templates.items():
        if template["stage"] != stage:
            continue
        status = "READY"
        reason = template["trigger"]
        slots = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
        if stage == "S1":
            for slot in audit["parent_live_reuse"]:
                if slot["arm"] == arm:
                    reused.append(slot)
                    slots.remove((slot["dataset"], slot["seed"]))
        if stage in ("S3a", "S3b") and families[arm[-2:]]["status"] == "SCIENTIFIC_CLOSED":
            status = "NOT_TRIGGERED"
        if stage == "S3c" and not previous["combination"]["triggered"]:
            status = "NOT_TRIGGERED"
        if stage == "S3d":
            module = next(name for name, group in EXTENSIONS.items() if arm in group)
            status = previous["stage_gates"]["S3d_extensions"][module]["status"]
            if module == "Gaussian" and status == "TECHNICAL_PAUSED":
                # The scientific trigger survives a source pause. Dispatch checks the source separately.
                status = "READY"
        if stage == "BASELINE_REPAIR":
            model = "HyperGate" if arm == "REPAIR_HG" else "AutoIFS"
            missing = repair_slots[model]
            slots = [slot for slot in slots if slot in missing]
            if not slots:
                status = "NOT_TRIGGERED"
        reasons[arm] = {"status": status, "reason": reason}
        if status == "READY":
            for dataset, seed in slots:
                add_spec(arm, dataset, seed)
    queue = {
        "schema_version": 1, "study_id": STUDY, "stage": stage,
        "jobs": jobs, "reuse_slots": reused, "dependencies": dependencies, "trigger_reasons": reasons,
        "prerequisite_receipts": {name: file_hash(root / name) for name in sorted(receipt_paths)},
    }
    queue["queue_hash"] = digest(queue)
    write_json(root / stage / "QUEUE.json", queue, immutable=True)
    return queue
