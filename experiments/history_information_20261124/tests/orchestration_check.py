"""Small structured checks for immutable queues and the stage decision contract."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..analysis import EndpointTable, paired_stats, strong_method_gates
from ..selection import (
    CELLS, C12_CONTROLS, DATASETS, READOUTS, SEEDS, STUDY, choose_cstar, combination_gate,
    assert_open, conditional_modules, control_gates, digest, extension_gates, file_hash, grid_selection,
    load, materialize, parent_gate, read_queue, short_kernel, write_json,
)


def table():
    points = {}

    def set_value(arm, data, readout, value, damaged=False):
        for seed in SEEDS:
            points[data, seed, arm, readout, "selected"] = {
                "data_hash": data,
                "macro_auc": value,
                **{f"{cell}_auc": value if not damaged or cell != CELLS[0] else .495 for cell in CELLS},
            }

    for data in DATASETS:
        for readout in READOUTS:
            for parent in ("B0", "B1", "H0"):
                set_value(parent, data, readout, .5)
            for arm, value in (
                ("C1_Z050", .502), ("C1_Z100", .50204), ("C2_A050", .501),
                ("C2_A100", .503), ("S_C1", .5004), ("S_C2", .5004),
                ("M_C1", .5001), ("P_C1", .5002), ("M_C2", .5001), ("P_C2", .5002),
            ):
                set_value(arm, data, readout, value)
            set_value("C12", data, readout, .504)
            for model, value in (("M3oE", .5), ("HyperGate", .5001), ("AutoIFS", .5002)):
                set_value(model, data, readout, value)
    return EndpointTable(points), set_value


def assert_error(error_type, function, *arguments):
    try:
        function(*arguments)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def check_queues(root, *, gaussian_unresolved=False):
    fingerprint = "a" * 64
    scope = {"scope_hash": dict.fromkeys(("NATIVE_ALL", "S_DENSE", "N", "HG_SHARED_DENSE"), fingerprint)}
    for name in ("STOP_CONTROLS.json", "TEST_POLICY.json", "DESIGN_SPECIFICATION.md", "REUSE_REGISTER.csv"):
        write_json(root / name, {})
    gaussian = write_json(root / "gaussian_fixture.json", {"pinned": True})
    closure = {"status": "VERIFIED", "source_hash": digest("gaussian"), "blocker": None,
               "files": [{"actual_path": str(gaussian), "sha256": file_hash(gaussian)}]}
    unresolved = {"status": "UNRESOLVED_IDENTITY", "source_hash": None, "files": [], "blocker": "fixture"}
    write_json(root / "SOURCE_MANIFEST.json", {
        "source_hash": fingerprint, "published_commit": "fixture-pinned-commit",
        "gaussian_source": unresolved if gaussian_unresolved else closure,
    })
    write_json(root / "DATA_MANIFEST.json", {d: {"data_hash": fingerprint} for d in DATASETS})
    write_json(root / "PARAMETER_SCOPE_MANIFEST.json",
               {d: dict.fromkeys(("m3oe", "hypergate", "autoifs"), scope) for d in DATASETS})
    write_json(root / "AUDIT_RECEIPT.json", {"status": "COMPLETE", "parent_live_reuse": []})
    write_json(root / "PUBLISHED_ROSTER.json", {"slots": []})

    def stage(name, expected, decisions=None):
        queue = materialize(root, name)
        assert len(queue["jobs"]) == expected
        write_json(root / name / "DECISIONS.json", {
            "stage": name, "status": "COMPLETE", "queue_hash": queue["queue_hash"], "input_receipts": {},
            **(decisions or {}),
        })
        return queue

    stage("S1", 30)
    sample, _ = table()
    families, _ = grid_selection(sample)
    stage("S2", 40, {"family_representatives": families})
    short_kernel(sample, families)
    stage("S3a", 20, {"family_representatives": families})
    control_gates(sample, families)
    combo = combination_gate(families)
    matched = stage("S3b", 40, {"family_representatives": families, "combination": combo})
    assert {job["target_algorithm_id"] for job in matched["jobs"] if job["arm"].startswith("M_")} == {"C1", "C2"}
    stage("S3c", 10, {
        "Cstar": {"readout": "online", "arms": dict.fromkeys(DATASETS, "C12")}, "Cstar_params_by_dataset": {
            d: {"algorithm_id": "C12", "zeta": .5, "alpha": 1, "beta_pair": .99} for d in DATASETS},
        "family_representatives": families, "triggered_arms": {}, "not_triggered_arms": {}, "reason_refs": {},
        "stage_gates": {"S3d_extensions": {
            "N": {"status": "READY"}, "HyperGate": {"status": "READY"},
            "Gaussian": {"status": "TECHNICAL_PAUSED" if gaussian_unresolved else "READY"},
        }},
    })
    queue = materialize(root, "S3d")
    assert len(queue["jobs"]) == 100
    check_conditional_dispatch(root, queue, gaussian, closure, unresolved, gaussian_unresolved)
    assert materialize(root, "BASELINE_REPAIR") is None


def check_conditional_dispatch(root, queue, gaussian, closure, unresolved, gaussian_unresolved):
    from .. import analysis, cli, kernel_checks, runner, scheduler, sources

    assert conditional_modules(read_queue(root, "S1")) == ()
    assert conditional_modules({"jobs": [{"arm": "REPAIR_HG"}]}) == ("HyperGate",)
    assert conditional_modules({"jobs": [{"arm": "REPAIR_AUTOIFS"}]}) == ("AutoIFS",)
    assert conditional_modules(queue) == ("N", "HyperGate", "Gaussian")
    source = load(root / "SOURCE_MANIFEST.json")
    args = cli.command_parser().parse_args(["run", "--stage", "S3d"])
    analyze_args = SimpleNamespace(command="analyze", stage="S3d")
    supervisor = runner.Supervisor(root, emit=lambda _: None)
    supervisor.jobs.update({key: {"status": "COMPLETE", "selected": {}}
                            for job in queue["jobs"] for key in job["dependencies"]})
    dispatched = []
    proof = root / "S3d" / "CONDITIONAL_CHECKS.json"
    original = gaussian.read_bytes()
    if gaussian_unresolved:
        gaussian.unlink()

    def check_source(repository, commit):
        assert repository == sources.ROOT.parent / "phase0-3_git" and commit == source["published_commit"]
        return closure if gaussian.is_file() and gaussian.read_bytes() == original else unresolved

    def report_for(module, artifact_root):
        assert artifact_root == root
        identity = {"source_hash": source["source_hash"]}
        if module == "Gaussian":
            identity["gaussian_source_hash"] = closure["source_hash"]
        return {"source": identity, "support": list(DATASETS), "result": "PASS"}

    def failing_oracle(module, artifact_root):
        if module == "Gaussian":
            raise ValueError("fixture Gaussian oracle failure")
        return report_for(module, artifact_root)

    def complete_job(out, job, *_):
        spec = job["spec"]
        dispatched.append(spec.job_key)
        identity = dict(dataset=spec.dataset, seed=spec.seed, config_id=spec.arm, job_key=spec.job_key,
                        attempt_id=1, split="validation", model_hash="fixture", data_hash=spec.data_hash,
                        source_hash=spec.source_hash, config_hash=spec.config_hash, group_hash=spec.group_hash)
        score = .501 if spec.arm == "Cstar_N" else .5
        metrics = dict(macro_auc=score, macro_logloss=.69,
                       **{f"{cell}_{kind}": value for cell in CELLS
                          for kind, value in (("auc", score), ("logloss", .69))})
        runner._csv(out / "runs.csv", [identity | {"status": "COMPLETE"}])
        runner._csv(out / "curves.csv", [identity | dict(epoch=epoch, readout=readout, macro_auc=score)
                                       for readout in READOUTS for epoch in range(1, 51)])
        runner._csv(out / "endpoints.csv", [identity | metrics | dict(epoch=epoch, readout=readout, endpoint=endpoint)
                                          for readout in READOUTS
                                          for endpoint, epoch in zip(analysis.PROGRESS, (1, 10, 25, 50))])
        job.update(status="COMPLETE", epoch=50, attempt=1)

    accept = analysis.accepted_stage

    def accept_with_prior_fixture(root, stage, **kwargs):
        if stage == "S3d":
            return accept(root, stage, **kwargs)
        return (table()[0].rows if stage == "S1" else {}), []

    measurement = {"errors": [], "devices": [{"uuid": "fixture"}], "host_ram_used": .1}
    with patch.object(cli, "check_review"), patch.object(cli, "policy_hash"), \
            patch.object(scheduler, "stop_status", return_value=None), \
            patch.object(scheduler, "sample", return_value=measurement), \
            patch.object(scheduler, "eligible", return_value=True), \
            patch.object(sources, "audit_gaussian_source", side_effect=check_source), \
            patch.object(kernel_checks, "conditional_contracts", side_effect=failing_oracle) as oracle, \
            patch.object(supervisor, "_visible", side_effect=lambda devices: devices), \
            patch.object(supervisor, "_inputs", return_value=root), \
            patch.object(supervisor, "_start", side_effect=complete_job), \
            patch.object(analysis, "accepted_stage", side_effect=accept_with_prior_fixture):
        result = cli.execute(root, args, supervisor)
        assert result["status"] == "TECHNICAL_PAUSED" and not proof.exists()
        pause = result["conditional_pause"]
        assert pause["module"] == "Gaussian" and pause["queue_hash"] == queue["queue_hash"]
        assert pause["source_hash"] == source["source_hash"] and pause["gaussian_commit"] == source["published_commit"]
        assert pause["gaussian_source_status"] == ("UNRESOLVED_IDENTITY" if gaussian_unresolved else "VERIFIED")
        assert [call.args[0] for call in oracle.call_args_list] == (
            ["N", "HyperGate"] if gaussian_unresolved else ["N", "HyperGate", "Gaussian"])
        clean = {digest(job) for job in queue["jobs"] if not job["arm"].startswith("G_")}
        assert len(dispatched) == len(clean) == 60 and set(dispatched) == clean
        assert materialize(root, "S3d") == queue
        report = cli.execute(root, analyze_args)
        assert report["status"] == "TECHNICAL_PAUSED"
        partial_path = Path(report["artifact"])
        partial = load(partial_path)
        assert partial["triggered_arms"]["G_Cstar"] == "TECHNICAL_PAUSED"
        assert partial["triggered_arms"]["HG_Cstar"] == "COMPLETE"
        assert partial["stage_gates"]["S3d"]["conditional_pause"]["queue_hash"] == queue["queue_hash"]
        summary = analysis.csv_rows(partial_path.parent / "summary.csv")
        assert any(row["comparison_id"] == "Cstar_N_vs_H0_N" and row["metric"] == "macro_auc"
                   and row["n"] == "5" and math.isclose(float(row["mean_delta"]), .001) for row in summary)
        assert not any(row["comparison_id"].startswith(("G_", "gaussian_")) for row in summary)
        assert not (root / "S3d" / "DECISIONS.json").exists()
        assert_error(ValueError, cli.freeze, root, supervisor)
        assert_error(FileNotFoundError, materialize, root, "BASELINE_REPAIR")
        assert not (root / cli.FINAL_FREEZE).exists() and not (root / cli.TEST_ACCESS).exists()

        gaussian.write_bytes(original)
        oracle.reset_mock()
        oracle.side_effect = report_for
        result = cli.execute(root, args, supervisor)
        assert result["status"] == "COMPLETE" and result["queue_hash"] == queue["queue_hash"]
        assert len(dispatched) == len(set(dispatched)) == 100
        assert set(dispatched) == {digest(job) for job in queue["jobs"]}
        receipt = load(proof)
        assert receipt["queue_hash"] == queue["queue_hash"] and receipt["source_hash"] == source["source_hash"]
        assert receipt["modules"]["Gaussian"]["source"]["gaussian_source_hash"] == digest("gaussian")
        assert oracle.call_count == 3
        assert cli.execute(root, args, supervisor)["status"] == "COMPLETE"
        assert oracle.call_count == 3 and len(dispatched) == 100
        report = cli.execute(root, analyze_args)
        assert report["status"] == "COMPLETE" and load(report["artifact"])["triggered_arms"]["G_Cstar"] == "COMPLETE"

        gaussian.write_bytes(b"different source bytes")
        assert cli.check_conditionals(root, queue)["module"] == "Gaussian"
        gaussian.write_bytes(original)
        write_json(proof, receipt | {"queue_hash": digest("another queue")})
        assert cli.check_conditionals(root, queue)["module"] is None
    supervisor.close()


def check_test_stop(root, policy):
    from .. import cli, scheduler, sources

    workspaces = (root / "first", root / "second")
    sentinel = workspaces[1] / "artifacts" / STUDY / "USER_STOPPED"
    sentinel.parent.mkdir(parents=True)
    policy_path = write_json(root / cli.TEST_POLICY, policy | {
        "current_task_evaluation_allowed": True, "authorization_source": "fixture", "authorization_scope": "fixture",
    })
    identity = {"gpu_uuid": "fixture-gpu", "owner_generation": "fixture-owner"}
    point = identity | {"actual_get_hash": "model", "ram_packet_hash": "packet"}
    actual = {"model_hash": "model", "ram_hash": "packet", "identity": identity}
    freeze = {"study_id": STUDY, "policy_hash": file_hash(policy_path),
              "algorithm_plan_sha256": file_hash(cli.PLAN), "readout_endpoint": "selected",
              "adaptive_changes_after_test": False, "identity_files": {}, "endpoints": [point]}
    write_json(root / cli.FINAL_FREEZE, freeze | {"freeze_hash": digest(freeze)})
    authorized_loader = sources.load_authorized_test

    with patch.object(scheduler, "WORKSPACES", workspaces), patch.object(cli, "check_review"), \
            patch.object(cli, "selected_device"), patch.object(cli, "_selected") as selected, \
            patch.object(cli, "_pinned_test_rows") as saved_rows, patch.object(sources, "load_authorized_test") as loader:
        scheduler.register_stop_controls(root)

        def stop_in_readback(*_):
            sentinel.touch()
            return actual

        selected.side_effect = stop_in_readback
        assert_error(PermissionError, cli.evaluate_test, root, None)
        selected.assert_called_once_with(None, point)
        assert not (root / cli.TEST_ACCESS).exists()
        saved_rows.assert_not_called()
        loader.assert_not_called()

        sentinel.unlink()
        selected.side_effect = None
        selected.return_value = actual

        def stop_after_claim(_):
            assert (root / cli.TEST_ACCESS).exists()
            sentinel.touch()
            return []

        saved_rows.side_effect = stop_after_claim
        assert_error(PermissionError, cli.evaluate_test, root, None)
        assert "USER_STOPPED" in load(root / "TEST_RECEIPT.json")["failure"]
        loader.assert_not_called()
        claim = load(root / cli.TEST_ACCESS) | {"dataset": DATASETS[0]}
        with patch.object(sources, "OUTPUT", root), patch.object(sources, "_population") as population:
            assert_error(PermissionError, lambda: authorized_loader(DATASETS[0], claim, artifact_root=root))
            population.assert_not_called()


def check():
    from ..cli import _selected, evaluate_test, policy_hash

    result, enter = table()
    families, selected_rows = grid_selection(result)
    assert len(selected_rows) == 24
    assert families["C1"]["original_by_dataset"] == dict.fromkeys(DATASETS, "C1_Z050")
    assert families["C2"]["original_by_dataset"] == dict.fromkeys(DATASETS, "C2_A100")
    assert all(row["selection_readout"] == "online" for row in families.values())

    short_kernel(result, families)
    assert all(not row["short_kernel"]["adopted"] for row in families.values())
    control_gates(result, families)
    combo = combination_gate(families)
    assert combo["triggered"]
    states = {"C1": 3, "S_C1": 3, "C2": 4, "S_C2": 4, "C12": 4}
    winner, _ = choose_cstar(result, families, combo, states)
    assert winner["id"] == "C12"
    enter("P_C2", DATASETS[0], "online", .50395)
    enter("P_C2", DATASETS[1], "online", .50395)
    winner, _ = choose_cstar(result, families, combo, states)
    assert winner["id"] == "C2"
    assert len(combo["control_deltas"][DATASETS[0]]) == len(C12_CONTROLS) == 6
    control_gates(result, families)
    assert not families["C2"]["eligible"]
    assert not combination_gate(families)["triggered"]

    result, enter = table()
    for data in DATASETS:
        enter("S_C1", data, "online", .50193)
        enter("C2_A050", data, "online", .501, True)
        enter("C2_A100", data, "online", .503, True)
    families, _ = grid_selection(result)
    assert families["C2"]["selection_readout"] == "fixed_ema"
    short_kernel(result, families)
    assert families["C1"]["short_kernel"]["adopted"]
    control_gates(result, families)
    assert families["C1"]["representative_id"] == "S_C1"
    assert not combination_gate(families)["triggered"]
    winner, _ = choose_cstar(result, families, combination_gate(families), states)
    assert winner["id"] == "S_C1"
    scopes = extension_gates(result, winner, {"HyperGate": "VERIFIED", "Gaussian": "UNRESOLVED_IDENTITY"})
    assert scopes["N"]["status"] == "READY"
    assert scopes["Gaussian"]["triggered"] and scopes["Gaussian"]["status"] == "TECHNICAL_PAUSED"

    for data in DATASETS:
        enter("C2_A050", data, "fixed_ema", .501, True)
        enter("C2_A100", data, "fixed_ema", .503, True)
    families, _ = grid_selection(result)
    assert families["C2"]["status"] == "TRADEOFF_CANDIDATE"
    assert families["C2"]["selection_readout"] == "online"
    del result.rows[DATASETS[0], SEEDS[0], "B1", "online", "selected"]
    assert not parent_gate(result, "C1_Z050", DATASETS[0], "online")["parent_increment"]

    class Boundary:
        def diff(self, arm, parent, data, readout, metric="macro_auc"):
            if metric != "macro_auc":
                return -.001
            return 0 if parent == "B0" else .0001

    boundary = parent_gate(Boundary(), "C1_Z050", DATASETS[0], "online")
    assert boundary["safe"] and not boundary["parent_increment"]
    assert paired_stats([1., None, 0., -1., 0.])["mean_delta"] is None
    stats = paired_stats([-.003, -.001, 0, .001, .003])
    assert stats["n"] == 5 and stats["positive"] == stats["negative"] == 2 and stats["zero"] == 1
    assert math.isclose(stats["ci95_low"], -stats["ci95_high"])
    original, _ = table()
    gate = strong_method_gates(original, dict.fromkeys(DATASETS, "C12"), "fixed_ema",
                               ("M3oE", "HyperGate", "AutoIFS"))
    assert all(item["strongest_reference"] == "AutoIFS" and item["strong_method_gate"]
               for item in gate.values())

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        check_queues(root / "stage_fixture")
        check_queues(root / "unresolved_gaussian_fixture", gaussian_unresolved=True)
        policy = {
            "study_id": STUDY, "holdout_identity": "HISTORICALLY_EXPOSED",
            "global_freeze_before_new_test": True, "actual_selected_ram_get_hash_required": True,
            "honor_existing_active_scope_stop_freeze_ram_gates": True,
            "new_test_endpoints": ["selected"], "adaptive_changes_after_test": False,
            "current_task_evaluation_allowed": False,
        }
        write_json(root / "TEST_POLICY.json", policy)
        assert_error(PermissionError, lambda: policy_hash(root, require_permission=True))
        queue = {"stage": "S1", "study_id": STUDY, "schema_version": 1, "jobs": [],
                 "prerequisite_receipts": {}}
        queue["queue_hash"] = digest(queue)
        location = root / "S1" / "QUEUE.json"
        write_json(location, queue, immutable=True)
        assert read_queue(root, "S1") == queue
        assert_error(ValueError, lambda: write_json(location, queue | {"jobs": [1]}, immutable=True))
        write_json(location, queue | {"jobs": [1]})
        assert_error(ValueError, read_queue, root, "S1")
        write_json(root / "FINAL_FREEZE.json", {"study_id": STUDY})
        assert_error(ValueError, assert_open, root)
        write_json(root / "TEST_ACCESS.json", {"study_id": STUDY})
        assert_error(PermissionError, evaluate_test, root, None)
        check_test_stop(root / "stop_fixture", policy)

    class IncorrectReadback:
        def selected_get(self, _job, _readout):
            return dict(model_hash="different", identity={})

    point = dict(job_key="synthetic", readout="online", model_hash="expected", epoch=1,
                 config_hash="config", data_hash="data", source_hash="source", group_hash="scope")
    assert_error(ValueError, _selected, IncorrectReadback(), point)
    return {"status": "PASS", "checks": [
        "immutable queue fingerprint and all stage counts", "five-seed paired and missing-data statistics",
        "safe before trade-off and readout priority", "short-kernel cross-dataset replacement",
        "C12 six-control and Cstar gates", "Gaussian source gate", "fixed published comparator",
        "conditional dispatch and source-bound oracle receipts",
        "Gaussian-only identity and acceptance pause, clean numeric analysis, same-queue supplement, freeze gate",
        "Test permission, mismatched RAM readback and stop during preflight or before opening",
    ]}


if __name__ == "__main__":
    print(check())
