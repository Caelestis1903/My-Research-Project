import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.three_directions_20261128 import c0_audit as audit, ram_state
from experiments.three_directions_20261128.macro_pair_sampler import MacroPairSampler, ordered_ids_hash
from experiments.three_directions_20261128.protocol import scope_digest


class TinyRouting(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = torch.nn.ModuleList(torch.nn.Linear(1, 1) for _ in range(6))
        for i, tower in enumerate(self.tower):
            with torch.no_grad():
                tower.weight.fill_(i + 1)
                tower.bias.fill_(i / 2)

    def forward(self, x):
        logits = torch.cat([tower(x[:1].T) for tower in self.tower], dim=1)
        domain = x[-1].long()
        return logits.gather(1, torch.stack((domain, domain + 3), dim=1)).sigmoid()


def tiny_train():
    rows = [(d, label) for d in range(3) for label in range(4) for _ in range(1 + (d + label) % 3)]
    x = np.array([(i % 5, d) for i, (d, _) in enumerate(rows)], dtype=np.float32)
    y = np.array([divmod(label, 2) for _, label in rows], dtype=np.float32)
    ids = (np.arange(len(rows), dtype=np.int64) + 20)[::-1]
    return x, y, ids


def test_exact_candidate_enumeration_and_weights():
    x, y, ids = tiny_train()
    sampler = MacroPairSampler("movielens", x[:, -1], y, ids, "fixture")
    table = audit.candidate_set(sampler, 2022)
    repeat = audit.candidate_set(sampler, 2022)
    for key, row in table.items():
        d, t, a, b = key
        left, right = row["left"], row["right"]
        labels = 2 * y[:, 0] + y[:, 1]
        expected = [(i, j) for i in range(len(y)) if x[i, -1] == d and labels[i] == a
                    for j in range(len(y)) if x[j, -1] == d and labels[j] == b]
        assert list(zip(left, right)) == expected
        assert (y[left, t] == 1).all() and (y[right, t] == 0).all()
        assert row["hash"] == repeat[key]["hash"]
        assert row["ordered_ids_hash"] == ordered_ids_hash(np.r_[ids[left], ids[right]])
        assert len(row["ranks"]) == row["universe"] == len(set(expected))
    for cell in audit.CELLS:
        assert sum(row["w"] for key, row in table.items() if f"d{key[0]}t{key[1]}" == cell) == pytest.approx(1 / 6)
    assert len(set(row["w"] for row in table.values())) > 1
    assert sum(row["q"] for row in table.values()) == pytest.approx(1)
    other = audit.candidate_set(MacroPairSampler("movielens", x[:, -1], y, ids, "another"), 2022)
    assert all(other[key]["hash"] != table[key]["hash"] for key in table)


def test_sampled_cap_matches_the_registered_rng(monkeypatch):
    monkeypatch.setattr(audit, "CAP", 3)
    x, y, ids = tiny_train()
    sampler = MacroPairSampler("movielens", x[:, -1], y, ids, "fixture")
    candidates = audit.candidate_set(sampler, 2023)
    for key, row in candidates.items():
        context = dict(dataset="movielens", seed=2023, epoch=0, global_step=0)
        stream = "c0_macro6_candidates|" + "|".join(map(str, key))
        expected = np.sort(audit.rng(context, stream).choice(row["universe"], min(3, row["universe"]), replace=False))
        np.testing.assert_array_equal(row["ranks"], expected)


def test_routed_fp32_margin_parity_and_strict_transition_counts():
    x, y, ids = tiny_train()
    sampler = MacroPairSampler("movielens", x[:, -1], y, ids, "fixture")
    candidates = audit.candidate_set(sampler, 2022)
    endpoints = np.unique(np.concatenate([a for c in candidates.values() for a in (c["left"], c["right"])]))
    model = TinyRouting()
    audit.setup(2022)
    obtained = audit.score_logits(model, x, endpoints)
    with torch.inference_mode():
        bx = torch.from_numpy(x[endpoints]).T
        probability = model(bx)
        direct = torch.cat([tower(bx[:1].T) for tower in model.tower], dim=1).numpy()
    assert obtained.dtype == np.float32
    torch.testing.assert_close(torch.tensor(obtained).sigmoid(), probability, atol=1e-6, rtol=1e-5)
    margins = audit.candidate_margins(obtained, endpoints, candidates)
    for (d, t, a, b), values in candidates.items():
        column = d + 3 * t
        expected = [np.float64(direct[np.where(endpoints == i)[0][0], column])
                    - np.float64(direct[np.where(endpoints == j)[0][0], column])
                    for i, j in zip(values["left"], values["right"])]
        np.testing.assert_array_equal(margins[d, t, a, b], expected)
    before = np.array([0, -.2, .0009, .001, .2, 6, .05])
    after = np.array([1, -2, -.01, -.001, 0, 1, 2])
    out, subsets = audit.transitions(before, after, True)
    assert (out["historical_correct"], out["strict_reversal"], out["tie_loss"], out["retained"]) == (5, 2, 1, 2)
    assert out["robust_strict_reversal"] == 1
    assert out["margin_groups"]["tie_loss"]["change"]["mean"] == -.2
    assert out["confidence"]["above_clip"] == 1
    assert [b["count"] for b in out["confidence"]["bins"]] == [3, 1, 0, 1]
    assert audit.checkpoint_metrics(after)["concordance"] == pytest.approx(3.5 / 7)
    assert set(subsets) >= {"candidates", "historical_correct", "strict_reversal", "tie_loss", "retained"}


def test_weighted_conditional_ratio_and_empty_conditioning_set():
    rows = [dict(w=.1, candidates=2, historical_correct=1, strict_reversal=1),
            dict(w=.3, candidates=10, historical_correct=8, strict_reversal=2)]
    result = audit.target_rates(rows, lambda r: r["strict_reversal"], lambda r: r["historical_correct"])
    assert result["value"] == pytest.approx((.1 * .5 + .3 * .2) / (.1 * .5 + .3 * .8))
    assert result["weighted_denominator"] == pytest.approx((.1 * .5 + .3 * .8) / .4)
    empty = audit.target_rates(rows, lambda r: 0, lambda r: 0)
    assert empty == dict(value=None, weighted_numerator=0, weighted_denominator=0, reason="EMPTY_CONDITIONING_SET")


def test_replay_and_exact_duplicate_exposure(monkeypatch):
    x, y, ids = tiny_train()
    sampler = MacroPairSampler("movielens", x[:, -1], y, ids, "fixture")
    draws, stamps = audit.replay(sampler, 2022, 2, 3, {1, 3, 6})
    for stamp in stamps.values():
        sampled = sampler.sample_macro6(stamp["context"])
        assert stamp["condition"] == list(sampled["condition"])
        assert stamp["pair_index_hash"] == sampled["pair_index_hash"]
    assert sum(len(items) for items in draws.values()) == 6
    monkeypatch.setattr(audit, "PAIR_COUNT", 4)
    candidate = dict(ranks=np.array([0, 2, 3]), universe=5)
    items = [(1, np.array([1, 1, 1, 1])), (2, np.array([0, 0, 3, 4]))]
    subsets = dict(all=np.ones(3, dtype=bool), historical=np.array([False, True, False]),
                   empty=np.zeros(3, dtype=bool))
    output = audit.exposure(candidate, items, subsets, dict(start_exclusive=1, end_inclusive=2))
    assert (output["draws"], output["distinct_pairs"], output["repeated_draws"]) == (4, 3, 1)
    assert {k: v for k, v in output["subsets"]["all"].items() if v is not None} == pytest.approx(dict(
        members=3, observed_members=2, draw_hits=3, coverage=2 / 3, draw_share=3 / 4,
        exposures_per_candidate=1, uniform_expected_hits=4 * 3 / 5, uniform_expected_coverage=1 - .8**4))
    assert output["subsets"]["empty"]["coverage"] is None
    empty = audit.exposure(candidate, [], subsets, dict(start_exclusive=1, end_inclusive=2))
    assert empty["subsets"]["all"]["draw_share"] is None
    assert empty["subsets"]["all"]["uniform_expected_coverage"] == 0


@pytest.mark.parametrize(("epoch", "e5", "e10", "formation"), [
    (4, "POST_SELECTED_SNAPSHOT", "POST_SELECTED_SNAPSHOT", False),
    (5, "SELECTED_AT_SNAPSHOT", "POST_SELECTED_SNAPSHOT", False),
    (8, "PRE_SELECTED", "POST_SELECTED_SNAPSHOT", False),
    (10, "PRE_SELECTED", "SELECTED_AT_SNAPSHOT", True),
    (12, "PRE_SELECTED", "PRE_SELECTED", True),
])
def test_window_timing(epoch, e5, e10, formation):
    windows = audit.transition_windows(epoch, 3)
    assert windows["E05_to_selected"]["classification"] == e5
    assert windows["E10_to_selected"]["classification"] == e10
    assert windows["E05_to_E10"]["eligible"] is formation
    assert windows["E10_to_selected"]["horizon_epochs"] == epoch - 10
    assert windows["E10_to_selected"]["end_inclusive"] == epoch * 3


def test_missing_roster_and_stop_record_explicit_dispositions(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.runner, "freeze_roster", lambda *a, **kw: pytest.fail("Missing roster must stay missing"))
    assert audit.run(tmp_path) == "INPUT_PENDING"
    output = tmp_path / "C0"
    status = audit.read(output / "STATUS.json")
    assert len(status["input_inventory"]) == 20
    assert {r["state"] for r in status["input_inventory"].values()} == {"INPUT_PENDING"}
    assert audit.read(output / "PAIR_METRICS.json") == []
    assert set(tmp_path.iterdir()) == {output}
    marker = dict(status="USER_STOPPED", path="fixture")
    monkeypatch.setattr(audit.scheduler, "stop_status", lambda _: marker)
    assert audit.run(tmp_path) == "USER_STOPPED"
    status = audit.read(output / "STATUS.json")
    assert {r["state"] for r in status["input_inventory"].values()} == {"STOPPED_UNINSPECTED"}
    assert status["stop"] == marker
    assert status["test_accesses"] == 0


def fixture_train(tmp_path, dataset):
    x, y, ids = tiny_train()
    cache = tmp_path / "cache"
    cache.mkdir()
    for field, values in zip(("x", "y", "row_ids"), (x, y, ids)):
        np.save(cache / f"train_{field}.npy", values)
    schema = audit.source.FeatureSchema(dataset, ["feature", "domain_indicator"], ["feature"], ["domain_indicator"],
                                        dict(domain_indicator=3), encoder_hash="encoder", scaler_hash="scaler")
    audit.write(cache / "features.json", schema.as_dict())
    joint = audit.joint_counts(x, y)
    supports = {}
    for d in range(3):
        for t in range(2):
            selected = x[:, -1] == d
            positives = int(y[selected, t].sum())
            supports[f"d{d}t{t}"] = dict(positive=positives, negative=int(selected.sum()) - positives,
                                        active_conditions=4)
    identity = dict(
        dataset=dataset, split_id="original", preprocessing_fit="train_only", raw_sha256="raw-sha",
        features=schema.as_dict(), encoder_hash=schema.encoder_hash, scaler_hash=schema.scaler_hash,
        array_hashes=dict(train_x=audit.source.array_hash(x), train_y=audit.source.array_hash(y)),
        split_rows=dict(train=len(y)), split_row_id_hashes=dict(train=audit.source.row_ids_hash(ids)),
        split_row_fingerprints=dict(train=audit.source.split_row_fingerprint(x, y, ids)),
        train_domain_counts=np.bincount(x[:, -1].astype(int), minlength=3).tolist(), train_joint_labels=joint,
    )
    historical = dict(identity, include_test=False)
    historical["data_hash"] = audit.digest(historical)
    frozen = dict(identity)
    historical_file = tmp_path / "historical.json"
    audit.write(historical_file, {dataset: historical})
    audited = dict(identity, cache=str(cache), test_loaded=False, data_hash=historical["data_hash"],
                   train_macro6_cell_support=supports, macro6_active_condition_count=24)
    audit.write(tmp_path / "DATA_AUDIT.json", dict(study_id=audit.STUDY, status="TRAIN_VALIDATION_VERIFIED",
                                                   historical_receipt=str(historical_file), datasets={dataset: audited}))
    binding = dict(data_hash=historical["data_hash"], raw_sha256=identity["raw_sha256"],
                   train_row_hash=identity["split_row_id_hashes"]["train"],
                   train_domain_counts=identity["train_domain_counts"],
                   scope_hash=scope_digest(audit.runner._scope_manifest(TinyRouting())))
    return cache, frozen, binding, x, y, ids


def test_train_only_cache_and_content_identity(tmp_path, monkeypatch):
    cache, frozen, binding, x, y, ids = fixture_train(tmp_path, "movielens")
    accessed = []
    original = np.load

    def load(path, **kwargs):
        accessed.append(Path(path).name)
        assert kwargs == dict(mmap_mode="r", allow_pickle=False)
        return original(path, **kwargs)

    monkeypatch.setattr(audit.np, "load", load)
    bundle = audit.load_train(tmp_path, "movielens", frozen, binding)
    assert all(not a.flags.writeable for a in (bundle.train_x, bundle.train_y, bundle.train_row_ids))
    assert accessed == ["train_x.npy", "train_y.npy", "train_row_ids.npy"]
    altered = ids.copy()
    altered[[0, 1]] = altered[[1, 0]]
    np.save(cache / "train_row_ids.npy", altered)
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.load_train(tmp_path, "movielens", frozen, binding)


@pytest.fixture
def completed_pair(tmp_path, monkeypatch):
    dataset, seed = "movielens", 2022
    cache, frozen, binding, x, y, ids = fixture_train(tmp_path, dataset)
    jobs = audit.runner.stage_jobs("A1")
    controls = [job for job in jobs if job.dataset == dataset and job.seed == seed and job.arm in audit.ARMS]
    roster = dict(datasets={dataset: binding}, source=dict(source_hash="source", source_manifest="manifest",
                   ga31="ga31", runtime=dict(a_step="A", macro_pair_sampler="pairs")),
                  jobs=[dict(job.record(), config_hash=job.config_hash) for job in jobs])
    audit.write(tmp_path / "A1" / "ROSTER.json", roster)
    monkeypatch.setattr(audit.runner, "freeze_roster", lambda artifacts, stage, *, create: (
        (roster, jobs) if (artifacts, stage, create) == (tmp_path, "A1", False) else pytest.fail("Read-only roster")))
    monkeypatch.setattr(audit.v3, "load_protocol", lambda: dict(datasets={dataset: dict(source=frozen)}))
    monkeypatch.setattr(audit.v3, "make_model", lambda bundle, cfg: TinyRouting())
    fingerprint = frozen["split_row_fingerprints"]["train"]
    sampler = MacroPairSampler(dataset, x[:, -1], y, ids, fingerprint)
    segments, journals = [], {}
    try:
        for job, best in zip(controls, (12, 13)):
            journal = audit.runner.job_journal(tmp_path, roster, job)
            journals[job.arm] = journal
            earlier = journal.new_attempt(0)
            writer = journal.new_attempt(1)
            required = {}
            for key, epoch, factor in (("fixed_ema-E05", 5, .5), ("fixed_ema-E10", 10, 1.),
                                       ("fixed_ema-selected", best, -1.)):
                model = TinyRouting()
                with torch.no_grad():
                    for tower in model.tower:
                        tower.weight.mul_(factor)
                r = journal.commit_model(writer, key, "fixed_ema", epoch, model.state_dict(), epoch)
                segments.append(r["name"])
                required[key] = audit.runner._receipt_fields(r)
            chosen = required["fixed_ema-selected"]["model_hash"]
            best_scores, chosen_scores, endpoints = {}, None, []
            for epoch in range(1, 51):
                val = .8 if epoch in (best, 20) else .6
                aucs, losses = [val] * 6, [.4 + epoch / 1000] * 6
                scores = {f"{c}_auc": auc for c, auc in zip(audit.CELLS, aucs)}
                scores.update({f"{c}_logloss": loss for c, loss in zip(audit.CELLS, losses)})
                scores.update(macro_auc=float(np.mean(aucs)), macro_logloss=float(np.mean(losses)))
                if not best_scores or scores["macro_auc"] > best_scores["macro_auc"]:
                    best_scores, selected = scores, epoch
                if epoch in (10, best):
                    checkpoint = "E10" if epoch == 10 else "selected"
                    model_hash = required[f"fixed_ema-{checkpoint}"]["model_hash"]
                    endpoints.append(audit.runner._endpoint(job, journal.binding, "fixed_ema", checkpoint, epoch,
                                                            model_hash, scores))
                if epoch == best:
                    chosen_scores = scores
                identity = dict(journal.binding, storage_generation=writer["generation"], attempt=writer["id"],
                                gpu=writer["gpu"], kind="epoch", epoch=epoch, global_step=epoch)
                audit.write(journal.directory / f"epoch_{epoch:02d}.json",
                            dict(epoch=epoch, global_step=epoch, rows=len(x), validation=dict(fixed_ema=scores),
                                 selected_epochs=dict(fixed_ema=selected),
                                 selected_model_hashes=dict(fixed_ema=chosen if epoch >= best else "first"),
                                 selected_metrics=dict(fixed_ema=best_scores), ram_receipt=dict(identity=identity)))
            assert best_scores == chosen_scores
            complete = dict(job=job.record(), binding=journal.binding, epoch=50, global_step=50, test_accesses=0,
                            selected=dict(fixed_ema=best), compact_models=required, endpoints=endpoints)
            metadata = audit.read(journal.path)
            metadata.update(acknowledged=50, complete=dict(result=complete, required=required))
            audit.write(journal.path, metadata)
            audit.write(journal.directory / "COMPLETE_VERIFIED.json", complete)
            for attempt, through in ((earlier, 6), (writer, best)):
                trace = journal.directory / "attempts" / attempt["id"] / "batches.jsonl"
                trace.parent.mkdir(parents=True)
                for step in range(1, through + 1):
                    context = dict(dataset=dataset, seed=seed, epoch=step, global_step=step, batch=1)
                    drawn = sampler.sample_macro6(context)
                    audit.runner.append(trace, dict(context=context, diagnostics=dict(
                        train_identity=fingerprint, condition=drawn["condition"], pair_index_hash=drawn["pair_index_hash"])))
        yield SimpleNamespace(journals=journals, controls=controls, roster=roster)
    finally:
        for name in segments:
            ram_state._unlink(name)


def test_full_pipeline_on_two_controls_and_read_only_sources(tmp_path, completed_pair, monkeypatch):
    files = [p for p in (tmp_path / "A1").rglob("*") if p.is_file()]
    before = {p: audit.sha(p) for p in files}
    readbacks = []
    original = ram_state.Journal.read_model

    def read_model(journal, key, *, expected):
        readbacks.append((journal.binding["arm"], key, expected))
        return original(journal, key, expected=expected)

    monkeypatch.setattr(ram_state.Journal, "read_model", read_model)
    assert audit.run(tmp_path) == "INPUT_PENDING"
    output = tmp_path / "C0"
    status = audit.read(output / "STATUS.json")
    assert len(status["completed_jobs"]) == 2 and len(status["input_inventory"]) == 20
    assert not status["complete_input_accounting"] and status["test_accesses"] == 0
    assert {p: audit.sha(p) for p in files} == before
    assert set(p for p in (tmp_path / "A1").rglob("*") if p.is_file()) == set(files)
    assert {key for _, key, _ in readbacks} == {"fixed_ema-E05", "fixed_ema-E10", "fixed_ema-selected"}
    assert len(readbacks) == 6
    assert all(r["epoch"] == r["global_step"] and len(r["model_hash"]) == 64 for _, _, r in readbacks)
    manifest = audit.read(output / "MANIFEST.json")
    assert all({*r["ram"]} == {*audit.SNAPSHOTS} for r in manifest["jobs"].values())
    assert all(any(len(stamp["agreeing_attempts"]) == 2 for stamp in r["trace"]) for r in manifest["jobs"].values())
    assert manifest["source_sha256"] == audit.sha(Path(audit.__file__))
    candidates = manifest["candidates"]["movielens_2022"]
    assert audit.sha(Path(candidates["path"])) == candidates["sha256"]
    for job in status["completed_jobs"]:
        with np.load(output / "margins" / f"{job}.npz", allow_pickle=False) as array:
            assert array["candidate_sha256"].item() == candidates["sha256"]
            assert all(array[label].dtype == np.float64 for label in audit.SNAPSHOTS)
    summary = audit.read(output / "SUMMARY.json")
    relevant = next(r for r in summary["groups"] if (r["dataset"], r["window"], r["cell"]) ==
                    ("movielens", "E10_to_selected", "macro"))
    assert relevant["paired_P0_minus_B0"]["per_seed"][0]["horizons"] == dict(B0=2, P0=3)
    assert relevant["paired_P0_minus_B0"]["statistics"]["strict_reversal_rate"]["paired_t_95_interval"] is None
    cost = audit.read(output / "COST.json")
    assert cost["replay"]["movielens_2022"]["through_epoch"] == 13
    assert cost["replay"]["movielens_2022"]["draws"] == 13 * audit.PAIR_COUNT
    assert cost["numeric_output_bytes"] > 0
    assert all(row["ram_bytes"] > 0 and row["forwards"] == 3 for j, row in cost["jobs"].items()
               if j in status["completed_jobs"])
    assert math.isclose(audit.distribution([.1, .2], interval=True)["mean"], .15)


def test_corrupt_snapshot_and_retry_are_technical_failures(tmp_path, completed_pair):
    b0, p0 = (completed_pair.journals[arm] for arm in audit.ARMS)
    meta = audit.read(b0.path)
    snapshot = meta["compact"]["fixed_ema-E05"]
    snapshot["identity"]["model_hash"] = "bad" * 20 + "bad!"
    audit.write(b0.path, meta)
    earlier = audit.read(p0.path)["attempts"][0]["id"]
    path = p0.directory / "attempts" / earlier / "batches.jsonl"
    path.write_text(path.read_text().replace('"pair_index_hash": "', '"pair_index_hash": "wrong'))
    assert audit.run(tmp_path) == "TECHNICAL_INPUT_FAILURE"
    status = audit.read(tmp_path / "C0" / "STATUS.json")
    assert status["input_inventory"][b0.binding["job_key"]]["reason"].endswith("expected model identity")
    assert "trace mismatch" in status["input_inventory"][p0.binding["job_key"]]["reason"]
    assert audit.read(tmp_path / "C0" / "PAIR_METRICS.json") == []


def test_selected_at_snapshot_reads_both_keys_and_reuses_only_matching_logits(monkeypatch):
    x, y, ids = tiny_train()
    model = TinyRouting()
    monkeypatch.setattr(audit.v3, "make_model", lambda *_: TinyRouting())
    identity = dict(scope_hash=scope_digest(audit.runner._scope_manifest(model)))
    train = SimpleNamespace(train_x=x, identity=identity)
    model_state = model.state_dict()
    with torch.no_grad():
        model.tower[0].bias.add_(1)
    other_state = TinyRouting().state_dict()
    payloads = {"E05": other_state, "E10": model_state, "selected": model_state}
    readbacks = []

    def read_model(key, *, expected):
        label = key.split("-", 1)[1]
        payload = payloads[label]
        assert expected["model_hash"] == audit.v3.state_digest(payload)
        readbacks.append(label)
        return payload, dict(identity=dict(expected), size=1)

    snapshots = {label: dict(key=f"fixed_ema-{label}",
                             expected=dict(epoch=5 if label == "E05" else 10,
                                           model_hash=audit.v3.state_digest(state)))
                 for label, state in payloads.items()}
    prepared = dict(selected_epoch=10, snapshots=snapshots, journal=SimpleNamespace(read_model=read_model))
    sampler = MacroPairSampler("movielens", x[:, -1], y, ids, "fixture")
    candidates = audit.candidate_set(sampler, 2022)
    job = SimpleNamespace(seed=2022)
    cost = {}
    margins, _ = audit.score_job(job, prepared, train, {}, candidates, cost, {})
    assert readbacks == list(audit.SNAPSHOTS)
    assert cost["forwards"] == 2 and cost["per_model"][-1]["reused_logits"]
    for key in candidates:
        np.testing.assert_array_equal(margins["E10"][key], margins["selected"][key])
    payloads["selected"] = other_state
    snapshots["selected"]["expected"]["model_hash"] = audit.v3.state_digest(other_state)
    with pytest.raises(ValueError, match="Selected-at-snapshot E10 must match model hash and margins"):
        audit.score_job(job, prepared, train, {}, candidates, {}, {})


def test_stop_after_completed_job_preserves_the_atomic_unit(tmp_path, completed_pair):
    b0, p0 = (completed_pair.journals[arm].binding["job_key"] for arm in audit.ARMS)
    output = tmp_path / "C0"
    path = output / "jobs" / f"{b0}.json"
    marker = dict(status="USER_STOPPED", signal="fixture")
    assert audit.run(tmp_path, lambda: marker if path.exists() else None) == "USER_STOPPED"
    status = audit.read(output / "STATUS.json")
    assert status["completed_jobs"] == [b0]
    assert status["input_inventory"][p0]["state"] == "STOPPED_UNINSPECTED"
    unit = audit.read(path)
    assert audit.sha(Path(unit["margins"]["path"])) == unit["margins"]["sha256"]
    assert not (output / "margins" / f"{p0}.npz").exists()
