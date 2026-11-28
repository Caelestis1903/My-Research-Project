"""Foreground C0 audit of retained A1 fixed EMA on registered Train pairs."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import pickle
import platform
import resource
import signal
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import t as student_t
import torch

from experiments.conditional_history_shared_update_20261122.protocol import digest, read, sha, v3, write
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import CONDITIONS, forward_logits, rng
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.data import joint_counts

from . import runner, scheduler
from .macro_pair_sampler import MacroPairSampler, PAIR_COUNT, ordered_ids_hash
from .protocol import DATASETS, ROOT, SEEDS, STUDY, scope_digest, source

ARMS = ("B0", "P0")
SNAPSHOTS = ("E05", "E10", "selected")
WINDOWS = ("E10_to_selected", "E05_to_selected", "E05_to_E10")
CELLS = tuple(f"d{d}t{t}" for d in range(3) for t in range(2))
CAP = 4096
BATCH = 40960
BAND = .001
CLIP = 5
BINS = ((0, .1), (.1, 1), (1, 2), (2, 5))
QUANTILES = (0, .25, .5, .75, 1)
INPUT_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, EOFError,
                AssertionError, pickle.UnpicklingError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_train(artifacts, dataset, frozen, identity):
    audit = read(artifacts / "DATA_AUDIT.json")
    require((audit["study_id"], audit["status"]) == (STUDY, "TRAIN_VALIDATION_VERIFIED"), "Data audit identity")
    expected = audit["datasets"][dataset]
    cache = Path(expected["cache"])
    historical_path = Path(audit["historical_receipt"])
    require(cache.is_absolute() and historical_path.is_absolute() and expected["test_loaded"] is False,
            "Train cache provenance")
    historical = read(historical_path)[dataset]
    features = source.FeatureSchema(**read(cache / "features.json"))
    require(features.dataset == dataset and features.columns[-1] == features.domain_column == "domain_indicator",
            "Train feature schema identity")
    x, y, rows = (
        np.load(cache / f"train_{name}.npy", mmap_mode="r", allow_pickle=False) for name in ("x", "y", "row_ids")
    )
    require(x.dtype == y.dtype == np.float32 and rows.dtype == np.int64 and rows.ndim == 1
            and x.shape == (len(rows), len(features.columns)) and y.shape == (len(rows), 2)
            and not any(a.flags.writeable for a in (x, y, rows)) and np.isin(y, (0, 1)).all()
            and np.isin(x[:, -1], (0, 1, 2)).all() and len(np.unique(rows)) == len(rows),
            "Train array shape, dtype, labels, domains or IDs")
    domain = x[:, -1].astype(np.int64)
    joint = joint_counts(x, y)
    support = {}
    for d in range(3):
        for t in range(2):
            labels = joint[str(d)]
            positive = sum(n for key, n in labels.items() if key[t] == "1")
            active = sum(bool(labels[f"{a:02b}"] and labels[f"{b:02b}"])
                         for dd, tt, a, b in CONDITIONS if (dd, tt) == (d, t))
            support[f"d{d}t{t}"] = dict(positive=positive, negative=sum(labels.values()) - positive,
                                        active_conditions=active)
    projection = dict(dataset=dataset, split_id="original", preprocessing_fit="train_only",
                      raw_sha256=frozen["raw_sha256"], features=features.as_dict(),
                      encoder_hash=features.encoder_hash, scaler_hash=features.scaler_hash,
                      array_hashes=dict(train_x=source.array_hash(x), train_y=source.array_hash(y)),
                      split_rows=dict(train=len(rows)), split_row_id_hashes=dict(train=source.row_ids_hash(rows)),
                      split_row_fingerprints=dict(train=source.split_row_fingerprint(x, y, rows)),
                      train_domain_counts=np.bincount(domain, minlength=3).tolist(), train_joint_labels=joint,
                      train_macro6_cell_support=support,
                      macro6_active_condition_count=sum(c["active_conditions"] for c in support.values()))
    for reference in (frozen, historical, expected):
        for key, value in projection.items():
            if key in reference:
                observed = reference[key]
                if key in ("array_hashes", "split_rows", "split_row_id_hashes", "split_row_fingerprints"):
                    observed = {name: observed[name] for name in value}
                require(value == observed, f"{dataset} Train identity mismatch: {key}")
    require(all(c["positive"] and c["negative"] for c in support.values())
            and historical["include_test"] is False
            and historical["data_hash"] == expected["data_hash"] == identity["data_hash"]
            == digest({k: v for k, v in historical.items() if k not in ("raw_path", "data_hash")})
            and identity["raw_sha256"] == projection["raw_sha256"]
            and identity["train_row_hash"] == projection["split_row_id_hashes"]["train"]
            and identity["train_domain_counts"] == projection["train_domain_counts"],
            "Roster Train data binding")
    projection.update(cache=str(cache), data_hash=identity["data_hash"], scope_hash=identity["scope_hash"],
                      features_sha256=sha(cache / "features.json"), historical_sha256=sha(historical_path))
    return SimpleNamespace(dataset=dataset, features=features, train_x=x, train_y=y, train_row_ids=rows,
                           identity=projection, domain_num=3, task_num=2)


def candidate_set(sampler, seed):
    context = dict(dataset=sampler.dataset, seed=seed, epoch=0, global_step=0)
    candidates = {}
    for condition, w, q in zip(sampler.active, sampler.w, sampler.q):
        d, t, a, b = condition
        positive, negative = sampler._pools[d, a], sampler._pools[d, b]
        universe = len(positive) * len(negative)
        gen = rng(context, f"c0_macro6_candidates|{d}|{t}|{a}|{b}")
        ranks = np.sort(gen.choice(universe, min(universe, CAP), replace=False))
        left, right = positive[ranks // len(negative)], negative[ranks % len(negative)]
        row_hash = ordered_ids_hash(sampler.row_ids[np.r_[left, right]])
        identity = dict(dataset=sampler.dataset, seed=seed, train_fingerprint=sampler.train_identity,
                        condition=condition, ordered_ids_hash=row_hash)
        candidates[condition] = dict(ranks=ranks, left=left, right=right, universe=universe,
                                     w=float(w), q=float(q), ordered_ids_hash=row_hash, hash=digest(identity))
    return candidates


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    try:
        with stream:
            np.savez_compressed(stream, **arrays)
        os.replace(stream.name, path)
    finally:
        Path(stream.name).unlink(missing_ok=True)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def save_candidates(output, sampler, seed, candidates):
    path = output / "candidates" / f"{sampler.dataset}_{seed}.npz"
    return save_arrays(
        path, condition_ids=np.concatenate([np.tile(key, (len(c["ranks"]), 1)) for key, c in candidates.items()]),
        positive_ids=np.concatenate([sampler.row_ids[c["left"]] for c in candidates.values()]),
        negative_ids=np.concatenate([sampler.row_ids[c["right"]] for c in candidates.values()]),
        condition_hashes=np.array([c["hash"] for c in candidates.values()]),
    )


def validation_records(journal, metadata, result, n_train, steps):
    epochs = {}
    require(metadata["acknowledged"] == result["epoch"] == 50 and result["global_step"] == 50 * steps,
            "A1 completion epoch or logical step")
    best = 0
    for epoch in range(1, 51):
        record = read(journal.directory / f"epoch_{epoch:02d}.json")
        receipt = record["ram_receipt"]
        identity = receipt["identity"]
        require(record["epoch"] == identity["epoch"] == epoch
                and identity["kind"] == "epoch" and record["rows"] == n_train
                and record["global_step"] == identity["global_step"] == epoch * steps
                and all(identity.get(k) == v for k, v in journal._expected(metadata, receipt).items()),
                f"Unverified committed epoch identity E{epoch:02d}")
        scores = record["validation"]["fixed_ema"]
        aucs = [scores[f"{cell}_auc"] for cell in CELLS]
        require(all(math.isfinite(x) for x in aucs) and scores["macro_auc"] == float(np.mean(aucs)),
                f"Undefined fixed EMA Validation AUC E{epoch:02d}")
        epochs[epoch] = record
        if best == 0 or scores["macro_auc"] > epochs[best]["validation"]["fixed_ema"]["macro_auc"]:
            best = epoch
    final = epochs[50]
    chosen = final["selected_model_hashes"]["fixed_ema"]
    require(result["selected"]["fixed_ema"] == final["selected_epochs"]["fixed_ema"] == best,
            "Fixed EMA selection differs from earliest exact Validation maximum")
    require(final["selected_metrics"]["fixed_ema"] == epochs[best]["validation"]["fixed_ema"],
            "Final fixed EMA selection metrics mismatch")
    snapshots = {}
    for label, epoch in zip(SNAPSHOTS, (5, 10, best)):
        record = epochs[epoch]
        scores = record["validation"]["fixed_ema"]
        require(all(math.isfinite(scores[f"{cell}_logloss"]) for cell in (*CELLS, "macro")),
                f"Undefined fixed EMA Validation LogLoss E{epoch:02d}")
        key = f"fixed_ema-{label}"
        model_hash = result["compact_models"][key]["model_hash"]
        require(isinstance(model_hash, str) and len(model_hash) == 64, f"Malformed model hash {key}")
        expected = dict(kind="compact_model", readout="fixed_ema", epoch=epoch, model_hash=model_hash)
        require(result["compact_models"][key] == expected, f"Completion has wrong snapshot identity {key}")
        if record["selected_epochs"]["fixed_ema"] == epoch:
            require(record["selected_model_hashes"]["fixed_ema"] == model_hash,
                    f"Snapshot {key} differs from the committed same-epoch selection")
        checkpoint = label
        rows = [row for row in result["endpoints"]
                if (row["checkpoint"], row["readout"]) == (checkpoint, "fixed_ema")]
        if label == "selected" or rows:
            require(rows == [dict(journal.binding, split="validation", readout="fixed_ema", checkpoint=checkpoint,
                                  epoch=epoch, model_hash=model_hash, **scores)],
                    f"Validation endpoint mismatch {key}")
        snapshots[label] = dict(key=key, expected=dict(expected, global_step=epoch * steps), scores=scores,
                                epoch_receipt=record["ram_receipt"])
    require(snapshots["selected"]["expected"]["model_hash"] == chosen,
            "Selected snapshot and final selection hash differ")
    return dict(selected_epoch=best, snapshots=snapshots,
                writers={epoch: r["ram_receipt"]["identity"]["attempt"] for epoch, r in epochs.items()},
                attempts=metadata["attempts"], journal=journal)


def prepare_job(artifacts, roster, job, train):
    # Journal construction and model read use the existing directories and journal lock.
    directory = artifacts / "A1" / "runs" / job.key
    completion = directory / "COMPLETE_VERIFIED.json"
    if not completion.exists():
        return None
    require((directory / "JOURNAL.lock").is_file(), "Completed A1 job has no existing journal lock")
    journal = runner.job_journal(artifacts, roster, job)
    result = read(completion)
    metadata = read(journal.path)
    require(result["job"] == job.record() and result["binding"] == metadata["binding"] == journal.binding
            and type(result["test_accesses"]) is int and result["test_accesses"] == 0,
            "A1 job completion binding or Test accounting mismatch")
    require(metadata["complete"] == dict(result=result, required=result["compact_models"]),
            "A1 completion differs from the RAM journal")
    require(job.config["batch_size"] == BATCH and job.config["epochs"] == 50, "A1 schedule mismatch")
    for label in SNAPSHOTS:
        key = f"fixed_ema-{label}"
        require(key in result["compact_models"] and key in metadata["compact"], f"Missing compact model {key}")
    n_train = len(train.train_y)
    return validation_records(journal, metadata, result, n_train, math.ceil(n_train / BATCH))


def transition_windows(selected, steps):
    windows = {}
    for label, start, stop in (("E10_to_selected", 10, selected), ("E05_to_selected", 5, selected),
                               ("E05_to_E10", 5, 10)):
        if label == "E05_to_E10":
            state = "FORMATION" if selected >= 10 else "POST_SELECTED_FORMATION"
        else:
            state = ("PRE_SELECTED" if stop > start else
                     "SELECTED_AT_SNAPSHOT" if stop == start else "POST_SELECTED_SNAPSHOT")
        windows[label] = dict(classification=state, eligible=state in ("PRE_SELECTED", "FORMATION"),
                              start_epoch=start, end_epoch=stop, horizon_epochs=stop - start,
                              start_exclusive=start * steps, end_inclusive=stop * steps)
    return windows


def score_logits(model, train_x, endpoints):
    blocks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(endpoints), BATCH):
            x = torch.as_tensor(train_x[endpoints[start:start + BATCH]], device="cpu").T
            _, logits = forward_logits(model, x)
            require(logits.dtype == torch.float32 and logits.shape == (x.shape[1], 2)
                    and bool(torch.isfinite(logits).all()), "Invalid FP32 routed Train logits")
            blocks.append(logits.numpy())
    return np.concatenate(blocks)


def candidate_margins(logits, endpoints, candidates):
    return {key: logits[np.searchsorted(endpoints, c["left"]), key[1]].astype(np.float64)
                 - logits[np.searchsorted(endpoints, c["right"]), key[1]].astype(np.float64)
            for key, c in candidates.items()}


def setup(seed):
    torch.set_default_dtype(torch.float32)
    v3.setup(seed, torch.device("cpu"))
    return dict(seed=seed, device="cpu", dtype=str(torch.get_default_dtype()), threads=torch.get_num_threads(),
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark)


def score_job(job, prepared, train, frozen, candidates, cost, receipts):
    settings = setup(job.seed)
    model = v3.make_model(train, frozen).requires_grad_(False)
    require(scope_digest(runner._scope_manifest(model)) == train.identity["scope_hash"]
            and all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters()),
            "Frozen M3oE parameter scope or FP32/CPU mismatch")
    endpoints = np.unique(np.concatenate([a for c in candidates.values() for a in (c["left"], c["right"])]))
    logits, margins = {}, {}
    cost.update(unique_endpoints=len(endpoints), candidates=sum(len(c["ranks"]) for c in candidates.values()),
                ram_read_verify_decode_seconds=0., inference_seconds=0., ram_bytes=0, forwards=0, train_rows=0,
                per_model=[])
    for label, snapshot in prepared["snapshots"].items():
        expected = snapshot["expected"]
        before = time.perf_counter()
        payload, receipt = prepared["journal"].read_model(snapshot["key"], expected=expected)
        seconds = time.perf_counter() - before
        model_hash = expected["model_hash"]
        cost["ram_bytes"] += receipt["size"]
        cost["ram_read_verify_decode_seconds"] += seconds
        receipts[label] = dict(receipt, read_verify_decode_seconds=seconds)
        row = dict(snapshot=label, model_hash=model_hash, ram_bytes=receipt["size"], read_verify_decode_seconds=seconds,
                   forwards=0, train_rows=0, inference_seconds=0., reused_logits=model_hash in logits)
        if model_hash not in logits:
            require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in payload.values()),
                    f"Snapshot {snapshot['key']} has a non-FP32 floating tensor")
            model.load_state_dict(payload, strict=True)
            del payload
            before = time.perf_counter()
            logits[model_hash] = score_logits(model, train.train_x, endpoints)
            row.update(inference_seconds=time.perf_counter() - before, forwards=math.ceil(len(endpoints) / BATCH),
                       train_rows=len(endpoints))
            for field in ("inference_seconds", "forwards", "train_rows"):
                cost[field] += row[field]
        else:
            del payload
        cost["per_model"].append(row)
        margins[label] = candidate_margins(logits[model_hash], endpoints, candidates)
    for label in SNAPSHOTS[:2]:
        same_epoch = prepared["snapshots"][label]["expected"]["epoch"] == prepared["selected_epoch"]
        if same_epoch:
            require(receipts[label]["identity"]["model_hash"] == receipts["selected"]["identity"]["model_hash"]
                    and all(np.array_equal(margins[label][key], margins["selected"][key]) for key in candidates),
                    f"Selected-at-snapshot {label} must match model hash and margins")
    return margins, settings


def replay(sampler, seed, steps, last_epoch, boundaries):
    draws = defaultdict(list)
    stamps = {}
    for step in range(1, steps * last_epoch + 1):
        epoch, batch = divmod(step - 1, steps)
        context = dict(dataset=sampler.dataset, seed=seed, epoch=epoch + 1, global_step=step, batch=batch + 1)
        sampled = sampler.sample_macro6(context)
        key = sampled["condition"]
        d, _, a, b = key
        left, right = sampled["offsets"].reshape(2, PAIR_COUNT)
        positive, negative = sampler._pools[d, a], sampler._pools[d, b]
        ranks = np.searchsorted(positive, left) * len(negative) + np.searchsorted(negative, right)
        draws[key].append((step, ranks))
        if step in boundaries:
            stamps[step] = dict(context=context, condition=list(key), pair_index_hash=sampled["pair_index_hash"])
    return draws, stamps


def boundary_steps(windows):
    return {step for w in windows.values() if w["eligible"]
            for step in (w["start_exclusive"] + 1, w["end_inclusive"])}


def verify_trace(prepared, windows, stamps, train_identity):
    required = boundary_steps(windows)
    if not required:
        return []
    attempts = prepared["attempts"]
    by_id = {attempt["id"]: index for index, attempt in enumerate(attempts)}
    require(len(by_id) == len(attempts), "Duplicate A1 trace attempt identity")
    committed = {step: prepared["writers"][stamps[step]["context"]["epoch"]] for step in required}
    matches = defaultdict(list)
    for index, attempt in enumerate(attempts):
        writer = attempt["id"]
        relevant = {step for step, owner in committed.items() if index <= by_id[owner]}
        path = prepared["journal"].directory / "attempts" / writer / "batches.jsonl"
        if not relevant or not path.exists():
            continue
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    # A killed writer may leave one unpublished trailing fragment.
                    break
                row = json.loads(line)
                context = row["context"]
                step = context["global_step"]
                if step not in relevant:
                    continue
                stamp = stamps[step]
                diagnostics = row["diagnostics"]
                require(all(context.get(k) == v for k, v in stamp["context"].items())
                        and diagnostics["train_identity"] == train_identity
                        and diagnostics["condition"] == stamp["condition"]
                        and diagnostics["pair_index_hash"] == stamp["pair_index_hash"]
                        and writer not in matches[step], f"Current sampler trace mismatch at logical step {step}")
                matches[step].append(writer)
    for step, owner in committed.items():
        require(owner in matches[step], f"Committed A1 writer trace missing at logical step {step}")
    return [dict(stamps[step], committed_writer=committed[step], agreeing_attempts=matches[step])
            for step in sorted(required)]


def ratio(num, den):
    return float(num / den) if den else None


def stats(values):
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                quantiles=np.quantile(values, QUANTILES).tolist() if len(values) else None)


def checkpoint_metrics(margin):
    pos, tie, neg = (int(sign.sum()) for sign in (margin > 0, margin == 0, margin < 0))
    return dict(candidates=len(margin), positive=pos, zero=tie, negative=neg,
                near_zero=int((np.abs(margin) <= BAND).sum()), strict_correct_rate=ratio(pos, len(margin)),
                concordance=ratio(pos + .5 * tie, len(margin)), margin=stats(margin))


def transitions(before, after, confidence):
    correct = before > 0
    flip, tie, retained = (correct & mask for mask in (after < 0, after == 0, after > 0))
    groups = dict(candidates=np.ones(len(before), dtype=bool), historical_correct=correct,
                  strict_reversal=flip, tie_loss=tie, retained=retained)
    out = {name: int(mask.sum()) for name, mask in groups.items()}
    out.update(loss_of_strict_correctness=out["strict_reversal"] + out["tie_loss"],
               robust_strict_reversal=int(((before >= BAND) & (after <= -BAND)).sum()),
               near_zero_historical=int((np.abs(before) <= BAND).sum()),
               near_zero_endpoint=int((np.abs(after) <= BAND).sum()),
               margin_groups={name: dict(historical=stats(before[mask]), endpoint=stats(after[mask]),
                                         change=stats(after[mask] - before[mask]))
                              for name, mask in groups.items() if name in ("strict_reversal", "tie_loss", "retained")})
    if confidence:
        clipped = np.clip(before, -CLIP, CLIP)
        bins = []
        for lo, hi in BINS:
            name = f"e10_confidence_{lo}_{hi}"
            mask = groups[name] = correct & (clipped > lo) & (clipped <= hi)
            n, r = int(mask.sum()), int((mask & flip).sum())
            bins.append(dict(subset=name, lower_exclusive=lo, upper_inclusive=hi, count=n, strict_reversal=r,
                             strict_reversal_rate=ratio(r, n), endpoint_margin=stats(after[mask])))
        out["confidence"] = dict(temperature=1, clip=[-CLIP, CLIP], above_clip=int((before > CLIP).sum()),
                                 at_clip=int((before == CLIP).sum()), bins=bins)
    return out, groups


def exposure(candidate, all_draws, groups, window):
    draws = [r for step, r in all_draws if window["start_exclusive"] < step <= window["end_inclusive"]]
    drawn = len(draws) * PAIR_COUNT
    frequencies = np.zeros(len(candidate["ranks"]), dtype=np.int64)
    distinct = 0
    if draws:
        ranks, counts = np.unique(np.concatenate(draws), return_counts=True)
        distinct = len(ranks)
        indices = np.searchsorted(ranks, candidate["ranks"])
        present = (indices < distinct) & (ranks[np.minimum(indices, distinct - 1)] == candidate["ranks"])
        frequencies[present] = counts[indices[present]]
    u = candidate["universe"]
    uniform_coverage = (0. if not drawn else 1. if u == 1 else -math.expm1(drawn * math.log1p(-1 / u)))
    subsets = {}
    for name, mask in groups.items():
        n, seen, hits = int(mask.sum()), int(np.count_nonzero(frequencies[mask])), int(frequencies[mask].sum())
        subsets[name] = dict(members=n, observed_members=seen, draw_hits=hits, coverage=ratio(seen, n),
                             draw_share=ratio(hits, drawn), exposures_per_candidate=ratio(hits, n),
                             uniform_expected_hits=drawn * n / u,
                             uniform_expected_coverage=uniform_coverage if n else None,
                             empty_subset_reason=None if n else "EMPTY_CANDIDATE_SUBSET",
                             no_draws_reason=None if drawn else "NO_CURRENT_DRAWS")
    return dict(draws=drawn, events=len(draws), distinct_pairs=distinct, repeated_draws=drawn - distinct,
                universe=u, subsets=subsets)


TRANSITION_COUNTS = ("historical_correct", "strict_reversal", "tie_loss", "retained",
                     "loss_of_strict_correctness", "robust_strict_reversal", "near_zero_historical",
                     "near_zero_endpoint")
CHECKPOINT_COUNTS = ("positive", "zero", "negative", "near_zero")


def target_rates(rows, numerator, denominator):
    mass = sum(row["w"] for row in rows)
    num = sum(row["w"] * numerator(row) / row["candidates"] for row in rows) / mass
    den = sum(row["w"] * denominator(row) / row["candidates"] for row in rows) / mass
    return dict(value=ratio(num, den), weighted_numerator=num, weighted_denominator=den,
                reason=None if den else "EMPTY_CONDITIONING_SET")


def pair_group(rows, transition):
    names = TRANSITION_COUNTS if transition else CHECKPOINT_COUNTS
    counts = {name: sum(row[name] for row in rows) for name in ("candidates", *names)}
    if transition:
        rates = {f"{name}_rate": target_rates(rows, lambda r: r[name], lambda r: r["historical_correct"])
                 for name in ("strict_reversal", "tie_loss", "retained", "loss_of_strict_correctness",
                              "robust_strict_reversal")}
    else:
        rates = dict(strict_correct_rate=target_rates(rows, lambda r: r["positive"], lambda r: r["candidates"]),
                     concordance=target_rates(rows, lambda r: r["positive"] + .5 * r["zero"],
                                              lambda r: r["candidates"]))
    return dict(counts=counts, rates=rates)


def overlap_group(rows):
    subsets = {}
    for name in rows[0]["subsets"]:
        total = {field: sum(r["subsets"][name][field] for r in rows)
                 for field in ("members", "observed_members", "draw_hits", "uniform_expected_hits")}

        def get(row, field):
            return row["subsets"][name][field]

        total.update(
            coverage=target_rates(rows, lambda r: get(r, "observed_members"), lambda r: get(r, "members")),
            exposures_per_candidate=target_rates(rows, lambda r: get(r, "draw_hits"), lambda r: get(r, "members")),
            uniform_expected_coverage=target_rates(
                rows, lambda r: (get(r, "uniform_expected_coverage") or 0) * get(r, "members"),
                lambda r: get(r, "members")),
            draw_share=ratio(total["draw_hits"], sum(r["draws"] for r in rows)),
        )
        subsets[name] = total
    return dict({field: sum(r[field] for r in rows)
                 for field in ("draws", "events", "distinct_pairs", "repeated_draws", "universe")},
                subsets=subsets)


def by_cell(rows, reducer):
    return dict({cell: reducer([r for r in rows if r["cell"] == cell]) for cell in CELLS}, macro=reducer(rows))


def measure(candidates, margins, windows, draws):
    description = {key: dict(condition=key, cell=f"d{key[0]}t{key[1]}", w=c["w"], q=c["q"],
                             candidates=len(c["ranks"]), universe=c["universe"], candidate_hash=c["hash"],
                             ordered_ids_hash=c["ordered_ids_hash"]) for key, c in candidates.items()}
    checkpoints = {}
    for label in SNAPSHOTS:
        rows = [dict(description[key], **checkpoint_metrics(margins[label][key])) for key in candidates]
        checkpoints[label] = dict(conditions=rows, groups=by_cell(rows, lambda rs: pair_group(rs, False)))

    paired, overlap = {}, {}
    for label, window in windows.items():
        paired[label] = dict(window, conditions=None, groups=None)
        overlap[label] = dict(window, conditions=None, groups=None)
        if not window["eligible"]:
            continue
        left, right = label.split("_to_")
        changes, current = [], []
        for key, candidate in candidates.items():
            out, subsets = transitions(margins[left][key], margins[right][key], left == "E10")
            changes.append(dict(description[key], **out))
            current.append(dict(description[key], **exposure(candidate, draws.get(key, ()), subsets, window)))
        for row in changes:
            row["rates"] = pair_group([row], True)["rates"]
        paired[label].update(conditions=changes, groups=by_cell(changes, lambda rs: pair_group(rs, True)))
        overlap[label].update(conditions=current, groups=by_cell(current, overlap_group))
    return dict(checkpoints=checkpoints, windows=paired), overlap


def join_validation(train, prepared, models, metrics, overlap):
    snapshots = {label: dict(epoch=row["expected"]["epoch"], global_step=row["expected"]["global_step"],
                             model_hash=models[label]["identity"]["model_hash"], scores=row["scores"],
                             committed_epoch_receipt=row["epoch_receipt"])
                 for label, row in prepared["snapshots"].items()}
    population = train.identity["train_macro6_cell_support"]
    aligned = []
    for name, window in metrics["windows"].items():
        start, end = name.split("_to_")
        original, endpoint, selected = (snapshots[label]["scores"] for label in (start, end, "selected"))
        for cell in (*CELLS, "macro"):
            support = population[cell] if cell in population else None
            aligned.append(dict(
                window=name, cell=cell, classification=window["classification"], eligible=window["eligible"],
                horizon_epochs=window["horizon_epochs"], start_epoch=window["start_epoch"],
                endpoint_epoch=window["end_epoch"], selected_epoch=prepared["selected_epoch"],
                historical_auc=original[f"{cell}_auc"], historical_logloss=original[f"{cell}_logloss"],
                endpoint_auc=endpoint[f"{cell}_auc"], endpoint_logloss=endpoint[f"{cell}_logloss"],
                selected_auc=selected[f"{cell}_auc"], selected_logloss=selected[f"{cell}_logloss"],
                selected_auc_change=selected[f"{cell}_auc"] - original[f"{cell}_auc"],
                endpoint_auc_change=endpoint[f"{cell}_auc"] - original[f"{cell}_auc"],
                train_rows=support["positive"] + support["negative"] if support else len(train.train_y),
                train_cell_support=support,
                pair=window["groups"][cell] if window["eligible"] else None,
                exposure=overlap[name]["groups"][cell] if window["eligible"] else None,
            ))
    return dict(snapshots=snapshots, aligned=aligned)


def distribution(values, interval=False):
    n = len(values)
    mean = float(np.mean(values)) if n else None
    sd = float(np.std(values, ddof=1)) if n >= 2 else None
    out = dict(n=n, mean=mean, sample_sd=sd)
    if interval:
        half = float(student_t.ppf(.975, n - 1) * sd / math.sqrt(n)) if n >= 2 else None
        out["paired_t_95_interval"] = [mean - half, mean + half] if half is not None else None
    return out


def summarize(rows):
    summaries = []
    for dataset in DATASETS:
        for name in WINDOWS:
            for cell in (*CELLS, "macro"):
                summaries.append(summarize_group(rows, dataset, name, cell))
    return summaries


def summarize_group(rows, dataset, name, cell):
    eligible = {arm: {} for arm in ARMS}
    for row in rows:
        if row["dataset"] != dataset:
            continue
        window = row["metrics"]["windows"][name]
        if window["eligible"]:
            counts = window["groups"][cell]["counts"]
            rates = {key: value["value"] for key, value in window["groups"][cell]["rates"].items()}
            eligible[row["arm"]][row["seed"]] = dict(seed=row["seed"], horizon_epochs=window["horizon_epochs"],
                                                     end_epoch=window["end_epoch"], counts=counts, rates=rates,
                                                     selected_epoch=row["validation"]["snapshots"]["selected"]["epoch"])
    names = [f"{key}_rate" for key in ("strict_reversal", "tie_loss", "retained", "loss_of_strict_correctness",
                                       "robust_strict_reversal")]
    matched = []
    for seed in sorted(eligible["B0"].keys() & eligible["P0"].keys()):
        b0, p0 = eligible["B0"][seed], eligible["P0"][seed]
        difference = {}
        for key in names:
            left, right = b0["rates"][key], p0["rates"][key]
            difference[key] = right - left if left is not None and right is not None else None
        matched.append(dict(seed=seed, horizons={arm: eligible[arm][seed]["horizon_epochs"] for arm in ARMS},
                            selected_epochs={arm: eligible[arm][seed]["selected_epoch"] for arm in ARMS},
                            difference=difference))
    return dict(
        dataset=dataset, window=name, cell=cell,
        arms={arm: dict(eligible_seeds=len(group), per_seed=list(group.values()),
                        statistics={key: distribution([row["rates"][key] for row in group.values()
                                                       if row["rates"][key] is not None]) for key in names})
              for arm, group in eligible.items()},
        paired_P0_minus_B0=dict(eligible_seeds=len(matched), per_seed=matched,
                                statistics={key: distribution([row["difference"][key] for row in matched
                                                                if row["difference"][key] is not None], interval=True)
                                            for key in names}),
    )


def exposure_cost(metrics, overlap):
    e10 = metrics["windows"]["E10_to_selected"]
    if not e10["eligible"]:
        return dict(classification=e10["classification"], eligible_relations=None, conditions=None, cells=None)

    def row(pair, sample):
        return dict(events_i=sample["events"], current_draws=sample["draws"], legal_pair_support=sample["universe"],
                    scored_candidates=pair["candidates"], eligible_relations=pair["historical_correct"],
                    current_hits=sample["subsets"]["historical_correct"]["draw_hits"],
                    exposures_per_relation=ratio(sample["subsets"]["historical_correct"]["draw_hits"],
                                                 pair["historical_correct"]))

    current = overlap["E10_to_selected"]
    return dict(
        classification=e10["classification"], eligible_relations=e10["groups"]["macro"]["counts"]["historical_correct"],
        conditions=[dict(condition=p["condition"], **row(p, r)) for p, r in zip(e10["conditions"], current["conditions"])],
        cells={cell: row(e10["groups"][cell]["counts"], current["groups"][cell]) for cell in CELLS},
    )


def save_job(output, job, models, margins, candidates, candidate_receipt, metrics, overlap, validation):
    receipt = save_arrays(
        output / "margins" / f"{job.key}.npz",
        **{label: np.concatenate(list(margins[label].values())) for label in SNAPSHOTS},
        model_hashes=np.array([models[label]["identity"]["model_hash"] for label in SNAPSHOTS]),
        candidate_sha256=np.array(candidate_receipt["sha256"]),
        condition_hashes=np.array([c["hash"] for c in candidates.values()]),
    )
    row = dict(job=job.key, dataset=job.dataset, seed=job.seed, arm=job.arm, margins=receipt,
               candidate=candidate_receipt, metrics=metrics, overlap=overlap, validation=validation)
    write(output / "jobs" / f"{job.key}.json", row)
    return row


def eligible_jobs(jobs):
    jobs = [job for job in jobs if job.arm in ARMS]
    expected = {(dataset, seed, arm) for dataset in DATASETS for seed in SEEDS for arm in ARMS}
    require(len(jobs) == len(expected) and {(j.dataset, j.seed, j.arm) for j in jobs} == expected,
            "A1 B0/P0 roster does not cover the 20 registered trajectories")
    return jobs


def run(artifacts, interrupt=lambda: None):
    start = time.perf_counter()
    output = artifacts / "C0"
    inventory = {f"A1-{dataset}-{arm}-seed{seed}": dict(dataset=dataset, seed=seed, arm=arm, state="UNINSPECTED")
                 for dataset in DATASETS for seed in SEEDS for arm in ARMS}
    manifest = dict(study=STUDY, stage="C0", source_sha256=sha(Path(__file__)), test_accesses=0,
                    design_sha256=sha(ROOT / "research_reports" / STUDY / "C0_EXECUTION_DESIGN.md"),
                    roster=None, inventory=inventory, caches={}, candidates={}, jobs={},
                    settings=dict(device="cpu", dtype="torch.float32", batch=BATCH, candidate_cap=CAP,
                                  pair_draws_per_event=PAIR_COUNT, sensitivity_logit_band=BAND,
                                  teacher_clip=[-CLIP, CLIP], teacher_temperature=1, quantiles=QUANTILES,
                                  seed_schedule=SEEDS, framework="v3.setup"), stop_checks=[])
    cost = dict(jobs={}, replay={}, cache_verification_seconds={}, roster_verification_seconds=0.,
                environment=dict(platform=platform.platform(), cpu=platform.machine(), python=platform.python_version(),
                                 torch=str(torch.__version__), numpy=np.__version__, device="cpu"),
                memory_exposure=dict(draws_per_event=PAIR_COUNT,
                                     conditional_draws_per_entry="2048 * events_i / b_i for assigned b_i > 0"))
    rows = []
    stopped = None

    def check(boundary):
        nonlocal stopped
        stopped = scheduler.stop_status(artifacts) or interrupt() or stopped
        manifest["stop_checks"].append(dict(boundary=boundary, wall_time_unix=time.time(), marker=stopped))
        return stopped

    def disposition(keys, state, reason=None):
        for key in keys:
            inventory[key].update(state=state, reason=reason)

    write(output / "STATUS.json", dict(status="ACTIVE", test_accesses=0))
    if not check("before_inspection"):
        begin = time.perf_counter()
        roster_path = artifacts / "A1" / "ROSTER.json"
        try:
            if not roster_path.exists():
                disposition(inventory, "INPUT_PENDING", "A1 roster has not been published")
            else:
                roster, jobs = runner.freeze_roster(artifacts, "A1", create=False)
                jobs = eligible_jobs(jobs)
                manifest["roster"] = dict(sha256=sha(roster_path), source=roster["source"], datasets=roster["datasets"],
                                          jobs=[record for record in roster["jobs"] if record["key"] in inventory])
        except INPUT_ERRORS as exc:
            disposition(inventory, "TECHNICAL_INPUT_FAILURE", f"{type(exc).__name__}: {exc}")
        cost["roster_verification_seconds"] = time.perf_counter() - begin

    if manifest["roster"] is not None:
        inspect_groups(artifacts, output, roster, jobs, manifest, cost, rows, disposition, check)
    check("before_publication")
    if stopped:
        for item in inventory.values():
            if item["state"] == "UNINSPECTED":
                item.update(state="STOPPED_UNINSPECTED", reason=stopped["status"])
    states = {item["state"] for item in inventory.values()}
    complete = states == {"COMPLETE"}
    status = (stopped["status"] if stopped else "COMPLETE" if complete else
              "TECHNICAL_INPUT_FAILURE" if "TECHNICAL_INPUT_FAILURE" in states else "INPUT_PENDING")
    publish(output, rows, manifest, cost, status, complete, stopped, start)
    return status


def inspect_groups(artifacts, output, roster, jobs, manifest, cost, rows, disposition, check):
    frozen = v3.load_protocol()["datasets"]
    for dataset in DATASETS:
        if check(f"before_cache:{dataset}"):
            break
        group = []
        for job in jobs:
            if job.dataset == dataset:
                completion = artifacts / "A1" / "runs" / job.key / "COMPLETE_VERIFIED.json"
                if completion.exists():
                    group.append(job)
                else:
                    disposition([job.key], "INPUT_PENDING", "A1 COMPLETE_VERIFIED.json has not been published")
        if not group:
            continue
        begin = time.perf_counter()
        try:
            train = load_train(artifacts, dataset, frozen[dataset]["source"], roster["datasets"][dataset])
            fingerprint = train.identity["split_row_fingerprints"]["train"]
            sampler = MacroPairSampler(dataset, train.train_x[:, -1], train.train_y, train.train_row_ids, fingerprint)
        except INPUT_ERRORS as exc:
            disposition((job.key for job in group), "TECHNICAL_INPUT_FAILURE", f"{type(exc).__name__}: {exc}")
            continue
        finally:
            cost["cache_verification_seconds"][dataset] = time.perf_counter() - begin
        manifest["caches"][dataset] = train.identity
        for seed in SEEDS:
            pair = [job for job in group if job.seed == seed]
            if not pair:
                continue
            if check(f"before_seed:{dataset}:{seed}"):
                return
            inspect_pair(artifacts, output, roster, pair, train, frozen[dataset]["source"], sampler,
                         manifest, cost, rows, disposition, check)


def inspect_pair(artifacts, output, roster, pair, train, frozen, sampler, manifest, cost, rows, disposition, check):
    steps = math.ceil(len(train.train_y) / BATCH)
    ready = []
    for job in pair:
        if check(f"before_metadata:{job.key}"):
            return
        begin = time.perf_counter()
        job_cost = cost["jobs"][job.key] = {}
        try:
            prepared = prepare_job(artifacts, roster, job, train)
            if prepared is None:
                disposition([job.key], "INPUT_PENDING", "A1 COMPLETE_VERIFIED.json has not been published")
            else:
                windows = transition_windows(prepared["selected_epoch"], steps)
                ready.append((job, prepared, windows))
        except INPUT_ERRORS as exc:
            disposition([job.key], "TECHNICAL_INPUT_FAILURE", f"{type(exc).__name__}: {exc}")
        finally:
            job_cost["input_verification_seconds"] = time.perf_counter() - begin
    if not ready:
        return
    dataset, seed = pair[0].dataset, pair[0].seed
    group_key = f"{dataset}_{seed}"
    if check(f"before_replay:{group_key}"):
        return
    begin = time.perf_counter()
    candidates = candidate_set(sampler, seed)
    max_epoch = max(prepared["selected_epoch"] for _, prepared, _ in ready)
    boundaries = set().union(*(boundary_steps(windows) for _, _, windows in ready))
    candidate_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    draws, stamps = replay(sampler, seed, steps, max_epoch, boundaries)
    group_cost = cost["replay"][group_key] = dict(
        dataset=dataset, seed=seed, seconds=time.perf_counter() - begin, candidate_seconds=candidate_seconds,
        steps_per_epoch=steps, through_epoch=max_epoch, steps=max_epoch * steps, draws=max_epoch * steps * PAIR_COUNT,
        conditions=[dict(condition=key, events=len(events), draws=len(events) * PAIR_COUNT)
                    for key in candidates for events in (draws.get(key, ()),)],
        candidates=sum(len(c["ranks"]) for c in candidates.values()),
    )
    begin = time.perf_counter()
    candidate_receipt = save_candidates(output, sampler, seed, candidates)
    group_cost["write_seconds"] = time.perf_counter() - begin
    manifest["candidates"][group_key] = dict(candidate_receipt, sampler=sampler.state_dict(),
                                             condition_hashes=[dict(condition=k, hash=c["hash"],
                                                                    ordered_ids_hash=c["ordered_ids_hash"])
                                                               for k, c in candidates.items()])
    for job, prepared, windows in ready:
        if check(f"before_job:{job.key}"):
            return
        start = time.perf_counter()
        job_cost = cost["jobs"][job.key]
        models = {}
        receipt = manifest["jobs"][job.key] = dict(ram=models, selected_epoch=prepared["selected_epoch"],
                                                  eligibility=windows, binding=prepared["journal"].binding)
        try:
            begin = time.perf_counter()
            trace = verify_trace(prepared, windows, stamps, sampler.train_identity)
            job_cost["trace_verification_seconds"] = time.perf_counter() - begin
            receipt["trace"] = trace
            margins, receipt["settings"] = score_job(job, prepared, train, frozen, candidates, job_cost, models)
            begin = time.perf_counter()
            metrics, overlap = measure(candidates, margins, windows, draws)
            validation = join_validation(train, prepared, models, metrics, overlap)
            job_cost["aggregation_seconds"] = time.perf_counter() - begin
            job_cost["e10"] = exposure_cost(metrics, overlap)
            begin = time.perf_counter()
            row = save_job(output, job, models, margins, candidates, candidate_receipt, metrics, overlap, validation)
            job_cost["write_seconds"] = time.perf_counter() - begin
            rows.append(row)
            receipt["margins"] = row["margins"]
            disposition([job.key], "COMPLETE")
        except INPUT_ERRORS as exc:
            disposition([job.key], "TECHNICAL_INPUT_FAILURE", f"{type(exc).__name__}: {exc}")
        finally:
            job_cost["wall_seconds"] = time.perf_counter() - start
            if inventory_cost := job_cost.get("e10"):
                job_cost["wall_seconds_per_eligible_e10_relation"] = ratio(
                    job_cost["wall_seconds"], inventory_cost["eligible_relations"])
                job_cost["wall_seconds_per_scored_candidate"] = ratio(job_cost["wall_seconds"], job_cost["candidates"])


def publish(output, rows, manifest, cost, status, complete, stopped, start):
    begin = time.perf_counter()
    matched_b0 = {(row["dataset"], row["seed"], v["window"], v["cell"]): v["historical_auc"]
                  for row in rows if row["arm"] == "B0" for v in row["validation"]["aligned"]}
    validation = []
    for row in rows:
        raw = row["validation"]
        aligned = [dict(v, matched_B0_historical_auc=matched_b0.get((row["dataset"], row["seed"], v["window"], v["cell"])))
                   for v in raw["aligned"]]
        validation.append(dict({k: row[k] for k in ("job", "dataset", "seed", "arm")},
                               snapshots=raw["snapshots"], aligned=aligned))
    write(output / "VALIDATION.json", validation)
    for field, file in (("metrics", "PAIR_METRICS.json"), ("overlap", "OVERLAP.json")):
        write(output / file, [dict({k: row[k] for k in ("job", "dataset", "seed", "arm")}, **row[field]) for row in rows])
    write(output / "SUMMARY.json", dict(complete_input_accounting=complete, expected_trajectories=len(manifest["inventory"]),
                                        observed_trajectories=len(rows), groups=summarize(rows)))
    write(output / "MANIFEST.json", manifest)
    numeric_paths = ([Path(row["margins"]["path"]) for row in rows]
                     + [Path(row["path"]) for row in manifest["candidates"].values()]
                     + [output / name for name in ("PAIR_METRICS.json", "OVERLAP.json", "VALIDATION.json", "SUMMARY.json")]
                     + [output / "jobs" / f"{row['job']}.json" for row in rows])
    elapsed = time.perf_counter() - start
    scored = sum(cost["jobs"][row["job"]]["candidates"] for row in rows)
    eligible = sum(cost["jobs"][row["job"]]["e10"]["eligible_relations"] or 0 for row in rows)
    cost.update(wall_seconds=elapsed, wall_seconds_per_scored_candidate=ratio(elapsed, scored),
                wall_seconds_per_eligible_e10_relation=ratio(elapsed, eligible),
                aggregate_write_seconds=time.perf_counter() - begin, process_peak_rss_bytes=resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss * 1024, numeric_output_bytes=sum(p.stat().st_size for p in numeric_paths))
    write(output / "COST.json", cost)
    write(output / "STATUS.json", dict(
        status=status, stop=stopped, complete_input_accounting=complete, test_accesses=0,
        completed_jobs=[row["job"] for row in rows], input_inventory=manifest["inventory"],
        stop_checks=manifest["stop_checks"], manifest_sha256=sha(output / "MANIFEST.json"),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    if not args.artifacts.is_absolute():
        parser.error("--artifacts requires an absolute path")
    stopped = None

    def stop(signum, _frame):
        nonlocal stopped
        stopped = dict(status="USER_STOPPED", signal=signal.Signals(signum).name)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    status = run(args.artifacts.resolve(), lambda: stopped)
    print(status)
    return 0 if status in ("COMPLETE", "USER_STOPPED", "INPUT_PENDING") else 1


if __name__ == "__main__":
    raise SystemExit(main())
