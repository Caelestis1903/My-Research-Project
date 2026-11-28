"""C1's frozen Train-only bank and stateless pair draws."""
import hashlib
import io
import math
from pathlib import Path

import numpy as np

from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import rng

from .c0_audit import CAP as CANDIDATE_CAP, CLIP, SNAPSHOTS, candidate_set, require
from .macro_pair_sampler import PAIR_COUNT, ordered_ids_hash
from .protocol import digest

BANK_CAP = 2048
TEACHER_EPOCH = 10
TEMPERATURE = 1.
MODES = {"C2": "random", "C3": "correct", "C4": "correct"}
HISTOGRAM_EDGES = (-CLIP, -2, -1, -.1, 0, .1, 1, 2, CLIP)


def _array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def _archive(receipt):
    payload = Path(receipt["path"]).read_bytes()
    require(len(payload) == receipt["bytes"] and hashlib.sha256(payload).hexdigest() == receipt["sha256"],
            f"C0 archive bytes differ from the verified receipt: {receipt['path']}")
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _same_array(actual, expected):
    return actual.dtype == expected.dtype and np.array_equal(actual, expected)


def load_candidates(sampler, seed, inputs):
    """Recreate the archived candidates and load only their registered P0 E10 targets."""
    receipt = inputs["candidate"]
    candidates = candidate_set(sampler, seed)
    hashes = np.array([c["hash"] for c in candidates.values()])
    condition_rows = [dict(condition=list(k), hash=c["hash"], ordered_ids_hash=c["ordered_ids_hash"])
                      for k, c in candidates.items()]
    require(receipt["sampler"] == sampler.state_dict() and receipt["condition_hashes"] == condition_rows,
            "C0 candidate sampler or condition order mismatch")
    expected = dict(condition_ids=np.concatenate(
        [np.tile(condition, (len(c["ranks"]), 1)) for condition, c in candidates.items()]),
        positive_ids=np.concatenate([sampler.row_ids[c["left"]] for c in candidates.values()]),
        negative_ids=np.concatenate([sampler.row_ids[c["right"]] for c in candidates.values()]),
        condition_hashes=hashes)
    with _archive(receipt) as archive:
        require(set(archive.files) == set(expected)
                and all(_same_array(archive[key], value) for key, value in expected.items()),
                "C0 candidate ordered identities, polarity or condition hashes differ")

    teacher = inputs["teacher"]
    snapshots = inputs["snapshots"]
    require([s["label"] for s in snapshots] == list(SNAPSHOTS)
            and teacher["epoch"] == TEACHER_EPOCH
            and snapshots[SNAPSHOTS.index("E10")]["model_hash"] == teacher["model_hash"],
            "C0 P0 E10 snapshot order or fixed-EMA identity mismatch")
    with _archive(inputs["margins"]) as archive:
        require(set(archive.files) == {*SNAPSHOTS, "candidate_sha256", "condition_hashes", "model_hashes"}
                and _same_array(archive["candidate_sha256"], np.array(receipt["sha256"]))
                and _same_array(archive["condition_hashes"], hashes)
                and _same_array(archive["model_hashes"], np.array([s["model_hash"] for s in snapshots])),
                "C0 P0 margin archive candidate, condition or model identity mismatch")
        raw = archive["E10"]
    require(raw.dtype == np.float64 and raw.shape == expected["positive_ids"].shape
            and np.isfinite(raw).all(), "C0 P0 E10 margins must be a finite aligned FP64 vector")
    bounds = np.cumsum([len(c["ranks"]) for c in candidates.values()])[:-1]
    return candidates, dict(zip(candidates, np.split(raw, bounds)))


def _stream(name, condition):
    return name + "|" + "|".join(map(str, condition))


def _sample(context, mode, condition, support, size):
    if len(support) == size:
        return support
    gen = rng(context, _stream(f"c1_bank_{mode}", condition))
    return np.sort(gen.choice(support, size, replace=False))


def _bank_hashes(bank):
    identities, full = [], []
    for condition, entry in bank["conditions"].items():
        identity = dict(condition=condition, candidate=entry["candidate_hash"], ids=_array_hash(entry["ids"]))
        identities.append(identity)
        full.append(dict(identity, indices=_array_hash(entry["indices"]), offsets=_array_hash(entry["offsets"]),
                         target_margins=_array_hash(entry["target_margins"])))
    return dict(ordered_ids_hash=digest(identities), bank_hash=digest([bank["binding"], full]))


def _bank_entry(candidate, indices, raw, row_ids):
    offsets = np.stack((candidate["left"][indices], candidate["right"][indices])).astype("<i4")
    entry = dict(candidate_hash=candidate["hash"], offsets=offsets, ids=row_ids[offsets],
                 indices=np.asarray(indices, dtype="<u2"), target_margins=np.clip(raw[indices], -CLIP, CLIP))
    for k in ("offsets", "ids", "indices", "target_margins"):
        entry[k].flags.writeable = False
    return entry


def _bank_summary(entry, raw):
    selected = raw[entry["indices"]]
    return dict(size=len(selected), unique_pairs=len(selected),
                unique_endpoints=len(np.unique(entry["ids"])),
                clip_saturation_low=int((selected < -CLIP).sum()),
                clip_saturation_high=int((selected > CLIP).sum()),
                margin_histogram=np.histogram(entry["target_margins"], HISTOGRAM_EDGES)[0].tolist(),
                ordered_ids_hash=ordered_ids_hash(entry["ids"]))


def build_banks(sampler, seed, inputs, candidates, margins):
    """Freeze both selections with canonical candidate order and signed clipped margins."""
    teacher = inputs["teacher"]
    context = dict(dataset=sampler.dataset, seed=seed, epoch=TEACHER_EPOCH, global_step=teacher["global_step"])
    require(teacher["epoch"] == TEACHER_EPOCH and 0 < teacher["global_step"]
            and len(sampler.row_ids) <= np.iinfo(np.int32).max, "Invalid E10 or Train offset identity")
    binding = dict(context, teacher_model_hash=teacher["model_hash"],
                   train=sampler.state_dict(), candidate_sha256=inputs["candidate"]["sha256"],
                   margins_sha256=inputs["margins"]["sha256"])
    banks = {mode: dict(binding=dict(binding, mode=mode), conditions={}) for mode in ("random", "correct")}
    summaries = []
    for condition, candidate in candidates.items():
        raw = margins[condition]
        require(raw.dtype == np.float64 and raw.shape == candidate["ranks"].shape
                and len(raw) <= CANDIDATE_CAP and np.isfinite(raw).all(), "Invalid C0 condition margins")
        correct = np.flatnonzero(raw > 0)
        size = min(BANK_CAP, len(correct))
        statistics = {}
        for mode, support in (("correct", correct), ("random", np.arange(len(raw)))):
            indices = _sample(context, mode, condition, support, size)
            entry = banks[mode]["conditions"][condition] = _bank_entry(candidate, indices, raw, sampler.row_ids)
            statistics[mode] = _bank_summary(entry, raw)
        random = banks["random"]["conditions"][condition]["indices"]
        historical = banks["correct"]["conditions"][condition]["indices"]
        summaries.append(dict(condition=list(condition), candidate_size=len(raw), candidate_hash=candidate["hash"],
                              raw_correct=len(correct), raw_tied=int((raw == 0).sum()),
                              raw_incorrect=int((raw < 0).sum()), **statistics,
                              overlap=int(np.intersect1d(random, historical, assume_unique=True).size)))
    for bank in banks.values():
        bank.update(_bank_hashes(bank))
    return banks, summaries


def prepare_banks(sampler, seed, inputs):
    return build_banks(sampler, seed, inputs, *load_candidates(sampler, seed, inputs))


def bank_metadata(bank, summaries):
    return dict(binding=bank["binding"], bank_hash=bank["bank_hash"], ordered_ids_hash=bank["ordered_ids_hash"],
                histogram_edges=HISTOGRAM_EDGES, conditions=summaries,
                array_bytes=sum(array.nbytes for entry in bank["conditions"].values()
                                for array in entry.values() if isinstance(array, np.ndarray)))


def validate_bank(state, expected):
    """Check serialized offsets, IDs and aligned margins against a fresh frozen build."""
    require(state.keys() == expected.keys()
            and all(state[k] == expected[k] for k in ("binding", "bank_hash", "ordered_ids_hash"))
            and list(state["conditions"]) == list(expected["conditions"]), "Restored C1 bank identity mismatch")
    for condition, wanted in expected["conditions"].items():
        given = state["conditions"][condition]
        require(given.keys() == wanted.keys() and given["candidate_hash"] == wanted["candidate_hash"]
                and all(isinstance(given[key], np.ndarray) and _same_array(given[key], wanted[key])
                        for key in ("indices", "offsets", "ids", "target_margins")),
                f"Restored C1 bank bytes differ for condition {condition}")
    require(_bank_hashes(state) == {k: state[k] for k in ("bank_hash", "ordered_ids_hash")},
            "Restored C1 bank content hash mismatch")


def draw_positions(size, condition, context):
    return rng(context, _stream("c1_memory_draw", condition)).choice(size, PAIR_COUNT, replace=True)


def draw_pair(bank, current, context):
    """Preserve the chosen macro6 event, weight and positive/negative block layout."""
    if current is None:
        return None
    condition = current["condition"]
    entry = bank["conditions"][condition]
    size = len(entry["indices"])
    if not size:
        return dict(current, memory_source="current_fallback", memory_miss="empty_bank")
    positions = draw_positions(size, condition, context)
    ids = entry["ids"][:, positions].reshape(-1)
    offsets = entry["offsets"][:, positions].reshape(-1)
    return dict(condition=condition, w=current["w"], q=current["q"], ids=ids, offsets=offsets,
                pair_index_hash=ordered_ids_hash(ids), unique_pairs=len(np.unique(positions)),
                unique_endpoints=len(np.unique(ids)), memory_source="bank", memory_miss=None,
                target_margins=entry["target_margins"][positions], bank_size=size,
                draw_positions_hash=ordered_ids_hash(positions))


def _uniform_coverage(draws, support):
    if not draws:
        return 0.
    return 1. if support == 1 else -math.expm1(draws * math.log1p(-1 / support))


def replay_exposure(sampler, bank, committed_rows):
    """Verify committed pair identity, work and weights, then count realized exposure."""
    groups = {}
    n_train = len(sampler.row_ids)

    def pair_codes(pair):
        pos, neg = pair["offsets"].reshape(2, PAIR_COUNT)
        return pos.astype(np.int64) * n_train + neg

    for step, row in enumerate(committed_rows, 1):
        context = row["context"]
        observed = row["diagnostics"]
        phase = "warmup" if context["epoch"] <= TEACHER_EPOCH else "memory"
        current = sampler.sample_macro6(context)
        applied = current if phase == "warmup" else draw_pair(bank, current, context)
        condition = None if applied is None else applied["condition"]
        source = ("none" if applied is None else "current" if phase == "warmup" else applied["memory_source"])
        miss = "no_event" if applied is None else applied.get("memory_miss")
        forwards = int(applied is not None)
        draws = PAIR_COUNT * forwards
        work = dict(pair_draws=draws, pair_rows=2 * draws, pair_forwards=forwards, pair_diagnostic_vjp_calls=forwards)
        require(context["global_step"] == step and context["seed"] == bank["binding"]["seed"]
                and observed["train_identity"] == sampler.train_identity and observed["memory_phase"] == phase
                and observed["memory_source"] == source and observed["memory_miss"] == miss
                and (None if observed["condition"] is None else tuple(observed["condition"])) == condition
                and observed["current_pair_index_hash"] == (None if current is None else current["pair_index_hash"])
                and observed["pair_index_hash"] == (None if applied is None else applied["pair_index_hash"])
                and all(row["counts"][name] == value for name, value in work.items()),
                f"Committed C1 pair replay or work differs at logical step {step}")
        if phase == "memory":
            require(observed["bank_hash"] == bank["bank_hash"], f"Committed C1 bank differs at logical step {step}")
        if applied is None:
            exposure = dict(w=None, q=None, w_over_q=None, unique_pairs=0, unique_endpoints=0)
        else:
            exposure = {name: applied[name] for name in ("w", "q", "unique_pairs", "unique_endpoints")}
            exposure["w_over_q"] = applied["w"] / applied["q"]
        require(all(observed[name] == value for name, value in exposure.items()),
                f"Committed C1 exposure or event weights differ at logical step {step}")
        if source == "bank":
            require(observed["draw_positions_hash"] == applied["draw_positions_hash"],
                    f"Committed C1 draw positions differ at logical step {step}")
        group = groups.setdefault((phase, condition), dict(current=[], applied=[], misses=0))
        group["misses"] += source in ("none", "current_fallback")
        if applied is not None:
            codes = pair_codes(current)
            group["current"].append(codes)
            group["applied"].append(pair_codes(applied) if source == "bank" else codes)

    output = []
    for (phase, condition), group in groups.items():
        events = len(group["applied"])
        draws = events * PAIR_COUNT
        output_row = dict(phase=phase, condition=condition, events=events, draws=draws, misses=group["misses"])
        if condition is not None:
            distinct = np.unique(np.concatenate(group["applied"]))
            current = np.unique(np.concatenate(group["current"]))
            d, _, a, b = condition
            universe = len(sampler._pools[d, a]) * len(sampler._pools[d, b])
            support = len(bank["conditions"][condition]["indices"]) if phase == "memory" else 0
            output_row.update(unique_pairs=len(distinct), unique_current_pairs=len(current),
                              current_overlap=int(np.intersect1d(distinct, current, assume_unique=True).size),
                              current_universe=universe, uniform_current_unique=universe * _uniform_coverage(draws, universe),
                              bank_size=support)
            if support:
                output_row.update(uniform_bank_unique=support * _uniform_coverage(draws, support),
                                  expected_draws_per_bank_member=draws / support,
                                  uniform_current_overlap=len(distinct) * _uniform_coverage(draws, universe))
        output.append(output_row)
    return output
