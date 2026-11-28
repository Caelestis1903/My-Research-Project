"""Frozen B trajectory identities and read-only natural Train/Validation inputs."""
from dataclasses import dataclass
from functools import cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.conditional_history_shared_update_20261122.history_controls import history_config
from experiments.conditional_history_shared_update_20261122.protocol import digest, read, sha, v3
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.data import joint_counts
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31 import audited_data as source

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
STUDY = PACKAGE.name
MAIN = "641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc"
DATASETS = ("movielens", "kuairand")
SEEDS = tuple(range(2022, 2027))
READOUTS = ("online", "fixed_ema")
SPLITS = ("train", "validation")
RUNTIME = ("__init__", "protocol", "freshness_optimizer", "runner", "scheduler", "ram_state")
B_HISTORY_PROVENANCE = "EXP080_A_HISTORY_A21"
GA31_OBJECTIVE = "vendor/ema02/vendor/gradient_aug_20261031_v2/objectives.py"
GA31_ORIGINAL_BLOB = "c4b58d0e230c03bfe69f8822ddac818d332d5ed1"
GA31_CALLBACK_SHA256 = "df8845ab9076a1362ed35542dcda6c8632140d583c55e90832d8470b008a7afd"


@dataclass(frozen=True)
class RunSpec:
    stage: str
    dataset: str
    seed: int
    arm: str
    config: dict
    adapter: dict

    def __post_init__(self):
        for name in ("config", "adapter"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def key(self):
        return f"{self.stage}-{self.dataset}-{self.arm}-seed{self.seed}"

    def record(self):
        return dict(key=self.key, stage=self.stage, dataset=self.dataset, seed=self.seed,
                    arm=self.arm, config=dict(self.config), adapter=dict(self.adapter))

    @property
    def config_hash(self):
        return digest([dict(self.config), dict(self.adapter)])

    @property
    def workload(self):
        return f"{self.stage}:{self.dataset}:{self.arm}"


def b_jobs():
    jobs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for arm in ("B0", "B1", "B2"):
                config = (v3.config("P00", dataset, .05) if arm == "B0"
                          else history_config("A21", dataset))
                adapter = dict(gamma={"B0": None, "B1": 0., "B2": .5}[arm], observer=arm == "B1")
                jobs.append(RunSpec("B1", dataset, seed, arm, config, adapter))
    return jobs


def algorithm_digest(job, source_identity):
    names = ("a_step", "macro_pair_sampler") if job.stage == "A1" else ("freshness_optimizer",)
    numerical = {name: source_identity["runtime"][name] for name in names}
    return digest(dict(main=MAIN, source_manifest=source_identity["source_manifest"],
                       ga31=source_identity["ga31"], numerical=numerical, config_hash=job.config_hash))


def scope_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


@cache
def verify_sources(artifacts, stage):
    audit_path = artifacts / "SOURCE_AUDIT.json"
    audit = read(audit_path)
    if audit["study_id"] != STUDY or audit["main_head"] != MAIN or tuple(audit["seeds"]) != SEEDS:
        raise ValueError("Frozen source audit identity mismatch")
    # Include the audited history entry points and the v3 evaluator and vendor dependencies.
    manifest_path = (ROOT.with_name("phase0-3") / "research_reports/2026-11-02_mdmt_information_momentum_v3"
                     / "IMPLEMENTATION_SOURCE_MANIFEST.json")
    manifest_hash = sha(manifest_path)
    if manifest_hash != "7034a3bb59e6d08e3296183258e0e7b16ab74d0248ee0a3b75c526a680462cc0":
        raise ValueError("Pinned v3 source manifest mismatch")
    pinned = {row["path"]: row["sha256"] for row in audit["verified_main_sources"]}
    pinned.update({str((v3.HERE / row["path"]).relative_to(ROOT)): row["sha256"]
                   for row in read(manifest_path)["files"]})
    ga31 = str((v3.HERE / GA31_OBJECTIVE).relative_to(ROOT))
    original = pinned[ga31]
    for path, expected in pinned.items():
        actual = sha(ROOT / path)
        allowed = (GA31_CALLBACK_SHA256,) if path == ga31 else (expected,)
        if actual not in allowed:
            raise ValueError(f"Frozen source mismatch: {path}")
    names = RUNTIME + (("a_step", "macro_pair_sampler") if stage == "A1" else ())
    files = {name: sha(PACKAGE / f"{name}.py") for name in names}
    design_dir = ROOT / "research_reports" / STUDY
    design = {"B": sha(design_dir / "B_EXECUTION_DESIGN.md")}
    if stage == "A1":
        design["A"] = sha(design_dir / "A_EXECUTION_DESIGN.md")
    identity = dict(stage=stage, main=MAIN, runtime=files, source_audit=sha(audit_path),
                    source_manifest=manifest_hash, data_audit=sha(artifacts / "DATA_AUDIT.json"),
                    ga31=dict(original_blob=GA31_ORIGINAL_BLOB, original_sha256=original,
                              callback_sha256=GA31_CALLBACK_SHA256), design=design)
    return dict(identity, source_hash=digest(identity))


@dataclass(frozen=True)
class TrainValidation:
    dataset: str
    features: source.FeatureSchema
    train_x: np.ndarray
    train_y: np.ndarray
    train_row_ids: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_row_ids: np.ndarray
    identity: dict
    domain_num: int = 3
    task_num: int = 2

    @property
    def data_hash(self):
        return self.identity["data_hash"]


def load_train_validation(dataset, audit_path):
    audit = read(audit_path)
    if audit["study_id"] != STUDY or audit["status"] != "TRAIN_VALIDATION_VERIFIED":
        raise ValueError("Train/Validation audit identity mismatch")
    expected = audit["datasets"][dataset]
    cache_path = Path(expected["cache"])
    if not cache_path.is_absolute() or expected["test_loaded"] is not False:
        raise ValueError(f"{dataset} requires an absolute Train/Validation-only cache")
    frozen = v3.load_protocol()["datasets"][dataset]
    pinned = frozen["source"]
    historical_path = Path(audit["historical_receipt"])
    if not historical_path.is_absolute():
        raise ValueError("Train/Validation provenance path must be absolute")
    historical = read(historical_path)[dataset]
    features = source.FeatureSchema(**read(cache_path / "features.json"))
    arrays, hashes, counts, fingerprints, ids_hashes = {}, {}, {}, {}, {}
    for split in SPLITS:
        x, y, rows = arrays[split] = tuple(
            np.load(cache_path / f"{split}_{field}.npy", mmap_mode="r", allow_pickle=False)
            for field in ("x", "y", "row_ids"))
        if (x.dtype != np.float32 or y.dtype != np.float32 or rows.dtype != np.int64
                or x.shape != (len(rows), len(features.columns)) or y.shape != (len(rows), 2)
                or rows.ndim != 1 or any(a.flags.writeable for a in (x, y, rows))
                or not np.isin(y, (0, 1)).all()):
            raise ValueError(f"Malformed {dataset}/{split} input arrays")
        prefix = "val" if split == "validation" else split
        hashes.update({f"{prefix}_x": source.array_hash(x), f"{prefix}_y": source.array_hash(y)})
        counts[split] = len(rows)
        ids_hashes[split] = source.row_ids_hash(rows)
        fingerprints[split] = source.split_row_fingerprint(x, y, rows)
    train_x, train_y, _ = arrays["train"]
    receipt = dict(dataset=dataset, split_id="original", raw_path=frozen["csv"],
                   raw_sha256=pinned["raw_sha256"], array_hashes=hashes,
                   split_row_fingerprints=fingerprints, split_row_id_hashes=ids_hashes,
                   split_rows=counts, encoder_hash=features.encoder_hash, scaler_hash=features.scaler_hash,
                   train_domain_counts=np.bincount(train_x[:, -1].astype(np.int64), minlength=3).tolist(),
                   train_joint_labels=joint_counts(train_x, train_y), features=features.as_dict(),
                   preprocessing_fit="train_only", include_test=False)
    receipt["data_hash"] = digest({k: v for k, v in receipt.items() if k != "raw_path"})
    for reference in (pinned, historical, expected):
        for key, value in receipt.items():
            if key not in reference:
                continue
            candidate = reference[key]
            if isinstance(value, dict):
                candidate = {k: candidate[k] for k in value}
            if value != candidate:
                raise ValueError(f"{dataset} frozen data identity mismatch: {key}")
    return TrainValidation(dataset, features, *arrays["train"], *arrays["validation"], receipt)


def stage_jobs(stage):
    if stage == "B1":
        return b_jobs()
    if stage == "A1":
        from .a_step import a_config
        return [RunSpec("A1", dataset, seed, arm, a_config(arm, dataset), {})
                for dataset in DATASETS for seed in SEEDS for arm in ("B0", "P0", "G0", "GP0")]
    raise ValueError(f"Unregistered execution stage {stage}")
