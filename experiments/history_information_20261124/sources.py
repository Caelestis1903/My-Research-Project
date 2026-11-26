"""Pinned code and a train/validation-only view of the natural populations."""

from dataclasses import dataclass
from functools import cache
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.data import joint_counts, sha_file
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31 import audited_data as source

from .protocol import OUTPUT, PLAN, ROOT, STUDY, digest, manifest

PUBLISHED_ROOT = ROOT / "research_plans/26-11-03_overnight_AB/protocol.json"
PUBLISHED_PACKAGE = "experiments/overnight_ab_20261103"
GAUSSIAN_PACKAGE = "experiments/gaussian_domain_coverage_20261104"
_GA31_VENDOR = "vendor/ema02/vendor/gradient_aug_20261031_v2"
_MAIN_GA31 = f"experiments/conditional_information_momentum_20261102_v3/{_GA31_VENDOR}"
_PUBLISHED_GA31 = f"experiments/mdmt_history_information_20261103/vendor/v3/{_GA31_VENDOR}"
GAUSSIAN_OBJECTIVES = f"{_PUBLISHED_GA31}/objectives.py"
_SHARED_GA31_BLOBS = {
    "views.py": "ecc354eba1c1d1e8b21694e8dbfb3c2b7221d254",
    "protocol.py": "7e59458c3473717f9c07e48dca984e2c19380bd7",
}
_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray

    def __post_init__(self):
        for array in (self.x, self.y, self.row_ids):
            array.setflags(write=False)


@dataclass(frozen=True)
class TrainValidation:
    dataset: str
    features: source.FeatureSchema
    train: Split
    validation: Split
    identity: dict
    domain_num: int = 3
    task_num: int = 2

    @property
    def data_hash(self):
        return self.identity["data_hash"]

    @property
    def train_x(self):
        return self.train.x

    @property
    def train_y(self):
        return self.train.y

    @property
    def train_ids(self):
        return self.train.row_ids

    train_row_ids = train_ids

    @property
    def val_x(self):
        return self.validation.x

    @property
    def val_y(self):
        return self.validation.y

    @property
    def val_ids(self):
        return self.validation.row_ids

    val_row_ids = val_ids


def _git(command, directory):
    return subprocess.check_output(["git", "-C", str(directory), *command], text=True).strip()


def _blob(contents):
    return hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()


def _frozen_roots():
    return (ROOT, ROOT.with_name("phase0-3_overnight_ab"), ROOT.with_name("phase0-3_gaussian_coverage"))


def _restore_missing(path, expected, repository, candidates):
    for candidate in candidates:
        try:
            contents = candidate.read_bytes()
        except OSError:
            continue
        if _blob(contents) == expected:
            break
    else:
        try:
            contents = subprocess.check_output(
                ["git", "-C", str(repository), "cat-file", "blob", expected], stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            raise FileNotFoundError(f"No frozen blob {expected} for missing source: {path}") from exc
        if _blob(contents) != expected:
            raise ValueError(f"Recovery bytes differ from frozen blob {expected}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_bytes(contents)
        try:
            os.link(staged, path)
        except FileExistsError:
            # Another caller may have restored the file. Verify the winning bytes.
            pass


def _verified_contents(path, expected, repository=None, candidates=()):
    try:
        contents = path.read_bytes()
    except FileNotFoundError:
        if repository is None:
            raise
        _restore_missing(path, expected, repository, candidates)
        contents = path.read_bytes()
    if _blob(contents) != expected:
        raise ValueError(f"Pinned source changed: {path}")
    return contents


def ensure_shared_sources():
    """Verify or recover the identical GA31 dependencies before importing them."""
    repository = ROOT.with_name("phase0-3_git")
    pinned = manifest()["repository"]
    for filename, expected in _SHARED_GA31_BLOBS.items():
        main = f"{_MAIN_GA31}/{filename}"
        published = f"{_PUBLISHED_GA31}/{filename}"
        versions = ((pinned["evidence_main_sha"], main), (pinned["published_model_source_sha"], published))
        for commit, relative in versions:
            if _git(["rev-parse", f"{commit}:{relative}"], repository) != expected:
                raise ValueError(f"Pinned shared GA31 blob changed: {commit}:{relative}")
        for relative in (main, published):
            path = ROOT / relative
            candidates = (root / replica for root in _frozen_roots() for replica in (main, published)
                          if root / replica != path)
            _verified_contents(path, expected, repository, candidates)


def verify_shared_imports():
    for directory in (_MAIN_GA31, _PUBLISHED_GA31):
        for filename in _SHARED_GA31_BLOBS:
            path = ROOT / directory / filename
            module = sys.modules[f"{directory.replace('/', '.')}.{path.stem}"]
            if Path(module.__file__).resolve() != path.resolve():
                raise ValueError(f"Loaded shared GA31 source differs from frozen path: {module.__file__}")


def _runtime(lock):
    differences = {}
    for line in lock.read_text().splitlines():
        if "==" not in line:
            continue
        name, expected = line.split("==", 1)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            differences[name] = dict(expected=expected, installed=installed)
    gpus = []
    if shutil.which("nvidia-smi"):
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, check=False)
        if not query.returncode:
            for row in query.stdout.splitlines():
                index, uuid, name, driver = (field.strip() for field in row.split(","))
                gpus.append(dict(index=int(index), uuid=uuid, name=name, driver=driver))
    return dict(python=sys.version.split()[0], executable=sys.executable, torch=str(torch.__version__),
                cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                gpu_devices=gpus, package_lock_differences=differences)


def _pinned_files(repository, commit, directory, paths, *, exclude=(), recover=False):
    upstream = _git(["ls-tree", "-r", commit, "--", *paths], repository)
    if not upstream:
        raise ValueError(f"No source closure found for {commit} and {paths}")
    records = []
    for row in upstream.splitlines():
        header, relative = row.split("\t", 1)
        path = directory / relative
        if relative in exclude or path.suffix not in {".py", ".json", ".cpp"} or "tests" in Path(relative).parts:
            continue
        expected = header.split()[-1]
        candidates = (root / relative for root in _frozen_roots() if root != directory) if recover else ()
        contents = _verified_contents(path, expected, repository if recover else None, candidates)
        records.append(dict(path=relative, actual_path=str(path), commit=commit, git_blob=expected,
                            sha256=hashlib.sha256(contents).hexdigest()))
    return records


def _source_hash(records):
    return digest([{k: v for k, v in row.items() if k != "actual_path"} for row in records])


def audit_gaussian_source(repository, commit):
    directory = ROOT.parent / "phase0-3_gaussian_coverage"
    try:
        if commit != manifest()["repository"]["published_model_source_sha"]:
            raise ValueError(f"Unexpected Gaussian source commit: {commit}")
        files = _pinned_files(repository, commit, directory, (GAUSSIAN_PACKAGE,), recover=True)
        files.extend(_pinned_files(repository, commit, ROOT, (GAUSSIAN_OBJECTIVES,), recover=True))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return dict(status="UNRESOLVED_IDENTITY", source_hash=None, files=[],
                    blocker=f"{type(exc).__name__}: {exc}")
    return dict(status="VERIFIED", source_hash=_source_hash(files), files=files, blocker=None)


def audit_sources():
    """Hash the current implementation and validate each selected dependency blob."""
    ensure_shared_sources()
    repository = ROOT.parent / "phase0-3_git"
    pinned = manifest()["repository"]
    roots = (
        (pinned["evidence_main_sha"], ROOT, (
            "experiments/conditional_information_momentum_20261102_v3",
            "experiments/conditional_history_shared_update_20261122",
        )),
        (pinned["published_model_source_sha"], ROOT, (
            PUBLISHED_PACKAGE, "experiments/mdmt_history_information_20261103",
            str(PUBLISHED_ROOT.relative_to(ROOT)),
        )),
    )
    records = []
    for commit, directory, paths in roots:
        records.extend(_pinned_files(repository, commit, directory, paths, exclude=(GAUSSIAN_OBJECTIVES,)))
    gaussian = audit_gaussian_source(repository, pinned["published_model_source_sha"])
    external = ROOT / PUBLISHED_PACKAGE / "vendor/sota2526/SOURCE_MANIFEST.json"
    head = {}
    for item in json.loads(external.read_text())["external_dependency_files"]:
        directory = Path(item["source_root"])
        if directory not in head:
            head[directory] = _git(["rev-parse", "HEAD"], directory)
        path = directory / item["path"]
        contents = path.read_bytes()
        if (head[directory] != item["source_commit"] or _blob(contents) != item["git_blob"]
                or hashlib.sha256(contents).hexdigest() != item["sha256"]):
            raise ValueError(f"Pinned published dependency changed: {path}")
        records.append(dict(path=item["path"], actual_path=str(path), commit=item["source_commit"],
                            git_blob=item["git_blob"], sha256=item["sha256"]))
    lock = ROOT / "requirements-phase0-3.lock"
    for path in sorted(Path(__file__).parent.glob("*.py")) + [PLAN, lock]:
        contents = path.read_bytes()
        records.append(dict(path=str(path.relative_to(ROOT)), actual_path=str(path), commit=None,
                            git_blob=_blob(contents), sha256=hashlib.sha256(contents).hexdigest()))
    return dict(source_hash=_source_hash(records), files=records, evidence_commit=pinned["evidence_main_sha"],
                published_commit=pinned["published_model_source_sha"],
                gaussian_source=gaussian, gaussian_source_verified=gaussian["status"] == "VERIFIED",
                published_source_verified=True, runtime=_runtime(lock))


def _population(dataset):
    ensure_shared_sources()
    from experiments.conditional_information_momentum_20261102_v3 import protocol as v3

    frozen = v3.load_protocol()["datasets"][dataset]
    planned = manifest()["datasets"][dataset]
    path = Path(frozen["csv"])
    expected = frozen["source"]
    if (str(path) != planned["csv"] or expected["raw_sha256"] != planned["raw_sha256"]
            or sha_file(path) != expected["raw_sha256"]):
        raise ValueError(f"Frozen {dataset} natural CSV identity changed")
    # The validated raw sources contain only these physical domains. The pinned
    # row counts therefore recover exact raw boundaries without parsing holdout rows.
    t, v, h = (expected["split_rows"][key] for key in ("train", "validation", "test"))
    length = t + v + h
    if t != int(length * .8) or t + v != int(length * .9):
        raise ValueError(f"Frozen {dataset} row boundaries changed")
    return path, expected, {"train": (0, t), "validation": (t, t + v), "test": (t + v, length)}


def _movielens(path, bounds, splits):
    data = pd.read_csv(path, nrows=max(bounds[name][1] for name in splits))
    data["cate_id"] = data["genres"].astype(str).str.split("|").str[0]
    ages = ({1, 18}, {25}, {35, 45, 50, 56})
    data["domain_indicator"] = data["age"].map({age: d for d, group in enumerate(ages) for age in group})
    if data["domain_indicator"].isna().any():
        raise ValueError("MovieLens domain mapping changed")
    columns = source.MOVIELENS_COLUMNS
    sparse = tuple(source.MOVIELENS_SPARSE_COLUMNS)
    train = data.iloc[:bounds["train"][1]]
    vocab = {col: source._fit_vocab(train[col]) for col in sparse if col != "domain_indicator"}
    train_age = pd.to_numeric(train["age"], errors="raise").astype(np.float64)
    age_min, age_max = float(train_age.min()), float(train_age.max())
    age_span = age_max - age_min
    schema = source._feature_schema("movielens", columns, ("age",), sparse,
        {col: 3 if col == "domain_indicator" else len(vocab[col]) + 1 for col in sparse},
        vocab, {"age_min": age_min, "age_max": age_max})
    result = {}
    for split in splits:
        start, stop = bounds[split]
        frame = data.iloc[start:stop]
        x = np.zeros((len(frame), len(columns)), dtype=np.float32)
        for j, col in enumerate(columns):
            if col == "age":
                age = pd.to_numeric(frame["age"], errors="raise").astype(np.float64).to_numpy()
                if age_span:
                    x[:, j] = ((age - age_min) / age_span).astype(np.float32)
            elif col == "domain_indicator":
                x[:, j] = frame[col].astype(np.int64).to_numpy().astype(np.float32)
            else:
                x[:, j] = source._apply_vocab(frame[col], vocab[col]).astype(np.float32)
        ratings = frame["rating"].astype(int)
        y = np.column_stack((ratings > 3, ratings > 4)).astype(np.float32)
        result[split] = Split(x, y, np.arange(start, stop, dtype=np.int64))
    return schema, result


def _kuairand(path, bounds, splits):
    required = list(source.KUAIRAND_SOURCE_COLUMNS) + ["is_click", "long_view"]
    columns, sparse = source.KUAIRAND_FEATURE_COLUMNS, source.KUAIRAND_SPARSE_COLUMNS
    category = [name for name in source.KUAIRAND_SOURCE_COLUMNS if name != "tab"]
    strings = {"user_active_degree", "follow_user_num_range", "fans_user_num_range",
               "friend_user_num_range", "register_days_range", "video_type", "upload_dt", "upload_type", "tag"}
    dtypes = dict(tab="int8", is_click="int8", long_view="int8")
    for col in category:
        dtypes[col] = ("string" if col in strings else "float32" if
                       col.startswith("onehot_feat") or col in {"server_width", "server_height", "music_type"}
                       else "int32")

    def chunks(stop):
        return pd.read_csv(path, usecols=required, dtype=dtypes, chunksize=250_000, nrows=stop)

    vocab_sets = {col: set() for col in category}
    with chunks(bounds["train"][1]) as reader:
        for frame in reader:
            for col in category:
                vocab_sets[col].update(source._safe_value(v) for v in frame[col].tolist())
    vocab = {col: {key: i + 1 for i, key in enumerate(sorted(values))}
             for col, values in vocab_sets.items()}
    del vocab_sets

    arrays = {name: np.empty((bounds[name][1] - bounds[name][0], len(columns)), np.float32) for name in splits}
    labels = {name: np.empty((len(x), 2), np.float32) for name, x in arrays.items()}
    domain_map = {0: 0, 1: 1, 4: 2}
    offset = 0
    last = max(bounds[name][1] for name in splits)
    with chunks(last) as reader:
        for frame in reader:
            end = offset + len(frame)
            for name in splits:
                start, stop = bounds[name]
                left, right = max(offset, start), min(end, stop)
                if left >= right:
                    continue
                block = frame.iloc[left - offset:right - offset]
                selection = slice(left - start, right - start)
                domain = block["tab"].map(domain_map)
                if domain.isna().any():
                    raise ValueError("KuaiRand tab mapping changed")
                for j, col in enumerate(columns):
                    arrays[name][selection, j] = (domain.to_numpy() if col == "domain_indicator"
                        else source._apply_vocab(block[col], vocab[col]).astype(np.float32))
                labels[name][selection] = block[["is_click", "long_view"]].to_numpy(np.float32)
            offset = end
    if offset != last:
        raise ValueError("KuaiRand CSV ended before the frozen boundary")

    schema = source._feature_schema("kuairand", columns, (), sparse,
        {col: 3 if col == "domain_indicator" else len(vocab[col]) + 1 for col in sparse},
        vocab, {})
    # Rows are mapped after cutting by raw position, as in the pinned loader.
    result = {}
    for name, x in arrays.items():
        ids = np.arange(*bounds[name], dtype=np.int64)
        mask = np.isin(x[:, -1], (0, 1, 2))
        result[name] = Split(x, labels[name], ids) if mask.all() else Split(x[mask], labels[name][mask], ids[mask])
    return schema, result


def _encode(dataset, path, bounds, splits):
    encode = {"movielens": _movielens, "kuairand": _kuairand}[dataset]
    return encode(path, bounds, splits)


def _verified(dataset, path, expected, features, splits):
    arrays, fingerprints, row_ids, counts = {}, {}, {}, {}
    for split, values in splits.items():
        prefix = "val" if split == "validation" else split
        arrays.update({f"{prefix}_x": source.array_hash(values.x), f"{prefix}_y": source.array_hash(values.y)})
        fingerprints[split] = source.split_row_fingerprint(values.x, values.y, values.row_ids)
        row_ids[split] = source.row_ids_hash(values.row_ids)
        counts[split] = len(values.y)
        if not np.isin(values.y, (0, 1)).all():
            raise ValueError(f"{dataset}/{split} labels are not binary")
    receipt = dict(dataset=dataset, split_id="original", raw_path=str(path), raw_sha256=expected["raw_sha256"],
                   array_hashes=arrays, split_row_fingerprints=fingerprints, split_row_id_hashes=row_ids,
                   split_rows=counts, encoder_hash=features.encoder_hash, scaler_hash=features.scaler_hash)
    for key, value in receipt.items():
        if key in expected:
            reference = expected[key]
            if isinstance(value, dict):
                reference = {k: reference[k] for k in value}
            if value != reference:
                raise ValueError(f"{dataset} frozen data identity mismatch: {key}")
    if "train" in splits:
        train = splits["train"]
        receipt["train_domain_counts"] = np.bincount(train.x[:, -1].astype(np.int64), minlength=3).tolist()
        if receipt["train_domain_counts"] != expected["train_domain_counts"]:
            raise ValueError(f"{dataset} train domain counts changed")
        receipt["train_joint_labels"] = joint_counts(train.x, train.y)
        if dataset == "movielens" and any(v["01"] for v in receipt["train_joint_labels"].values()):
            raise ValueError("MovieLens label nesting changed")
    receipt.update(features=features.as_dict(), preprocessing_fit="train_only", include_test="test" in splits)
    receipt["data_hash"] = digest({k: v for k, v in receipt.items() if k != "raw_path"})
    return receipt


@cache
def load_train_validation(dataset, cache_root=None):
    path, expected, bounds = _population(dataset)
    if cache_root is None:
        features, splits = _encode(dataset, path, bounds, _SPLITS)
    else:
        target = Path(cache_root) / dataset
        features = source.FeatureSchema(**json.loads((target / "features.json").read_text()))
        splits = {name: Split(*(np.load(target / f"{name}_{field}.npy", mmap_mode="r", allow_pickle=False)
                                for field in ("x", "y", "row_ids"))) for name in _SPLITS}
    identity = _verified(dataset, path, expected, features, splits)
    return TrainValidation(dataset, features, splits["train"], splits["validation"], identity)


def prepare_train_validation_cache(root, dataset):
    """Publish immutable input arrays once; worker loads verify by pinned array identities."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / dataset
    if not target.exists():
        bundle = load_train_validation(dataset)
        with tempfile.TemporaryDirectory(prefix=f"{dataset}-", dir=root) as temporary:
            staging = Path(temporary) / dataset
            staging.mkdir()
            (staging / "features.json").write_text(json.dumps(bundle.features.as_dict()))
            for name, split in (("train", bundle.train), ("validation", bundle.validation)):
                for field in ("x", "y", "row_ids"):
                    np.save(staging / f"{name}_{field}.npy", getattr(split, field), allow_pickle=False)
            staging.rename(target)
        load_train_validation.cache_clear()
    return load_train_validation(dataset, root)


def load_authorized_test(dataset, authorization, *, artifact_root=OUTPUT):
    """Called only by the evaluator after freeze and selected RAM readback."""
    policy_path = OUTPUT / "TEST_POLICY.json"
    policy = json.loads(policy_path.read_text())
    required = dict(study_id=STUDY, split="test", dataset=dataset,
                    policy_hash=sha_file(policy_path), actual_selected_ram_get_hash_verified=True,
                    new_test_endpoints=["selected"])
    if (not policy.get("current_task_evaluation_allowed") or policy.get("study_id") != STUDY
            or policy.get("new_test_endpoints") != ["selected"]
            or not isinstance(authorization, dict)
            or any(authorization.get(k) != val for k, val in required.items())
            or not re.fullmatch("[a-f0-9]{64}", str(authorization.get("freeze_hash", "")))):
        raise PermissionError("Frozen selected Test authorization is required")
    root = Path(artifact_root)
    try:
        claim = json.loads((root / "TEST_ACCESS.json").read_text())
        freeze = json.loads((root / "FINAL_FREEZE.json").read_text())
    except FileNotFoundError as exc:
        raise PermissionError("Frozen Test claim or final freeze receipt is absent") from exc
    freeze_hash = freeze.pop("freeze_hash", None)
    if (claim != {k: v for k, v in authorization.items() if k != "dataset"}
            or claim.get("status") != "TEST_ACCESS_CLAIMED"
            or freeze_hash != claim["freeze_hash"] or freeze_hash != digest(freeze)
            or freeze.get("study_id") != STUDY or freeze.get("policy_hash") != claim["policy_hash"]
            or freeze.get("readout_endpoint") != "selected"
            or freeze.get("adaptive_changes_after_test") is not False):
        raise PermissionError("Frozen Test claim disagrees with the selected-endpoint freeze")
    ensure_shared_sources()
    from .scheduler import stop_status

    if stop := stop_status(root):
        raise PermissionError(f"{stop[0]} before opening frozen Test: {stop[1]}")
    path, expected, bounds = _population(dataset)
    if stop := stop_status(root):
        raise PermissionError(f"{stop[0]} before opening frozen Test: {stop[1]}")
    features, splits = _encode(dataset, path, bounds, ("test",))
    _verified(dataset, path, expected, features, splits)
    return splits["test"]
