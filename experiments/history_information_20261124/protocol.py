"""Immutable trajectory identities resolved from the approved matrix."""

from dataclasses import asdict, dataclass
from functools import cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = "history_information_20261124"
PLAN = ROOT / "research_plans/26-11-24_history_information/experiment_manifest.json"
OUTPUT = ROOT / "artifacts" / STUDY
CANDIDATES = {"C1", "C2", "SHORT_C1", "SHORT_C2", "C12"}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@cache
def manifest():
    return json.loads(PLAN.read_text())


@cache
def templates():
    return {row["arm_id"]: row for row in manifest()["arm_templates"]}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    arm: str
    stage: str | None = None
    algorithm_id: str | None = None
    backbone: str | None = None
    base_objective: str | None = None
    scope: str | None = None
    zeta: float | None = None
    alpha: float | None = None
    beta_pair: float | None = None
    lambda_pair: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    eval_batch_size: int | None = None
    readouts: tuple[str, ...] | None = None
    selection_readout: str | None = None
    target_algorithm_id: str | None = None
    source_hash: str = ""
    data_hash: str = ""
    group_hash: str = ""
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        template = templates().get(self.arm)
        if template is None or self.dataset not in template["datasets"] or self.seed not in template["seeds"]:
            raise ValueError(f"Unregistered trajectory {(self.arm, self.dataset, self.seed)}")
        base = {"m3oe": "CLEAN_BCE", "hypergate": "HYPERGATE_NATIVE", "autoifs": "AUTOIFS_NATIVE"}
        backdrop = "GA31_R3_D0" if self.arm.startswith("G_") else base[template["backbone"]]
        backbone_config = manifest()["backbones"][template["backbone"]]
        fixed = dict(stage=template["stage"], backbone=template["backbone"], base_objective=backdrop,
                     scope=template.get("scope", "NATIVE_ALL"), epochs=template["epochs"],
                     batch_size=backbone_config["batch_size"], eval_batch_size=backbone_config["eval_batch_size"],
                     readouts=tuple(template["readouts"]))
        for key, expected in fixed.items():
            value = getattr(self, key)
            if key == "readouts" and value is not None:
                value = tuple(value)
            if value is not None and value != expected:
                raise ValueError(f"{self.arm} requires {key}={expected!r}")
            object.__setattr__(self, key, expected)

        expected = template["algorithm_id"]
        kind = self.algorithm_id or expected
        if expected == "FROZEN_CSTAR":
            if kind not in CANDIDATES:
                raise ValueError(f"{self.arm} requires a frozen candidate algorithm")
        elif kind != expected:
            raise ValueError(f"{self.arm} requires algorithm_id={expected}")
        object.__setattr__(self, "algorithm_id", kind)
        lam = float(template["lambda_pair"] if self.lambda_pair is None else self.lambda_pair)
        if lam not in (0., .05) or (lam and not template["lambda_pair"]):
            raise ValueError(f"Invalid PAIR weight {lam} for {self.arm}")
        object.__setattr__(self, "lambda_pair", lam)

        if kind == "H0_NORM_MATCHED":
            allowed = {"C1", "SHORT_C1"} if self.arm == "M_C1" else {"C2", "SHORT_C2"}
            if self.target_algorithm_id not in allowed:
                raise ValueError(f"{self.arm} requires a frozen target algorithm")
        elif self.target_algorithm_id is not None:
            raise ValueError(f"{self.arm} has no norm-match target")
        effective = self.target_algorithm_id or kind
        beta = .9 if effective in ("SHORT_C1", "SHORT_C2") else template.get("beta_pair")
        if effective in CANDIDATES:
            beta = .9 if effective.startswith("SHORT_") else .99
        if self.beta_pair is not None and self.beta_pair != beta:
            raise ValueError(f"{self.arm} requires beta_pair={beta}")
        object.__setattr__(self, "beta_pair", beta)
        for key, families in (("zeta", {"C1", "SHORT_C1", "C12"}), ("alpha", {"C2", "SHORT_C2", "C12"})):
            param = getattr(self, key)
            if param is None and effective in families and key not in template:
                raise ValueError(f"Missing frozen {key} for {self.arm}")
            param = float(template.get(key, 0.) if param is None else param)
            if param not in (0., .5, 1.) or (param and effective not in families):
                raise ValueError(f"Invalid {key}={param} for {self.arm}")
            if key in template and param not in (0., template[key]):
                raise ValueError(f"{self.arm} requires {key}={template[key]}")
            object.__setattr__(self, key, param)
        if self.selection_readout is not None and self.selection_readout not in self.readouts:
            raise ValueError(f"Invalid selection readout {self.selection_readout}")
        if isinstance(self.dependencies, str) or any(not isinstance(key, str) for key in self.dependencies):
            raise ValueError("Dependencies must contain immutable trajectory keys")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))

    @property
    def study_id(self):
        return STUDY

    @property
    def lr(self):
        return .0001 if self.backbone == "autoifs" else manifest()["backbones"][self.backbone]["lr"]

    @property
    def weight_decay(self):
        if self.backbone == "m3oe":
            return manifest()["datasets"][self.dataset]["weight_decay_m3oe"]
        return .0003 if self.backbone == "autoifs" else 0.

    @property
    def rho(self):
        return manifest()["datasets"][self.dataset]["rho"]

    @property
    def steps_per_epoch(self):
        return manifest()["datasets"][self.dataset][f"{self.backbone}_steps_per_epoch"]

    @property
    def canonical_arm(self):
        prefix = "HG_" if self.backbone == "hypergate" else "G_" if self.base_objective == "GA31_R3_D0" else ""
        if self.backbone == "autoifs":
            return self.arm
        if not self.lambda_pair:
            return prefix + "B0"
        if self.algorithm_id in CANDIDATES and self.beta_pair == .99 and self.zeta == self.alpha == 0.:
            return "H0_N" if self.scope == "N" else prefix + "H0"
        return self.arm

    @property
    def optimizer_spec(self):
        arm = self.canonical_arm
        if arm in {"H0_N", "HG_B0", "HG_H0", "G_B0", "G_H0"}:
            # Queue identity belongs to the outer RunSpec, including on the parent route.
            return RunSpec(self.dataset, self.seed, arm)
        return self

    def to_dict(self):
        values = asdict(self)
        for key in ("readouts", "dependencies"):
            values[key] = list(values[key])
        return dict(study_id=STUDY, **values)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if values.pop("study_id", STUDY) != STUDY:
            raise ValueError("RunSpec study mismatch")
        return cls(**values)

    @property
    def config_hash(self):
        values = self.to_dict()
        for key in ("seed", "stage", "source_hash", "data_hash", "group_hash", "dependencies"):
            del values[key]
        return digest(values)

    @property
    def job_key(self):
        return digest(self.to_dict())
