"""Load pinned upstream model code from caller-provided external checkouts."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


AUTOIFS_COMMIT = "65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e"
RECSTUDIO_COMMIT = "f08cca607e998e8be4c5e915a393346baffa71f6"
HYPERGATE_COMMIT = "fdc82e3f3feee23694a8123f91092615f977900e"


def _git_head(root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_source_roots(source_roots: dict[str, str | Path]) -> dict[str, Path]:
    """Require external checkouts at the exact inspected commits."""
    roots = {name: Path(path).expanduser().resolve() for name, path in source_roots.items()}
    for name in ("recstudio", "autoifs"):
        if name not in roots:
            raise ValueError(f"Missing external source root: {name}")
        if not roots[name].is_dir():
            raise FileNotFoundError(f"External {name} checkout not found: {roots[name]}")
    expected = {"recstudio": RECSTUDIO_COMMIT, "autoifs": AUTOIFS_COMMIT}
    for name, commit in expected.items():
        actual = _git_head(roots[name])
        if actual != commit:
            raise ValueError(f"{name} source commit mismatch: expected {commit}; found {actual}")
    if not (roots["recstudio"] / "recstudio/model/multitask/hypergate.py").is_file():
        raise FileNotFoundError("Pinned RecStudio checkout does not contain HyperGate")
    if not (roots["autoifs"] / "modules/autoifs.py").is_file():
        raise FileNotFoundError("Pinned AutoIFS checkout does not contain modules/autoifs.py")
    return roots


def load_autoifs_class(source_root: str | Path):
    """Import the official AutoIFS class from its unmodified external checkout."""
    root = Path(source_root).expanduser().resolve()
    if _git_head(root) != AUTOIFS_COMMIT:
        raise ValueError("AutoIFS source checkout is not at the pinned commit")
    source = str(root)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        module = importlib.import_module("modules.autoifs")
    except Exception:
        sys.path.remove(source)
        raise
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        sys.path.remove(source)
        raise ImportError(f"Imported AutoIFS outside its pinned checkout: {module_path}")
    return module.AutoIFS


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned RecStudio module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_hypergate_initializer(source_root: str | Path):
    """Return the official initializer from the pinned RecStudio checkout."""
    root = Path(source_root).expanduser().resolve()
    if _git_head(root) != RECSTUDIO_COMMIT:
        raise ValueError("RecStudio source checkout is not at the pinned gitlink")
    path = root / "recstudio/model/init.py"
    module = sys.modules.get("recstudio.model.init")
    if module is None or Path(module.__file__).resolve() != path:
        module = _module("recstudio.model.init", path)
    return module.xavier_normal_initialization


def load_hypergate_class(source_root: str | Path):
    """Load the official HyperGate class and RecStudio layers without nni.

    RecStudio's root package imports optional experiment tooling (including
    NNI). This loader exposes only the pinned model/layer modules HyperGate
    needs and supplies the narrow base lifecycle that RecStudio's runner owns.
    The HyperGate equations and layers remain the upstream implementations.
    """
    root = Path(source_root).expanduser().resolve()
    if _git_head(root) != RECSTUDIO_COMMIT:
        raise ValueError("RecStudio source checkout is not at the pinned gitlink")
    base = root / "recstudio"
    if "recstudio.model.multitask.hypergate" in sys.modules:
        module = sys.modules["recstudio.model.multitask.hypergate"]
        if Path(module.__file__).resolve() == base / "model/multitask/hypergate.py":
            return module.HyperGate

    package_paths = {
        "recstudio": base,
        "recstudio.model": base / "model",
        "recstudio.model.module": base / "model/module",
        "recstudio.model.multitask": base / "model/multitask",
    }
    for name, path in package_paths.items():
        package = sys.modules.get(name)
        if package is None:
            package = type(sys)(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package

    prefix = "recstudio.model.module"
    _module("recstudio.model.init", base / "model/init.py")
    layers = _module(prefix + ".layers", base / "model/module/layers.py")
    ctr = _module(prefix + ".ctr", base / "model/module/ctr.py")
    module_package = sys.modules[prefix]
    module_package.ctr = ctr
    module_package.MLPModule = layers.MLPModule
    module_package.get_act = layers.get_act

    initializer = load_hypergate_initializer(root)

    class MultiTaskBase(nn.Module):
        def __init__(self, config=None, *args, **kwargs):
            super().__init__()
            self.config = config or {}

        def _init_model(self, train_data, drop_unused_field=True):
            self.fields = train_data.use_field
            self.frating = train_data.frating
            self.embed_dim = int(self.config["model"]["embed_dim"])
            key = self.config["eval"].get("multi_domain_key")
            self.domain_dict = train_data.field2token2idx[key] if key else None

        def _init_parameter(self):
            if self.config["train"].get("init_method") != "xavier_normal":
                raise ValueError("Pinned HyperGate lifecycle requires init_method=xavier_normal")
            for _, module in self.named_children():
                module.apply(initializer)

    base_module = type(sys)("recstudio.model.multitask._base")
    base_module._MultiTaskBase = MultiTaskBase
    sys.modules[base_module.__name__] = base_module
    multitask_package = sys.modules["recstudio.model.multitask"]
    multitask_package._base = base_module

    official = _module(
        "recstudio.model.multitask.hypergate",
        base / "model/multitask/hypergate.py",
    )
    return official.HyperGate


def recstudio_license_text(source_root: str | Path) -> str:
    path = Path(source_root) / "LICENSE"
    if not path.is_file():
        raise FileNotFoundError(f"RecStudio license not found: {path}")
    return path.read_text(encoding="utf-8")


__all__ = [
    "AUTOIFS_COMMIT", "RECSTUDIO_COMMIT", "HYPERGATE_COMMIT",
    "validate_source_roots", "load_autoifs_class", "load_hypergate_class",
    "load_hypergate_initializer",
    "recstudio_license_text",
]
