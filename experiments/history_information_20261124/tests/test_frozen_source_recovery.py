"""Exercise missing frozen sources in a disposable workspace."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from experiments.history_information_20261124 import sources


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "phase0-3"
    for relative in (
        "experiments/history_information_20261124",
        "experiments/conditional_information_momentum_20261102_v3",
        "experiments/conditional_history_shared_update_20261122",
        "experiments/mdmt_history_information_20261103",
        sources.PUBLISHED_PACKAGE,
    ):
        shutil.copytree(sources.ROOT / relative, root / relative, ignore=shutil.ignore_patterns("__pycache__"))
    for relative in (sources.PLAN.relative_to(sources.ROOT), sources.PUBLISHED_ROOT.relative_to(sources.ROOT),
                     Path("requirements-phase0-3.lock")):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources.ROOT / relative, root / relative)
    gaussian = "phase0-3_gaussian_coverage"
    shutil.copytree(sources.ROOT.with_name(gaussian) / sources.GAUSSIAN_PACKAGE,
                    root.with_name(gaussian) / sources.GAUSSIAN_PACKAGE)
    root.with_name("phase0-3_git").symlink_to(sources.ROOT.with_name("phase0-3_git"), target_is_directory=True)
    return root


def run_in_workspace(workspace, script, *arguments):
    subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, '.');\n" + script, *arguments],
                   cwd=workspace, check=True)


@pytest.mark.parametrize("filename", ("views.py", "protocol.py"))
def test_shared_dependency_recovered_before_import(workspace, filename):
    main = workspace / sources._MAIN_GA31 / filename
    published = workspace / sources._PUBLISHED_GA31 / filename
    expected = published.read_bytes()
    published.unlink()
    run_in_workspace(workspace, """
from unittest.mock import patch
from experiments.history_information_20261124 import sources

filename = sys.argv[1]
git_output = sources.subprocess.check_output

def checked_out_copy_required(command, **kwargs):
    assert command[3] != "cat-file"
    return git_output(command, **kwargs)

with patch.object(sources.subprocess, "check_output", checked_out_copy_required):
    audit = sources.audit_sources()
    from experiments.history_information_20261124 import objectives

for directory, commit in ((sources._MAIN_GA31, audit["evidence_commit"]),
                          (sources._PUBLISHED_GA31, audit["published_commit"])):
    relative = f"{directory}/{filename}"
    entry, = (row for row in audit["files"] if row["path"] == relative)
    assert entry["commit"] == commit and entry["git_blob"] == sources._SHARED_GA31_BLOBS[filename]
    assert entry["actual_path"] == str(sources.ROOT / relative)
objectives.sources.verify_shared_imports()
""", filename)
    assert main.read_bytes() == published.read_bytes() == expected


def test_unrecoverable_gaussian_source_isolated_from_clean_audit(workspace):
    coverage = workspace.with_name("phase0-3_gaussian_coverage") / sources.GAUSSIAN_PACKAGE / "objectives.py"
    coverage.unlink()
    run_in_workspace(workspace, """
import json
from types import SimpleNamespace
from unittest.mock import patch
from experiments.history_information_20261124 import sources

checkout = sources.audit_sources()
assert checkout["gaussian_source_verified"]
entry, = (row for row in checkout["gaussian_source"]["files"]
          if row["path"] == f"{sources.GAUSSIAN_PACKAGE}/objectives.py")
path = sources.Path(entry["actual_path"])
path.unlink()
git_output = sources.subprocess.check_output

def missing_coverage_blob(command, **kwargs):
    if command[-3:] == ["cat-file", "blob", entry["git_blob"]]:
        raise sources.subprocess.CalledProcessError(128, command)
    return git_output(command, **kwargs)

with patch.object(sources.subprocess, "check_output", missing_coverage_blob):
    isolated = sources.audit_sources()
    from experiments.history_information_20261124 import objectives

assert isolated["files"] == checkout["files"]
assert isolated["source_hash"] == checkout["source_hash"]
assert isolated["gaussian_source"]["status"] == "UNRESOLVED_IDENTITY"
assert entry["git_blob"] in isolated["gaussian_source"]["blocker"] and not path.exists()
assert sources.audit_sources()["gaussian_source"] == checkout["gaussian_source"]

from experiments.history_information_20261124 import kernel_checks

(sources.ROOT / "SOURCE_MANIFEST.json").write_text(json.dumps(isolated))
with patch.object(sources, "load_train_validation", return_value=SimpleNamespace(identity={"include_test": False})):
    with patch.object(kernel_checks, "_gaussian_source_contract", return_value=[{}]):
        recovered = kernel_checks.conditional_contracts("Gaussian", sources.ROOT)
assert recovered["result"] == "PASS"
assert recovered["source"]["gaussian_source_hash"] == checkout["gaussian_source"]["source_hash"]

path.write_bytes(b"unexpected existing bytes")
changed = sources.audit_sources()
assert changed["gaussian_source"]["status"] == "UNRESOLVED_IDENTITY"
assert changed["source_hash"] == checkout["source_hash"]
assert path.read_bytes() == b"unexpected existing bytes"
""")
