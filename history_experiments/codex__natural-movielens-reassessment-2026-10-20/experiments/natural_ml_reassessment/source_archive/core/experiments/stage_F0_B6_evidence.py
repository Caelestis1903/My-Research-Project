"""Bounded, post-incident evidence collection for the F0 B6 policy.

This module deliberately has no training, model, claim, release, lock-acquisition,
or production-gate API. Collection is explicit and writes only to a caller-supplied
new evidence directory. It never follows symlinks and fails closed on unstable or
unreadable scope paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "F0-B6-POSTINCIDENT-EVIDENCE-v1"
POLICY_LABEL = "POST_INCIDENT_BASELINE"
HISTORICAL_STATUS = "OPEN_FAIL_IRRECOVERABLE_GAP"
EXECUTION_AUTHORIZATION = "NO_GO"
DECLARATION = (
    "B6 历史事故及证据缺口永久保留为已披露事实。当前归档是事故后基线，只约束明确列出的路径和采集窗口，"
    "不是事故前 preimage，不证明此前完整无副作用，也不证明原科学 runner 的字节身份已恢复。后续隔离测试仅支持"
    "各自实际覆盖条件下的前瞻隔离；通过测试不补回缺失历史证据，不自动关闭 B6，不自动允许 release 或科学 RUN。"
    "是否允许任何后续受限推进，必须由独立 Reviewer 对显式政策修订、原始证据及确切执行范围作单独裁定。"
)
_BASELINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COVERAGE_CATEGORIES = {
    "current_gate_hold_release", "v14_historical_controls",
    "incident_failure_stream_evidence", "audit_logs_and_incident_slice",
    "C97_H51_state_runs_checkpoints", "F140_status_locks_runs_attempts",
    "queue_and_search_gate", "source_runtime_control_and_tests",
    "frozen_data_split_manifests",
}
_F140_ROLES = {"status", "lock", "run", "attempt"}
_QUIESCENCE_CATEGORIES = {"processes", "claims", "slots", "proc_locks"}
_F140_COVERAGE_SCHEMA = "F0-B6-F140-ABSENCE-v1"
_F140_MEMBER_COUNT = 140


def _scoped_path(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or ":" not in value:
        raise EvidenceError("evidence path must bind a scope-root id")
    root_id, relative = value.split(":", 1)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", root_id):
        raise EvidenceError("evidence path has invalid scope-root id")
    return root_id, _relative(relative)


def _validate_coverage_shape(coverage: Any) -> dict[str, Any]:
    """Require disjoint category evidence and explicit F140 member/role bindings."""
    if not isinstance(coverage, Mapping) or set(coverage) != _COVERAGE_CATEGORIES:
        raise EvidenceError("scope coverage must enumerate every policy category")
    normalized: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for category in sorted(_COVERAGE_CATEGORIES):
        values = coverage[category]
        if category == "F140_status_locks_runs_attempts":
            if (not isinstance(values, Mapping)
                    or set(values) != {"schema", "members"}
                    or values.get("schema") != _F140_COVERAGE_SCHEMA
                    or not isinstance(values.get("members"), list)
                    or len(values["members"]) != _F140_MEMBER_COUNT):
                raise EvidenceError("F140 coverage must bind exactly 140 members and four roles each")
            members = []
            member_ids = set()
            f140_paths = set()
            for member in values["members"]:
                if not isinstance(member, Mapping) or set(member) != {"member_id", "paths"}:
                    raise EvidenceError("F140 member coverage record is malformed")
                member_id = member.get("member_id")
                path_roles = member.get("paths")
                if (not isinstance(member_id, str) or not re.fullmatch(r"[0-9a-f]{20}", member_id)
                        or member_id in member_ids):
                    raise EvidenceError("F140 member ids must be unique canonical 20-hex run ids")
                if not isinstance(path_roles, Mapping) or set(path_roles) != _F140_ROLES:
                    raise EvidenceError("each F140 member must bind status, lock, run, and attempt paths")
                member_ids.add(member_id)
                normalized_roles = {}
                roots_for_member = set()
                for role in sorted(_F140_ROLES):
                    key = path_roles[role]
                    root_id, relative = _scoped_path(key)
                    expected_relative = {
                        "status": "queue_status/" + member_id + ".json",
                        "lock": "canonical_locks/" + member_id + ".lock",
                        "run": member_id,
                        "attempt": member_id + "/attempts/attempt-00",
                    }[role]
                    if relative != expected_relative:
                        raise EvidenceError("F140 {} path does not match canonical member identity".format(role))
                    roots_for_member.add(root_id)
                    if key in f140_paths:
                        raise EvidenceError("F140 absence path reused across members or roles")
                    f140_paths.add(key)
                    normalized_roles[role] = root_id + ":" + relative
                if len(roots_for_member) != 1:
                    raise EvidenceError("F140 member roles must share one scoped runs root")
                members.append({"member_id": member_id, "paths": normalized_roles})
            if len(member_ids) != _F140_MEMBER_COUNT or [m["member_id"] for m in members] != sorted(member_ids):
                raise EvidenceError("F140 members must be uniquely and deterministically sorted")
            for path in f140_paths:
                prior = owners.get(path)
                if prior is not None:
                    raise EvidenceError("coverage path reused across categories: {} / {}".format(
                        prior, category))
                owners[path] = category
            normalized[category] = {"schema": _F140_COVERAGE_SCHEMA, "members": members}
            continue
        if not isinstance(values, (list, tuple)) or not values:
            raise EvidenceError("policy coverage category has no required evidence: " + category)
        paths = [(_scoped_path(value)[0] + ":" + _scoped_path(value)[1])
                 for value in values]
        if any(not path for path in paths) or len(paths) != len(set(paths)):
            raise EvidenceError("policy coverage category has invalid or duplicate evidence: " + category)
        for path in paths:
            prior = owners.get(path)
            if prior is not None:
                raise EvidenceError("coverage path reused across categories: {} / {}".format(prior, category))
            owners[path] = category
        normalized[category] = paths
    return normalized


def _coverage_paths(coverage: Mapping[str, Any]) -> dict[str, list[str]]:
    result = {}
    for category, values in coverage.items():
        if category == "F140_status_locks_runs_attempts":
            result[category] = [member["paths"][role]
                                for member in values["members"]
                                for role in sorted(_F140_ROLES)]
        else:
            result[category] = list(values)
    return result


def _validate_quiescence_shape(path_ids: Any) -> dict[str, str]:
    if not isinstance(path_ids, Mapping) or set(path_ids) != _QUIESCENCE_CATEGORIES:
        raise EvidenceError("quiescence paths must include processes, claims, slots, and proc_locks")
    normalized = {}
    for category, value in path_ids.items():
        if not isinstance(value, str) or ":" not in value:
            raise EvidenceError("quiescence snapshot path is malformed: " + str(category))
        root_id, relative = value.split(":", 1)
        if not root_id or relative == ".":
            raise EvidenceError("quiescence snapshot requires a distinct regular-file path")
        normalized[category] = root_id + ":" + _relative(relative)
    if len(set(normalized.values())) != len(_QUIESCENCE_CATEGORIES):
        raise EvidenceError("quiescence snapshots must use four distinct paths")
    return normalized


class EvidenceError(RuntimeError):
    """Raised when the requested evidence cannot be collected safely."""


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EvidenceError("evidence is not deterministic UTF-8 JSON: {}".format(exc)) from exc


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stat_doc(st: os.stat_result) -> dict[str, int]:
    return {"device": int(st.st_dev), "inode": int(st.st_ino), "nlink": int(st.st_nlink),
            "mode": int(stat.S_IMODE(st.st_mode)), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "ctime_ns": int(st.st_ctime_ns)}


def _assert_no_symlink_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    """Inspect each component with lstat; never resolve through a symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise EvidenceError("path component is absent: {}".format(current))
        except OSError as exc:
            raise EvidenceError("cannot inspect path component {}: {}".format(current, exc)) from exc
        if stat.S_ISLNK(st.st_mode):
            raise EvidenceError("symlink path component refused: {}".format(current))


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceError("scope path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise EvidenceError("unsafe scope path: {!r}".format(value))
    normalized = path.as_posix()
    if normalized != value:
        raise EvidenceError("scope path is not canonical: {!r}".format(value))
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise EvidenceError("scope path is not valid UTF-8") from exc
    return normalized


def _same_file_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, left.st_mode, left.st_nlink, left.st_size,
            left.st_mtime_ns, left.st_ctime_ns) == (
            right.st_dev, right.st_ino, right.st_mode, right.st_nlink, right.st_size,
            right.st_mtime_ns, right.st_ctime_ns)


def _lstat_path(root: Path, relative: str) -> tuple[Path, os.stat_result | None]:
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            return current, None
        except OSError as exc:
            raise EvidenceError("lstat failed for {}: {}".format(current, exc)) from exc
        if stat.S_ISLNK(st.st_mode) and index != len(parts) - 1:
            raise EvidenceError("intermediate symlink refused: {}".format(current))
    return current, st


def _read_regular(path: Path, expected: os.stat_result) -> tuple[bytes, os.stat_result, os.stat_result]:
    if not stat.S_ISREG(expected.st_mode):
        raise EvidenceError("not a regular file: {}".format(path))
    if expected.st_nlink != 1:
        raise EvidenceError("hard-linked source refused (nlink={}): {}".format(expected.st_nlink, path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError("open failed for {}: {}".format(path, exc)) from exc
    try:
        before = os.fstat(fd)
        if not _same_file_stat(expected, before):
            raise EvidenceError("file changed between lstat and open: {}".format(path))
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        path_after = os.lstat(path)
        if not _same_file_stat(before, after) or not _same_file_stat(after, path_after):
            raise EvidenceError("file changed during read: {}".format(path))
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise EvidenceError("short/unstable read: {}".format(path))
        return data, before, after
    finally:
        os.close(fd)


def _ensure_archive_dir(root: Path, relative_dir: PurePosixPath) -> Path:
    current = root
    for part in relative_dir.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
            _fsync_dir(current.parent)
        except FileExistsError:
            st = os.lstat(current)
            if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
                raise EvidenceError("archive parent is not a real directory: {}".format(current))
    return current


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _archive_bytes(archive_root: Path, rel: PurePosixPath, data: bytes,
                   source_stat: os.stat_result | None) -> tuple[str, str, dict[str, int]]:
    parent = _ensure_archive_dir(archive_root, rel.parent)
    destination = parent / rel.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    written_stat = None
    try:
        fd = os.open(destination, flags, 0o600)
    except OSError as exc:
        raise EvidenceError("exclusive archive creation failed for {}: {}".format(destination, exc)) from exc
    try:
        pos = 0
        while pos < len(data):
            pos += os.write(fd, data[pos:])
        os.fsync(fd)
        archive_stat = os.fstat(fd)
        written_stat = archive_stat
        if archive_stat.st_nlink != 1:
            raise EvidenceError("archive file has aliases: {}".format(destination))
        if source_stat is not None and (archive_stat.st_dev, archive_stat.st_ino) == (
                source_stat.st_dev, source_stat.st_ino):
            raise EvidenceError("archive aliases source inode: {}".format(destination))
    finally:
        os.close(fd)
    _fsync_dir(parent)
    current_stat = os.lstat(destination)
    if written_stat is None or not _same_file_stat(written_stat, current_stat):
        raise EvidenceError("archive path changed after write: {}".format(destination))
    reread, _, read_after = _read_regular(destination, current_stat)
    if not _same_file_stat(current_stat, read_after):
        raise EvidenceError("archive changed during readback: {}".format(destination))
    if reread != data:
        raise EvidenceError("archive readback mismatch: {}".format(destination))
    final_stat = os.lstat(destination)
    if not stat.S_ISREG(final_stat.st_mode) or final_stat.st_nlink != 1:
        raise EvidenceError("archive is not an unaliased ordinary file: {}".format(destination))
    return rel.as_posix(), _sha(reread), _stat_doc(final_stat)


def _scan_once(*, roots: Mapping[str, Path], scope_paths: Mapping[str, Sequence[str]],
               expected_absent: set[str], protected_paths: set[str], archive_root: Path,
               observation: int) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    inode_owners: dict[tuple[int, int], str] = {}

    for root_id, root in sorted(roots.items()):
        try:
            before = os.lstat(root)
            names_before = sorted(os.listdir(root))
            for name in names_before:
                name.encode("utf-8", "strict")
            after = os.lstat(root)
            names_after = sorted(os.listdir(root))
        except (OSError, UnicodeError) as exc:
            raise EvidenceError("cannot inventory scope root {}: {}".format(root, exc)) from exc
        if (not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode)
                or not _same_file_stat(before, after) or names_before != names_after):
            raise EvidenceError("scope root changed during inventory: {}".format(root))
        root_key = root_id + ":."
        rows[root_key] = {"path": root_key, "root_id": root_id, "relative_path": ".",
                          "selected": False, "kind": "directory", "size": 0, "sha256": None,
                          "symlink_target": None, "stat_before": _stat_doc(before),
                          "stat_after": _stat_doc(after), "children": names_before,
                          "archive_rel_path": None, "archive_sha256": None}

    def record(root_id: str, relative: str, *, selected: bool = False) -> None:
        key = root_id + ":" + relative
        if key in rows:
            return
        root = roots[root_id]
        path, st = _lstat_path(root, relative)
        row: dict[str, Any] = {"path": key, "root_id": root_id, "relative_path": relative,
                               "selected": selected, "stat_before": None, "stat_after": None,
                               "archive_rel_path": None, "archive_sha256": None}
        if st is None:
            row.update({"kind": "absent", "size": 0, "sha256": None, "symlink_target": None})
            rows[key] = row
            return
        row["stat_before"] = _stat_doc(st)
        if stat.S_ISLNK(st.st_mode):
            if key in protected_paths:
                raise EvidenceError("protected control path is a symlink: {}".format(key))
            try:
                target = os.readlink(path)
                target_bytes = target.encode("utf-8", "strict")
                after = os.lstat(path)
            except (OSError, UnicodeError) as exc:
                raise EvidenceError("cannot capture symlink {}: {}".format(path, exc)) from exc
            if not _same_file_stat(st, after):
                raise EvidenceError("symlink changed during read: {}".format(path))
            archive_rel = PurePosixPath("observations") / str(observation) / "symlink-targets" / root_id / PurePosixPath(relative + ".target")
            archived, archived_sha, _ = _archive_bytes(archive_root, archive_rel, target_bytes, None)
            row.update({"kind": "symlink", "size": len(target_bytes), "sha256": _sha(target_bytes),
                        "symlink_target": target, "stat_after": _stat_doc(after),
                        "archive_rel_path": archived, "archive_sha256": archived_sha})
        elif stat.S_ISREG(st.st_mode):
            data, before, after = _read_regular(path, st)
            identity = (before.st_dev, before.st_ino)
            if identity in inode_owners:
                raise EvidenceError("source inode alias: {} and {}".format(inode_owners[identity], key))
            inode_owners[identity] = key
            archive_rel = PurePosixPath("observations") / str(observation) / "raw" / root_id / PurePosixPath(relative)
            archived, archived_sha, _ = _archive_bytes(archive_root, archive_rel, data, before)
            row.update({"kind": "regular", "size": len(data), "sha256": _sha(data),
                        "stat_before": _stat_doc(before), "stat_after": _stat_doc(after),
                        "archive_rel_path": archived, "archive_sha256": archived_sha})
        elif stat.S_ISDIR(st.st_mode):
            try:
                names_before = sorted(os.listdir(path))
                for name in names_before:
                    name.encode("utf-8", "strict")
                after = os.lstat(path)
                names_after = sorted(os.listdir(path))
            except (OSError, UnicodeError) as exc:
                raise EvidenceError("cannot enumerate directory {}: {}".format(path, exc)) from exc
            if not _same_file_stat(st, after) or names_before != names_after:
                raise EvidenceError("directory changed during enumeration: {}".format(path))
            row.update({"kind": "directory", "size": 0, "sha256": None,
                        "symlink_target": None, "stat_after": _stat_doc(after),
                        "children": names_before})
            rows[key] = row
            for name in names_before:
                child = relative + "/" + name
                record(root_id, child)
            return
        else:
            raise EvidenceError("unsupported filesystem object at {}".format(path))
        rows[key] = row

    for root_id in sorted(scope_paths):
        if root_id not in roots:
            raise EvidenceError("scope path references unknown root: {}".format(root_id))
        for value in scope_paths[root_id]:
            record(root_id, _relative(value), selected=True)
    for row in rows.values():
        key = row["path"]
        if row["kind"] == "absent" and key not in expected_absent:
            row["error"] = "UNEXPECTED_ABSENCE"
        elif row["kind"] != "absent" and key in expected_absent:
            row["error"] = "EXPECTED_ABSENCE_PATH_EXISTS"
    return [rows[key] for key in sorted(rows)]


def _stability_projection(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = ("path", "kind", "size", "sha256", "symlink_target", "stat_before", "stat_after", "children", "error")
    return _json_bytes([{name: row.get(name) for name in fields} for row in rows])


def _write_manifest(archive_root: Path, manifest: Mapping[str, Any]) -> str:
    data = _json_bytes(manifest)
    path = archive_root / "manifest.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        pos = 0
        while pos < len(data):
            pos += os.write(fd, data[pos:])
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(archive_root)
    manifest_data, _, _ = _read_regular(path, os.lstat(path))
    if manifest_data != data:
        raise EvidenceError("manifest readback mismatch")
    return _sha(data)


def collect_postincident_baseline(*, baseline_id: str, scope_roots: Mapping[str, str | Path],
                                  scope_paths: Mapping[str, Sequence[str]], archive_parent: str | Path,
                                  forbidden_roots: Sequence[str | Path], required_paths: Iterable[str],
                                  expected_absent: Iterable[str], protected_paths: Iterable[str],
                                  policy_path: str | Path, repo_path: str | Path,
                                  collector_command: Sequence[str],
                                  environment: Mapping[str, str], history_missing: Sequence[str],
                                  residual_risks: Sequence[str],
                                  quiescence_paths: Mapping[str, str],
                                  scope_coverage: Mapping[str, Any],
                                  exclusions: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    """Collect two complete inventories and ordinary-file raw archives.

    `scope_paths` maps each root id to explicit relative paths. Directories are
    enumerated recursively without following symlinks. An absent path is accepted
    only when listed in `expected_absent`; required paths must be present.
    This function never acquires locks or mutates any path beneath a scope root.
    """
    if not _BASELINE_ID.fullmatch(baseline_id):
        raise EvidenceError("invalid baseline_id")
    if not isinstance(environment, Mapping) or not isinstance(collector_command, Sequence):
        raise EvidenceError("collector metadata is malformed")
    start_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    roots: dict[str, Path] = {}
    for root_id, value in scope_roots.items():
        if not root_id or not re.fullmatch(r"[A-Za-z0-9._-]+", root_id):
            raise EvidenceError("invalid root id")
        root = Path(os.path.abspath(os.fspath(value)))
        _assert_no_symlink_components(root)
        st = os.lstat(root)
        if not stat.S_ISDIR(st.st_mode):
            raise EvidenceError("scope root is not a real directory: {}".format(root))
        roots[root_id] = root
    if set(scope_paths) != set(roots):
        raise EvidenceError("scope_paths must define every and only declared scope root")
    if not any(scope_paths.values()):
        raise EvidenceError("empty scope is not a baseline")
    all_keys = {root_id + ":" + _relative(rel) for root_id, values in scope_paths.items() for rel in values}
    def covered(key: str) -> bool:
        return key in all_keys or any(key.startswith(base + "/") for base in all_keys)
    expected_absent_set = set(expected_absent)
    required_set = set(required_paths)
    protected_set = set(protected_paths)
    quiescence = _validate_quiescence_shape(quiescence_paths)
    quiescence_set = set(quiescence.values())
    coverage_lists = _validate_coverage_shape(scope_coverage)
    coverage_path_lists = _coverage_paths(coverage_lists)
    coverage = {category: set(paths) for category, paths in coverage_path_lists.items()}
    coverage_values = set().union(*coverage.values()) if coverage else set()
    if (not coverage_values or not all(covered(key) for key in coverage_values)
            or not coverage_values <= required_set):
        raise EvidenceError("every policy coverage path must be an explicit required evidence path")
    f140_paths = coverage["F140_status_locks_runs_attempts"]
    if not f140_paths or not f140_paths <= expected_absent_set:
        raise EvidenceError("all F140 status/lock/run/attempt paths must be explicitly expected absent")
    if (not all(covered(key) for key in expected_absent_set)
            or not all(covered(key) for key in required_set)
            or not all(covered(key) for key in protected_set)
            or not all(covered(key) for key in quiescence_set)):
        raise EvidenceError("required, absent, and protected sets must name explicit scope entries")
    if not quiescence_set <= required_set:
        raise EvidenceError("all quiescence snapshots must be required scope paths")
    if quiescence_set & coverage_values:
        raise EvidenceError("quiescence snapshots cannot be reused as policy coverage evidence")
    archive_parent_path = Path(os.path.abspath(os.fspath(archive_parent)))
    for root in roots.values():
        if archive_parent_path == root or archive_parent_path in root.parents or root in archive_parent_path.parents:
            raise EvidenceError("archive parent overlaps a scoped root")
    for forbidden in forbidden_roots:
        fpath = Path(os.path.abspath(os.fspath(forbidden)))
        _assert_no_symlink_components(fpath)
        if fpath == archive_parent_path or fpath in archive_parent_path.parents:
            raise EvidenceError("archive parent is inside a forbidden production tree")
    _assert_no_symlink_components(archive_parent_path)
    parent_stat = os.lstat(archive_parent_path)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise EvidenceError("archive parent must be an existing real directory")
    archive_root = archive_parent_path / baseline_id
    try:
        archive_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise EvidenceError("baseline archive already exists; refusing overwrite") from exc
    _fsync_dir(archive_parent_path)
    if os.path.samefile(archive_root, archive_parent_path):
        raise EvidenceError("archive root aliases its parent")

    errors: list[str] = []
    observations = []
    try:
        for number in (1, 2):
            rows = _scan_once(roots=roots, scope_paths=scope_paths,
                              expected_absent=expected_absent_set,
                              protected_paths=protected_set, archive_root=archive_root,
                              observation=number)
            observations.append(rows)
        stable = _stability_projection(observations[0]) == _stability_projection(observations[1])
        all_rows = observations[0]
        for row in all_rows:
            if row.get("error"):
                errors.append("{}:{}".format(row["path"], row["error"]))
        row_by_path = {row["path"]: row for row in all_rows}
        for category, paths in coverage.items():
            for key in sorted(paths):
                row = row_by_path.get(key)
                if row is None:
                    errors.append("COVERAGE_PATH_MISSING:{}:{}".format(category, key))
                elif category == "F140_status_locks_runs_attempts":
                    if row.get("kind") != "absent" or key not in expected_absent_set:
                        errors.append("F140_COVERAGE_NOT_ABSENT:{}".format(key))
                elif row.get("kind") != "regular" or not row.get("sha256"):
                    errors.append("COVERAGE_EVIDENCE_NOT_REGULAR:{}:{}".format(category, key))
        for category, key in quiescence.items():
            row = row_by_path.get(key)
            if (row is None or row.get("kind") != "regular" or not row.get("sha256")
                    or not row.get("archive_rel_path") or not row.get("archive_sha256")
                    or not isinstance(row.get("stat_before"), Mapping)
                    or not isinstance(row.get("stat_after"), Mapping)):
                errors.append("QUIESCENCE_SNAPSHOT_INVALID:{}:{}".format(category, key))
        present = {row["path"] for row in all_rows if row["kind"] != "absent"}
        missing_required = sorted(required_set - present - expected_absent_set)
        if missing_required:
            errors.append("MISSING_REQUIRED:" + ",".join(missing_required))
        if not stable:
            errors.append("DOUBLE_READ_UNSTABLE")
        second_rows = observations[1]
        tool_path = Path(os.path.abspath(__file__))
        _assert_no_symlink_components(tool_path)
        tool_data, _, _ = _read_regular(tool_path, os.lstat(tool_path))
        policy_file = Path(os.path.abspath(os.fspath(policy_path)))
        _assert_no_symlink_components(policy_file)
        policy_data, _, _ = _read_regular(policy_file, os.lstat(policy_file))
        repo = Path(os.path.abspath(os.fspath(repo_path)))
        _assert_no_symlink_components(repo)
        repo_stat = os.lstat(repo)
        if not stat.S_ISDIR(repo_stat.st_mode):
            raise EvidenceError("repo_path must be a real directory")
        normalized_exclusions = []
        for item in exclusions:
            if set(item) != {"path", "reason"} or not item["path"] or not item["reason"]:
                raise EvidenceError("each exclusion requires an exact path and reason")
            normalized_exclusions.append({"path": str(item["path"]), "reason": str(item["reason"])})
        manifest: dict[str, Any] = {
            "schema": SCHEMA, "baseline_id": baseline_id, "label": POLICY_LABEL,
            "collected_start_utc": start_utc,
            "collected_end_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "collector": {"command": list(collector_command), "environment": dict(environment),
                          "host": platform.node(), "repo_path": str(repo),
                          "tool_path": str(tool_path), "tool_sha256": _sha(tool_data),
                          "policy_path": str(policy_file), "policy_sha256": _sha(policy_data)},
            "scope_roots": {key: str(value) for key, value in sorted(roots.items())},
            "scope_paths": {key: sorted(_relative(item) for item in values)
                            for key, values in sorted(scope_paths.items())},
            "required_paths": sorted(required_set),
            "protected_paths": sorted(protected_set),
            "scope_coverage": {key: (coverage_lists[key] if key == "F140_status_locks_runs_attempts"
                                      else sorted(coverage_lists[key]))
                               for key in sorted(coverage_lists)},
            "exclusions": normalized_exclusions, "absence_paths": sorted(expected_absent_set),
            "inventory_complete": not errors,
            "inventory": all_rows,
            "second_read_inventory": second_rows,
            "inventory_sha256": _sha(_json_bytes(all_rows)),
            "second_inventory_sha256": _sha(_json_bytes(second_rows)),
            "stable_double_read": stable,
            "historical_B6_status": HISTORICAL_STATUS,
            "postincident_baseline_status": ("COLLECTED_UNREVIEWED" if stable and not errors
                                               else "REJECTED_UNSTABLE" if not stable
                                               else "REJECTED"),
            "prospective_qualification_status": "RECORDED_UNREVIEWED",
            "forward_disposition_status": "NOT_GRANTED",
            "execution_authorization": EXECUTION_AUTHORIZATION,
            "historical_no_side_effects_proven": False,
            "historical_full_equality_proven": False,
            "original_runner_byte_identity_proven": False,
            "history_missing": list(history_missing),
            "historical_declaration": DECLARATION,
            "residual_risks": list(residual_risks),
            "quiescence_observations": {
                "path_ids": dict(sorted(quiescence.items())),
                "observations": {category: next(row for row in all_rows if row["path"] == key)
                                 for category, key in sorted(quiescence.items())
                                 if key in row_by_path},
                "lock_acquired": False, "claim_or_slot_mutation": False},
            "errors": errors,
        }
        manifest_sha = _write_manifest(archive_root, manifest)
        return {"archive_path": str(archive_root), "manifest_path": str(archive_root / "manifest.json"),
                "manifest_sha256": manifest_sha, "status": manifest["postincident_baseline_status"],
                "execution_authorization": EXECUTION_AUTHORIZATION, "errors": errors}
    except Exception:
        # Keep any already-written raw observation as evidence; never clean it up.
        raise


def validate_postincident_manifest(path: str | Path) -> dict[str, Any]:
    """Validate a persisted archive without modifying it or following links."""
    manifest_path = Path(os.path.abspath(os.fspath(path)))
    _assert_no_symlink_components(manifest_path)
    try:
        st = os.lstat(manifest_path)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise EvidenceError("manifest must be an unaliased ordinary file")
        raw, _, _ = _read_regular(manifest_path, st)
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("manifest is unreadable or malformed: {}".format(exc)) from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise EvidenceError("wrong evidence schema")
    if (doc.get("historical_B6_status") != HISTORICAL_STATUS
            or doc.get("execution_authorization") != EXECUTION_AUTHORIZATION
            or doc.get("historical_no_side_effects_proven") is not False
            or doc.get("historical_full_equality_proven") is not False
            or doc.get("original_runner_byte_identity_proven") is not False):
        raise EvidenceError("manifest violates non-authorization or history constraints")
    if doc.get("label") != POLICY_LABEL or doc.get("historical_declaration") != DECLARATION:
        raise EvidenceError("manifest policy declaration mismatch")
    status = doc.get("postincident_baseline_status")
    if status == "REJECTED_UNSTABLE":
        raise EvidenceError("baseline rejected: double-read inventory is unstable")
    if status == "REJECTED":
        raise EvidenceError("baseline rejected: required evidence is absent or inconsistent")
    if status != "COLLECTED_UNREVIEWED":
        raise EvidenceError("unknown baseline status")
    try:
        coverage_lists = _validate_coverage_shape(doc.get("scope_coverage"))
        qsection = doc.get("quiescence_observations")
        quiescence = _validate_quiescence_shape(
            qsection.get("path_ids") if isinstance(qsection, Mapping) else None)
    except EvidenceError as exc:
        raise EvidenceError("manifest coverage/quiescence schema invalid: {}".format(exc)) from exc
    coverage_path_lists = _coverage_paths(coverage_lists)
    coverage = {category: set(paths) for category, paths in coverage_path_lists.items()}
    coverage_values = set().union(*coverage.values())
    quiescence_values = set(quiescence.values())
    if coverage_values & quiescence_values:
        raise EvidenceError("quiescence snapshots reuse policy coverage paths")
    inventory_rows = doc.get("inventory", [])
    inventory_by_path = {row.get("path"): row for row in inventory_rows if isinstance(row, dict)}
    if not coverage_values <= set(inventory_by_path):
        raise EvidenceError("manifest coverage references paths outside inventory")
    if not quiescence_values <= set(inventory_by_path):
        raise EvidenceError("manifest quiescence snapshots reference paths outside inventory")
    if not set(coverage["F140_status_locks_runs_attempts"]) <= set(doc.get("absence_paths", [])):
        raise EvidenceError("F140 historical paths are not explicitly recorded absent")
    required_paths = set(doc.get("required_paths", []))
    absence_paths = set(doc.get("absence_paths", []))
    protected_paths = set(doc.get("protected_paths", []))
    if not coverage_values <= required_paths or not quiescence_values <= required_paths:
        raise EvidenceError("coverage and quiescence snapshots must be explicit required evidence")
    if not required_paths <= set(inventory_by_path):
        raise EvidenceError("required paths are missing from inventory")
    if not protected_paths <= set(inventory_by_path):
        raise EvidenceError("protected paths are missing from inventory")
    for category, keys in coverage.items():
        for key in keys:
            row = inventory_by_path[key]
            expected_kind = ("absent" if category == "F140_status_locks_runs_attempts"
                             else "regular")
            if row.get("kind") != expected_kind:
                raise EvidenceError("policy category lacks required archived evidence: {}:{}".format(
                    category, key))
            if expected_kind == "regular" and (not row.get("sha256")
                    or not row.get("archive_rel_path") or not row.get("archive_sha256")
                    or not isinstance(row.get("stat_before"), Mapping)
                    or not isinstance(row.get("stat_after"), Mapping)):
                raise EvidenceError("policy category evidence is not hash/inventory bound: {}".format(key))
    qsection = doc.get("quiescence_observations")
    if (not isinstance(qsection, Mapping)
            or set(qsection) != {"path_ids", "observations", "lock_acquired",
                                 "claim_or_slot_mutation"}
            or qsection.get("lock_acquired") is not False
            or qsection.get("claim_or_slot_mutation") is not False):
        raise EvidenceError("manifest lacks the complete read-only quiescence section")
    qobservations = qsection.get("observations")
    if not isinstance(qobservations, Mapping) or set(qobservations) != _QUIESCENCE_CATEGORIES:
        raise EvidenceError("manifest lacks all four quiescence snapshot observations")
    for category, key in quiescence.items():
        row = inventory_by_path[key]
        observed = qobservations.get(category)
        if (row.get("kind") != "regular" or not row.get("sha256")
                or not row.get("archive_rel_path") or not row.get("archive_sha256")
                or not isinstance(row.get("stat_before"), Mapping)
                or not isinstance(row.get("stat_after"), Mapping)
                or row.get("stat_before") != row.get("stat_after")
                or observed != row):
            raise EvidenceError("quiescence snapshot is not a distinct stable archived file: {}".format(
                category))
    for key, row in inventory_by_path.items():
        if row.get("kind") == "absent" and key not in absence_paths:
            raise EvidenceError("inventory contains an undeclared absence")
        if key in absence_paths and row.get("kind") != "absent":
            raise EvidenceError("declared absent path is present or a symlink")
        if key in protected_paths and row.get("kind") == "symlink":
            raise EvidenceError("protected path is a symlink")
    if doc.get("postincident_baseline_status") == "COLLECTED_UNREVIEWED":
        if (not doc.get("stable_double_read") or not doc.get("inventory_complete")
                or doc.get("errors") or doc.get("inventory_sha256") != _sha(_json_bytes(doc.get("inventory")))):
            raise EvidenceError("claimed stable baseline is incomplete or digest-mismatched")
        if doc.get("second_inventory_sha256") != _sha(_json_bytes(doc.get("second_read_inventory"))):
            raise EvidenceError("second inventory digest mismatch")
        if _stability_projection(doc["inventory"]) != _stability_projection(doc["second_read_inventory"]):
            raise EvidenceError("two inventories do not match")
        for name in ("inventory", "second_read_inventory"):
            paths = [row.get("path") for row in doc[name]]
            if paths != sorted(paths) or len(paths) != len(set(paths)):
                raise EvidenceError("inventory paths are not uniquely sorted")
        archive_root = manifest_path.parent
        archived_inodes: set[tuple[int, int]] = set()
        scope_roots = doc.get("scope_roots")
        if not isinstance(scope_roots, Mapping):
            raise EvidenceError("manifest scope roots are missing")
        for observation_rows in (doc["inventory"], doc["second_read_inventory"]):
            for row in observation_rows:
                if row.get("kind") not in ("regular", "symlink"):
                    continue
                rel = PurePosixPath(row["archive_rel_path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise EvidenceError("unsafe archive path in manifest")
                raw_path = archive_root / rel
                _assert_no_symlink_components(raw_path)
                raw_stat = os.lstat(raw_path)
                if not stat.S_ISREG(raw_stat.st_mode):
                    raise EvidenceError("archived raw is not an ordinary file")
                identity = (raw_stat.st_dev, raw_stat.st_ino)
                source_before = row.get("stat_before")
                if (not isinstance(source_before, Mapping)
                        or (identity[0], identity[1]) ==
                           (source_before.get("device"), source_before.get("inode"))):
                    raise EvidenceError("archive inode aliases recorded source inode")
                root_id = row.get("root_id")
                relative = row.get("relative_path")
                root_value = scope_roots.get(root_id)
                if not isinstance(root_value, str) or not isinstance(relative, str):
                    raise EvidenceError("archived source identity is malformed")
                source_root = Path(root_value)
                _assert_no_symlink_components(source_root)
                source_path, current_source_stat = _lstat_path(source_root, relative)
                if current_source_stat is None:
                    raise EvidenceError("archived source path is now absent")
                if ((current_source_stat.st_dev, current_source_stat.st_ino) == identity
                        or _stat_doc(current_source_stat) != dict(source_before)):
                    raise EvidenceError("archive/source inode or source metadata identity mismatch")
                if raw_stat.st_nlink != 1:
                    raise EvidenceError("archived raw is not an unaliased ordinary file")
                if identity in archived_inodes:
                    raise EvidenceError("archive raw inode alias")
                archived_inodes.add(identity)
                data, _, _ = _read_regular(raw_path, raw_stat)
                if _sha(data) != row.get("archive_sha256"):
                    raise EvidenceError("archive raw SHA mismatch")
                if row["kind"] == "regular" and _sha(data) != row.get("sha256"):
                    raise EvidenceError("archived source bytes mismatch")
                if row["kind"] == "symlink" and data.decode("utf-8") != row.get("symlink_target"):
                    raise EvidenceError("archived symlink target mismatch")
    return doc
