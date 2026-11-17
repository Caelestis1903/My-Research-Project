"""Authorized C05 post-readback continuation; never rebuilds the validated tar."""
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import time
from datetime import datetime, timezone

W = Path('/home/xzf/Code/phase0-3_tvg_rm_20261116')
R = Path('/tmp/phase0-3-c05-01a1e9a9/attempt1')
S = R.parent / 'publication'
A = W / 'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05'
N = A / 'attempt1'
N4 = A.parent / 'cycle_04/attempt2'
ARCHIVE = S / 'RAW_RUNS.tar.gz'
ARCHIVE_SHA = 'fa08b88dd6d488b48f1810a4e900e680ba3a35e577d2118f6b68accf92e44250'
SOURCE_SHA = 'f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d'
TABLES_SHA = '79d9673e032a4b0eab4bff575fa9594e11bfcb972975a68da53a3a5c6b804965'
NATIVE = '01a1eac7-b77a-7082-a2ff-e3e8d7b4d156'
EXPECTED_COUNTS = dict(raw=2381, source=55, control=1530, design=7, receipt=53)


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    # Receipts are the sole mutable files in this transaction besides C05 outputs.
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def regular(path):
    path = Path(path)
    check(all(not x.is_symlink() for x in (path, *path.parents)), 'Symlink path: ' + str(path))
    check(stat.S_ISREG(path.stat().st_mode), 'Not a regular file: ' + str(path))
    return path


def verify(path, row):
    regular(path)
    check(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Byte mismatch: ' + str(path))


def milestone(name, **fields):
    print(json.dumps(dict(milestone=name, pid=os.getpid(), **fields)), flush=True)


def main():
    started = time.monotonic()
    milestone('START', native_id=NATIVE)
    sys.path.insert(0, str(R / 'source_snapshot'))
    from experiments.cell_mass_bce_20261117 import tables
    p = tables.p
    check(sha(Path(tables.__file__)) == TABLES_SHA, 'Frozen tables identity differs')
    check(sha(W / 'experiments/cell_mass_bce_20261117/tables.py') == TABLES_SHA, 'Live tables identity differs')
    check(not p.stop_requested(), 'USER_STOPPED')
    proof_path = A / 'RAW_ARCHIVE_COMPLETION.json'
    original = load(proof_path)
    check(original['status'] == 'PAUSED_CAPACITY_BLOCKER', 'Continuation must start at known capacity blocker')
    check(original['pack']['invocations'] == 1 and original['pack']['pid'] == 3060800, 'Original pack proof differs')
    check(original['archive']['sha256'] == ARCHIVE_SHA and original['archive']['member_count'] == 4026, 'Original archive proof differs')
    check(original['archive']['full_content_readback'].startswith('PASS'), 'Original full readback proof absent')
    original_proof = dict(path=str(proof_path), sha256=sha(proof_path), pack=original['pack'], archive=original['archive'])
    verify(ARCHIVE, dict(bytes=677004657, sha256=ARCHIVE_SHA))
    archive_stat = ARCHIVE.stat()
    archived = {}
    with tarfile.open(ARCHIVE, 'r|gz') as stream:
        for header in stream:
            name = tables.member_name(header.name)
            check(header.isfile() and name not in archived, 'Nonregular or duplicate tar member')
            digest = hashlib.sha256()
            count = 0
            with stream.extractfile(header) as content:
                for chunk in iter(lambda: content.read(1024 * 1024), b''):
                    digest.update(chunk)
                    count += len(chunk)
            check(count == header.size, 'Tar member size mismatch')
            archived[name] = dict(bytes=count, sha256=digest.hexdigest())
    check(len(archived) == 4026, 'Tar member count differs')
    milestone('ARCHIVE_SHA_AND_ALL_MEMBER_STREAM_PASS', members=len(archived))

    def byte_checked_member(path, name, category, *, root, expected_sha=None):
        path = p.contained(path, root)
        name = tables.member_name(name)
        regular(path)
        check(path.name not in tables.PRIVATE_NAMES and not any(s in tables.OPAQUE_SUFFIXES for s in path.suffixes), 'Private or opaque archive file')
        check(name in archived, 'New file not in validated archive: ' + name)
        actual = sha(path)
        size = path.stat().st_size
        check(archived[name] == dict(bytes=size, sha256=actual), 'Archived/current bytes differ: ' + name)
        check(expected_sha is None or expected_sha == actual, 'Expected hash differs: ' + name)
        return dict(member=name, source_path=str(path), category=category, bytes=size, sha256=actual)

    namespace = dict(vars(tables))
    namespace['checked_member'] = byte_checked_member
    selection_text = inspect.getsource(tables.archive_members)
    exec(compile(selection_text, '<frozen archive_members with identical-byte metadata builder>', 'exec'), namespace)
    members = namespace['archive_members'](R)
    check([r['member'] for r in members] == sorted(archived), 'Exact current selection differs')
    counts = dict(collections.Counter(r['category'] for r in members))
    check(counts == EXPECTED_COUNTS, 'Archive categories differ')
    need = sum(r['bytes'] for r in members)
    check(need == original['archive']['uncompressed_member_bytes'], 'Uncompressed bytes differ')
    milestone('EXACT_CURRENT_SELECTION_BYTE_PROOF_PASS', counts=counts, bytes=need)
    pack_source = inspect.getsource(tables.pack)
    marker = 'size=archive.stat().st_size'
    check(pack_source.count(marker) == 1, 'Tail boundary differs')
    literal_tail = pack_source[pack_source.index(marker):]
    tail_text = '    ' + literal_tail
    wrapper = 'def frozen_remaining_tail(a, manifest, archive, members, need):\n' + tail_text
    tail_namespace = dict(vars(tables))
    exec(compile(wrapper, '<exact frozen pack remaining tail>', 'exec'), tail_namespace)
    check('write_archive' not in literal_tail and 'validate_attempt' not in literal_tail, 'Tail includes completed work')
    for path in (N/'RAW_RUNS_MANIFEST.json', N/'RAW_RUNS_PART_INDEX.json', S/'RAW_RUNS_MANIFEST.json', S/'RAW_RUNS_PART_INDEX.json'):
        check(not path.exists(), 'Output already exists: ' + str(path))
    check(not list(N.glob('RAW_RUNS.tar.gz.part-*')) and not list(S.glob('RAW_RUNS.tar.gz.part-*')), 'Parts already exist')
    manifest4_path = N4/'RAW_RUNS_MANIFEST.json'
    manifest4 = load(manifest4_path)
    publication_path = N4.parent/'PUBLICATION_RECEIPT.json'
    publication4 = load(publication_path)
    check(publication4['status'] == 'PUBLISHED_AND_REMOTE_VERIFIED' and publication4['commit'] == publication4['remote_verified_sha'] == 'cbccf5ae234e46bc3f976467de2b3327139f021b', 'C04 publication proof differs')
    published4 = {r['path']: r for r in publication4['files']}
    for row in manifest4['parts']:
        path = N4/row['path']
        check(path.parent == N4 and path.name.startswith('RAW_RUNS.tar.gz.part-'), 'C04 part path differs')
        verify(path, row)
        remote = published4[str(path.relative_to(W))]
        check(remote['sha256'] == row['sha256'] and remote['bytes'] == row['bytes'], 'C04 part lacks published byte proof')
    old4 = dict(manifest_sha256=sha(manifest4_path), publication_sha256=sha(publication_path), part_count=len(manifest4['parts']))
    required = 3*677004657 + 256*1024**2 + 1024**3
    check(required == 3373191251, 'Real gate requirement differs')
    reservation = dict(status='RESERVING', native_id=NATIVE, pid=os.getpid(), required_workspace_free_bytes=required,
        safety_bytes=64*1024**2, free_before=shutil.disk_usage(W).free, moves=[], c04=old4, test_accessed=False)
    staged = S/'temporary_c04_parts'
    staged.mkdir(exist_ok=True)
    result = None
    failure = None
    restoration_errors = []
    try:
        for row in manifest4['parts']:
            if shutil.disk_usage(W).free >= required + reservation['safety_bytes']:
                break
            src = N4/row['path']
            dst = staged/row['path']
            check(not dst.exists(), 'Temporary C04 destination exists')
            verify(src, row)
            before = shutil.disk_usage(W).free
            shutil.copy2(src, dst)
            verify(dst, row)
            move = dict(original_path=str(src), staged_path=str(dst), bytes=row['bytes'], sha256=row['sha256'],
                free_before=before, original_removed=False, restored=False)
            reservation['moves'].append(move)
            save(A/'ARCHIVE_SPACE_RESERVATION_04.json', reservation)
            src.unlink()
            move.update(original_removed=True, free_after=shutil.disk_usage(W).free)
            save(A/'ARCHIVE_SPACE_RESERVATION_04.json', reservation)
        reservation.update(status='RESERVED', free_at_reservation=shutil.disk_usage(W).free)
        check(reservation['free_at_reservation'] >= required + reservation['safety_bytes'], 'Authorized C04 parts insufficient')
        save(A/'ARCHIVE_SPACE_RESERVATION_04.json', reservation)
        milestone('REAL_CAPACITY_RESERVED', moved_parts=len(reservation['moves']), free=reservation['free_at_reservation'], required=required)
        check(not p.stop_requested(), 'USER_STOPPED before continuation')
        manifest = p.validate_attempt(R, True)
        check(manifest['source_hash'] == SOURCE_SHA, 'Validated source identity differs')
        acceptance = p.read_json(R/'PARENT_STORAGE_ACCEPTANCE.json')
        p.require(acceptance['parent_accepted'] is True and acceptance['source_hash']==manifest['source_hash'] and acceptance['design_bundle_sha256']==p.DESIGN_HASH and acceptance['test_accessed'] is False,'Parent storage acceptance required before pack')
        p.require(acceptance['owner']['endpoint_hash']==p.owner_binding(R)['endpoint_hash'],'Pack owner binding differs')
        p.require(p.read_json(R/'tables/TABLE_RECEIPT.json')['source_hash']==manifest['source_hash'],'Raw export required before pack')
        reservation['free_immediately_before_exact_tail'] = shutil.disk_usage(W).free
        check(reservation['free_immediately_before_exact_tail'] >= required, 'Actual real 3A gate cannot pass')
        save(A/'ARCHIVE_SPACE_RESERVATION_04.json', reservation)
        milestone('VALIDATE_TRUE_PASS_EXACT_TAIL_START', free=reservation['free_immediately_before_exact_tail'])
        result = tail_namespace['frozen_remaining_tail'](R, manifest, ARCHIVE, members, need)
        milestone('EXACT_FROZEN_TAIL_RETURNED', result=result)
    except BaseException as exc:
        failure = exc
    finally:
        for move in reservation['moves']:
            src, dst = Path(move['original_path']), Path(move['staged_path'])
            try:
                verify(dst, move)
                if src.exists():
                    verify(src, move)
                else:
                    shutil.copy2(dst, src)
                    verify(src, move)
                move.update(restored=True, restored_sha256=sha(src), free_after_restore=shutil.disk_usage(W).free)
                save(A/'ARCHIVE_SPACE_RESERVATION_04.json', reservation)
            except BaseException as exc:
                restoration_errors.append(dict(path=str(src), error=repr(exc)))
        reservation.update(status='RESTORED' if not restoration_errors else 'RESTORE_BLOCKED', free_after_restore=shutil.disk_usage(W).free, restoration_errors=restoration_errors)
        save(A/'ARCHIVE_SPACE_RESERVATION_04.json', reservation)
        milestone('C04_RESTORATION', status=reservation['status'], moved_parts=len(reservation['moves']), free=reservation['free_after_restore'])
    check(not restoration_errors, 'C04 restoration failure: ' + repr(restoration_errors))
    for row in manifest4['parts']:
        verify(N4/row['path'], row)
    check(sha(manifest4_path) == old4['manifest_sha256'] and sha(publication_path) == old4['publication_sha256'], 'C04 metadata changed')
    if failure is not None:
        save(A/'ARCHIVE_CONTINUATION_04.json', dict(status='PAUSED_CONTINUATION_BLOCKER', failure=repr(failure), reservation=reservation, test_accessed=False))
        raise failure
    check(result == dict(status='PACKED', archive_sha256=ARCHIVE_SHA, archive_bytes=677004657, parts=81, test_accessed=False), 'Tail result differs')
    receipt = load(N/'RAW_RUNS_MANIFEST.json')
    index = load(N/'RAW_RUNS_PART_INDEX.json')
    check(receipt == load(S/'RAW_RUNS_MANIFEST.json') and index == load(S/'RAW_RUNS_PART_INDEX.json'), 'Staged/logical metadata differs')
    check(receipt['members'] == members and receipt['member_count']==4026 and len(receipt['parts'])==81 and receipt['source_hash']==SOURCE_SHA and receipt['test_accessed'] is False, 'Manifest member closure differs')
    check(index == dict(archive_sha256=ARCHIVE_SHA, archive_bytes=677004657, parts=receipt['parts'], test_accessed=False), 'Part index differs')
    combined = hashlib.sha256()
    total = 0
    for i, row in enumerate(receipt['parts']):
        check(row['path'] == f'RAW_RUNS.tar.gz.part-{i:02d}', 'Part order differs')
        for root in (N, S):
            verify(root/row['path'], row)
        with (N/row['path']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                combined.update(chunk)
                total += len(chunk)
    check(total == 677004657 and combined.hexdigest() == ARCHIVE_SHA, 'Ordered concatenation differs')
    verify(ARCHIVE, dict(bytes=677004657, sha256=ARCHIVE_SHA))
    check((ARCHIVE.stat().st_ino, ARCHIVE.stat().st_mtime_ns) == (archive_stat.st_ino, archive_stat.st_mtime_ns), 'Original archive replaced or modified')
    for row in members:
        if row['category']=='source':
            verify(Path(row['source_path']), row)
            verify(W/Path(row['source_path']).relative_to(R/'source_snapshot'), row)
    check(sha(Path(tables.__file__)) == TABLES_SHA and sha(W/'experiments/cell_mass_bce_20261117/tables.py') == TABLES_SHA, 'Frozen pack source changed')
    for suffix in ('json', 'md'):
        src = A/f'RAW_ARCHIVE_COMPLETION.{suffix}'
        dst = A/f'RAW_ARCHIVE_COMPLETION_03.{suffix}'
        check(not dst.exists(), 'Original completion preservation destination exists')
        shutil.copy2(src, dst)
        check(sha(src)==sha(dst), 'Original completion preservation differs')
    original_proof['preserved_path'] = str(A/'RAW_ARCHIVE_COMPLETION_03.json')
    report = dict(status='PACKED_CONTINUATION_COMPLETE', native_id=NATIVE, parent_id='01a1dfb8-605b-7430-8462-5fd3da0b048f', fork_context=False,
        pid=os.getpid(), recorded_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        source_hash=SOURCE_SHA, source_files=55, source_changes=False, guard_changes=False, standard_pack_cli_reran=False,
        archive_rebuilt=False, semantic_proof='validatedsemanticproofreusedonlyforidenticalarchived/currentbytes',
        original_readback_proof=original_proof, frozen_tables_sha256=TABLES_SHA,
        literal_tail_sha256=hashlib.sha256(literal_tail.encode()).hexdigest(), tail_text_sha256=hashlib.sha256(tail_text.encode()).hexdigest(),
        tail_text=tail_text, tail_boundary=marker, selection_function_sha256=hashlib.sha256(selection_text.encode()).hexdigest(),
        script_path=str(Path(__file__).resolve()), script_sha256=sha(Path(__file__)),
        member_byte_proof=dict(status='PASS', member_count=len(members), category_counts=counts, uncompressed_bytes=need,
            exact_names_and_current_file_hashes_match_streamed_tar=True, members_metadata_sha256=p.digest(members)),
        actual_validate_attempt_require_engineering='PASS (one invocation; status-only existing owner IPC)',
        archive=dict(path=str(ARCHIVE), sha256=ARCHIVE_SHA, bytes=total, member_count=4026, parts=81, category_counts=counts,
            ordered_part_concatenation='PASS', staged_and_logical_part_readback='PASS', original_inode_and_mtime_unchanged=True),
        reservation=reservation, c04_all_original_parts_restored_and_hash_verified=True, c04_manifest_and_publication_unchanged=True,
        manifest=dict(path=str(N/'RAW_RUNS_MANIFEST.json'), sha256=sha(N/'RAW_RUNS_MANIFEST.json')),
        part_index=dict(path=str(N/'RAW_RUNS_PART_INDEX.json'), sha256=sha(N/'RAW_RUNS_PART_INDEX.json')),
        owner_pid=2926581, owner_tool_session=16855, owner_lifecycle='Unchanged; parent retains polling supervision',
        training=False, export_or_witness_collection=False, model_get_or_release=False, test_accessed=False,
        remote_side_effects=False, refstatus='Independent final evidence reviewer and publisher remain parent-owned')
    save(A/'ARCHIVE_CONTINUATION_04.json', report)
    md = ('C05 archive continuation complete. The original validated 677004657-byte archive was retained unchanged; SHA-256 '+ARCHIVE_SHA+'.\n\n'
        'Executed only the exact frozen remaining pack tail, including the unmodified real 3A+256MiB+1GiB capacity predicate. '
        'Reused original semantic validation only for identical streamed archive/current bytes: 4026 members (2381 raw, 55 source, 1530 controls, 7 design, 53 receipts). '
        'All 81 staged/logical parts and ordered concatenation passed hash readback.\n\n'
        f'Temporarily relocated {len(reservation["moves"])} manifest/publication-verified owned C04 parts. '
        f'Actual workspace free before exact tail: {reservation["free_immediately_before_exact_tail"]} bytes; required: {required} bytes. '
        'Every original C04 part was restored and all 89 parts hash verified; original manifest/publication receipts unchanged. '
        f'Actual workspace free after restoration: {reservation["free_after_restore"]} bytes.\n\n'
        'Scientific source and R5 remained unchanged. One validate_attempt(require_engineering=True) used only existing owner status IPC. '
        'No pack rebuild, numeric semantic reparse, model get/release, training, export, witness collection or Test access. '
        'This is a mechanical continuation, not a rerun of the standard pack CLI. Original completion preserved as RAW_ARCHIVE_COMPLETION_03.json/md.\n')
    (A/'ARCHIVE_CONTINUATION_04.md').write_text(md)
    save(A/'RAW_ARCHIVE_COMPLETION.json', report)
    (A/'RAW_ARCHIVE_COMPLETION.md').write_text(md)
    milestone('PACKED_CONTINUATION_COMPLETE', archive_sha256=ARCHIVE_SHA, parts=81, c04_restored=True, test_accessed=False)


if __name__ == '__main__':
    main()
